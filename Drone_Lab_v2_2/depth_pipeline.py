"""Local Depth Anything V2 METRIC adapter. No network/downloads during startup.

Only the render thread captures RGB or updates textures. Inference runs in a
daemon worker with one pending frame (latest wins), never in the physics loop.
"""
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty, Full
from threading import Thread, Event, Lock
from time import monotonic
import json
import sys


@dataclass(frozen=True)
class DepthResult:
    sequence: int
    captured_at: float
    position: tuple
    yaw: float
    sectors: dict
    valid: bool
    latency_ms: float
    width: int
    height: int
    heatmap: bytes
    points: tuple = ()


def load_config(filename):
    path=Path(filename).resolve()
    config=json.loads(path.read_text(encoding='utf-8-sig'))
    for key in ('repo','checkpoint'):
        value=Path(config[key]).expanduser()
        config[key]=str(value if value.is_absolute() else (path.parent/value).resolve())
    if config.get('encoder','vits') not in ('vits','vitb','vitl'):
        raise ValueError('Depth encoder must be vits/vitb/vitl.')
    if not 140 <= int(config.get('input_size',266)) <= 700:
        raise ValueError('Depth input_size must be 140..700.')
    if not 0 < float(config.get('max_depth',20)) <= 200:
        raise ValueError('Invalid max_depth.')
    return config


def summarize_depth(depth, max_depth=20.):
    """5th percentile in L/C/R, central image band to reduce sky/ground hits.
    Missing/NaN/zero sectors are invalid, never interpreted as free space.
    Returns metric camera-axis distances, plus an RGB visualization.
    """
    import numpy as np
    array=np.asarray(depth,dtype=np.float32)
    if array.ndim != 2 or min(array.shape)<6:
        raise ValueError('Expected a nonempty HxW metric depth image.')
    h,w=array.shape
    roi=array[int(h*.30):max(int(h*.60),int(h*.30)+1)]
    sectors={}
    valid=True
    for i,name in enumerate(('LEFT','CENTER','RIGHT')):
        part=roi[:,i*w//3:(i+1)*w//3]
        good=part[np.isfinite(part)&(part>.05)&(part<=max_depth)]
        if len(good)<max(8,part.size*.5):
            sectors[name]=0.
            valid=False
        else: sectors[name]=float(np.percentile(good,5))
    # Fixed 0..8 m scale: warm near, cool far; invalid pixels black.
    good=np.isfinite(array)&(array>.05)
    t=np.clip(np.nan_to_num(array,nan=0,posinf=8,neginf=0)/8,0,1)
    rgb=np.stack((1-t,1-np.abs(t*2-1),t),axis=-1)
    rgb[~good]=0
    return sectors,valid,(rgb*255).astype(np.uint8)


def load_model(config):
    # Optional imports are confined to this path, so base simulator stays usable.
    import torch
    import cv2  # noqa: F401 - clear dependency error at initialization
    metric=Path(config['repo'])/'metric_depth'
    if not (metric/'depth_anything_v2'/'dpt.py').is_file():
        raise FileNotFoundError(f'Metric source missing: {metric}')
    checkpoint=Path(config['checkpoint'])
    if not checkpoint.is_file(): raise FileNotFoundError(f'Checkpoint missing: {checkpoint}')
    # Use metric_depth package, never the root relative-depth variant.
    existing=sys.modules.get('depth_anything_v2')
    locations=[str(getattr(existing,'__file__',''))]+[str(p) for p in getattr(existing,'__path__',[])]
    if existing is not None and not any(str(metric).lower() in p.lower() for p in locations):
        raise RuntimeError('Relative depth package already imported; restart with the metric adapter.')
    sys.path.insert(0,str(metric))
    from depth_anything_v2.dpt import DepthAnythingV2
    configs={'vits':dict(features=64,out_channels=[48,96,192,384]),
             'vitb':dict(features=128,out_channels=[96,192,384,768]),
             'vitl':dict(features=256,out_channels=[256,512,1024,1024])}
    encoder=config.get('encoder','vits')
    torch.set_num_threads(max(1,int(config.get('cpu_threads',2))))
    device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model=DepthAnythingV2(encoder=encoder,max_depth=float(config.get('max_depth',20)),**configs[encoder])
    state=torch.load(str(checkpoint),map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    return model.to(device).eval(),device


def depth_points(depth):
    """Sparse camera-local metric points for reactive steering, inside the FOV.
    No simulator obstacle positions/bounds are used here.
    """
    import numpy as np
    from math import tan, radians
    h,w=depth.shape
    tx=tan(radians(75/2)); ty=tx/(16/9)
    points=[]
    for row in range(int(h*.30),int(h*.60),max(1,h//24)):
        for column in range(0,w,max(1,w//48)):
            z=float(depth[row,column])
            if np.isfinite(z) and .05<z<=8:
                points.append(((2*(column+.5)/w-1)*tx*z,
                               (1-2*(row+.5)/h)*ty*z,z))
    return tuple(points)


class DepthWorker:
    def __init__(self, config_path, enabled=True):
        self.jobs=Queue(maxsize=1)
        self.stop=Event()
        self.lock=Lock()
        self.latest=None
        self.status='DISABLED' if not enabled else 'LOADING'
        self.error=''
        self.ready=False
        self.sequence=0
        self.thread=None
        self.config_path=str(config_path)
        if enabled:
            self.thread=Thread(target=self._run,name='metric-depth',daemon=True)
            self.thread.start()

    def submit(self, frame, position, yaw, captured_at=None):
        if not self.ready or self.stop.is_set(): return False
        self.sequence+=1
        job=(self.sequence,monotonic() if captured_at is None else captured_at,tuple(position),float(yaw),frame)
        try: self.jobs.put_nowait(job)
        except Full:
            try: self.jobs.get_nowait()
            except Empty: pass
            try: self.jobs.put_nowait(job)
            except Full: return False
        return True

    def snapshot(self):
        with self.lock: return self.latest

    def _run(self):
        try:
            config=load_config(self.config_path)
            model,device=load_model(config)
            import numpy as np
            self.ready=True
            self.status=f'READY {device.upper()}'
            while not self.stop.is_set():
                try: seq,stamp,position,yaw,frame=self.jobs.get(timeout=.2)
                except Empty: continue
                w,h,data=frame
                # Panda RAM RGB is bottom-up; upstream infer_image expects top-down BGR.
                bgr=np.frombuffer(data,dtype=np.uint8).reshape(h,w,3)[::-1,:,::-1].copy()
                started=monotonic()
                depth=model.infer_image(bgr,input_size=int(config.get('input_size',266)))
                sectors,valid,heatmap=summarize_depth(depth,float(config.get('max_depth',20)))
                latency=(monotonic()-started)*1000
                result=DepthResult(seq,stamp,position,yaw,sectors,valid,latency,w,h,heatmap.tobytes(),depth_points(depth))
                with self.lock: self.latest=result
                self.status=f'{device.upper()} {latency:.0f} ms'
        except Exception as exc:
            self.ready=False
            self.error=f'{type(exc).__name__}: {exc}'
            self.status='ERROR - SEE DEPTH SETUP'
            print(f'[Depth] {self.error}\nInstall requirements-depth.txt and check depth_config.json.')

    def close(self):
        self.stop.set()
        if self.thread: self.thread.join(timeout=.2)


def check_model(config_path, image_path, output_dir):
    """Real inference check: never replaced by a mock or a geometric depth image."""
    import cv2
    import numpy as np
    config=load_config(config_path)
    model,device=load_model(config)
    image=cv2.imread(str(image_path))
    if image is None: raise ValueError(f'Cannot read image: {image_path}')
    start=monotonic()
    depth=model.infer_image(image,input_size=int(config.get('input_size',266)))
    sectors,valid,heatmap=summarize_depth(depth,float(config.get('max_depth',20)))
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    np.save(out/'model_depth_m.npy',depth)
    cv2.imwrite(str(out/'model_depth.png'),heatmap[:,:,::-1])
    report=dict(device=device,elapsed_seconds=monotonic()-start,shape=list(depth.shape),sectors_m=sectors,valid=valid)
    (out/'depth_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return report
