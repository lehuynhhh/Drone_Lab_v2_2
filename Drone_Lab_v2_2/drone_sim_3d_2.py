"""Drone Lab v2.2 — Python 3.12+; pip install -r requirements-depth.txt
Run: python drone_sim_3d_2.py | Test: python drone_sim_3d_2.py --self-test
T/G takeoff/land; WASD translate; QE yaw; RF altitude; +/- speed;
C or 1..5 views; O avoidance; M manual/auto; P perception; V RGB/depth;
Space hold; Esc release fields. Map click selects a waypoint, GO starts flight.
Units: metres, seconds, degrees. X right, Y up, Z forward at yaw=0.
Built-in scene assets; optional local metric depth checkpoint in depth_config.json.
See README_v2_2.md for reactive navigation / model limits and setup.
"""
from dataclasses import dataclass, field
from collections import deque
from math import sin, cos, tan, atan2, radians, degrees, sqrt, inf, isfinite, exp
import argparse
from pathlib import Path
from time import monotonic

BRAKE, STOP, RANGE, HFOV = 1.5, 0.3, 8.0, 75.0
RADIUS, HALF_HEIGHT, REST = .78, .24, .26
ACCEL, DECEL, STEP = 4.0, 10.0, 1 / 120


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def approach(x, target, amount):
    return x + clamp(target - x, -amount, amount)


@dataclass
class V:
    x: float = 0
    y: float = 0
    z: float = 0

    def __add__(self, b): return V(self.x+b.x, self.y+b.y, self.z+b.z)
    def __sub__(self, b): return V(self.x-b.x, self.y-b.y, self.z-b.z)
    def __mul__(self, n): return V(self.x*n, self.y*n, self.z*n)
    def dot(self, b): return self.x*b.x+self.y*b.y+self.z*b.z
    def length(self): return sqrt(self.dot(self))
    def unit(self): return self * (1/max(self.length(), 1e-12))
    def tup(self): return self.x, self.y, self.z


def axes(yaw):
    a = radians(yaw)
    return V(sin(a), 0, cos(a)), V(cos(a), 0, -sin(a))


@dataclass
class Obstacle:
    id: int
    kind: str
    p: V
    size: V

    def bounds(self, padding=V()):
        h = self.size*.5 + padding
        return self.p-h, self.p+h


def ray_box(origin, direction, low, high, limit=inf):
    """Slab intersection; also used for continuous swept collision (limit=1)."""
    enter, leave = 0.0, limit
    for p, d, a, b in zip(origin.tup(), direction.tup(), low.tup(), high.tup()):
        if abs(d) < 1e-12:
            if p < a or p > b: return None
        else:
            t1, t2 = (a-p)/d, (b-p)/d
            enter, leave = max(enter, min(t1, t2)), min(leave, max(t1, t2))
            if enter > leave: return None
    return enter


def ray_surface(o, origin, direction):
    """Exact visual surface intersections: AABB, ellipsoid, elliptical cylinder."""
    if o.kind in ('Box', 'Wall'):
        return ray_box(origin, direction, *o.bounds(), RANGE)
    h, q = o.size*.5, origin-o.p
    p = V(q.x/h.x, q.y/h.y, q.z/h.z)
    d = V(direction.x/h.x, direction.y/h.y, direction.z/h.z)
    if o.kind == 'Sphere':
        a, b, c = d.dot(d), 2*p.dot(d), p.dot(p)-1
    else:
        a, b, c = d.x*d.x+d.z*d.z, 2*(p.x*d.x+p.z*d.z), p.x*p.x+p.z*p.z-1
    candidates = []
    disc = b*b-4*a*c
    if a > 1e-12 and disc >= 0:
        for t in ((-b-sqrt(disc))/(2*a), (-b+sqrt(disc))/(2*a)):
            if t >= 0 and (o.kind == 'Sphere' or abs(p.y+t*d.y) <= 1):
                candidates.append(t)
    if o.kind == 'Cylinder' and abs(d.y) > 1e-12:
        for y in (-1, 1):
            t = (y-p.y)/d.y
            if t >= 0 and (p.x+t*d.x)**2+(p.z+t*d.z)**2 <= 1:
                candidates.append(t)
    return min(candidates) if candidates else None


class ObstacleManager:
    def __init__(self):
        self.items, self.selected, self.next_id, self.revision = [], None, 1, 0

    def add(self, kind, x, z, sx, sy, sz, drone):
        values = (x, z, sx, sy, sz)
        if not all(isfinite(v) for v in values): raise ValueError('Use finite numbers.')
        if kind not in ('Box', 'Cylinder', 'Wall', 'Sphere'): raise ValueError('Unknown shape.')
        if not (.3 <= sx <= 15 and .3 <= sy <= 15 and .3 <= sz <= 15):
            raise ValueError('Sizes must be 0.3 to 15 m.')
        if abs(x)+sx/2 > 29 or abs(z)+sz/2 > 29:
            raise ValueError('Keep obstacle inside the 58 m map.')
        if len(self.items) >= 80: raise ValueError('Limit: 80 obstacles.')
        obj = Obstacle(self.next_id, kind, V(x, sy/2, z), V(sx, sy, sz))
        lo, hi = obj.bounds(V(RADIUS, HALF_HEIGHT, RADIUS))
        if all(a <= p <= b for p, a, b in zip(drone.p.tup(), lo.tup(), hi.tup())):
            raise ValueError('Obstacle would overlap the drone.')
        self.items.append(obj)
        self.next_id += 1
        self.selected = obj.id
        self.revision += 1
        return obj

    def delete(self):
        self.items = [o for o in self.items if o.id != self.selected]
        self.selected = None
        self.revision += 1

    def clear(self):
        self.items.clear()
        self.selected = None
        self.revision += 1

    def sweep(self, position, delta):
        """Conservative full-airframe AABB sweep, independent of sensor/avoidance."""
        fraction = 1.0
        for obj in self.items:
            t = ray_box(position, delta, *obj.bounds(V(RADIUS, HALF_HEIGHT, RADIUS)), 1)
            if t is not None: fraction = min(fraction, max(0, t-1e-5))
        return fraction


@dataclass
class SensorReading:
    distance: float = inf
    sector: str = '--'
    obstacle_id: int | None = None
    sectors: dict = field(default_factory=lambda: dict(LEFT=inf, CENTER=inf, RIGHT=inf))
    points: list = field(default_factory=list)


class FrontSensor:
    """Replace scan() with RGB->depth adapter returning SensorReading.

    Horizontal 75 degrees, 16:9 pinhole, range 8 m. Camera is level/gimballed.
    25x9 ray samples plus per-object targeted samples; occlusion uses nearest hit.
    Reported distance is ray range minus front-envelope offset (clearance).
    """
    def __init__(self):
        self.reading = SensorReading()
        self.tx, self.ty = tan(radians(HFOV/2)), tan(radians(HFOV/2))/(16/9)
        self.rays = [(self.tx*(2*x/24-1), self.ty*(2*y/8-1)) for x in range(25) for y in range(9)]

    def scan(self, drone, manager):
        f, r = axes(drone.yaw)
        origin = drone.p+f*.43
        candidates = [o for o in manager.items if (o.p-origin).length() <= RANGE+o.size.length()/2]
        samples = list(self.rays)
        for o in candidates:
            h = o.size*.5
            # Centre, closest point and corners help resolve thin geometry.
            lo, hi = o.bounds()
            points = [o.p, V(*(clamp(p, a, b) for p, a, b in zip(origin.tup(), lo.tup(), hi.tup())))]
            points += [o.p+V(x*h.x, y*h.y, z*h.z) for x in (-1,1) for y in (-1,1) for z in (-1,1)]
            for point in points:
                d = point-origin
                z = d.dot(f)
                if z > 1e-8:
                    x, y = d.dot(r)/z, d.y/z
                    if abs(x) <= self.tx and abs(y) <= self.ty: samples.append((x,y))
        out = SensorReading()
        for x, y in samples:
            direction = (f+r*x+V(0,y,0)).unit()
            nearest, hit_id = inf, None
            for o in candidates:
                t = ray_surface(o, origin, direction)
                if t is not None and 0 <= t <= RANGE and t < nearest:
                    nearest, hit_id = t, o.id
            if hit_id is None: continue
            out.points.append((origin+direction*nearest).tup())
            distance = max(0, nearest-(RADIUS-.43))
            sector = 'LEFT' if x < -self.tx/3 else 'RIGHT' if x > self.tx/3 else 'CENTER'
            out.sectors[sector] = min(out.sectors[sector], distance)
            if distance < out.distance:
                out.distance, out.sector, out.obstacle_id = distance, sector, hit_id
        self.reading = out
        return out


class PerceptionSensor:
    """GEOMETRY / HYBRID / DEPTH. Stale depth never means a clear path.

    HYBRID explicitly falls back to geometric sensor when depth is unavailable.
    DEPTH uses model distances only, and holds motion on invalid/stale frames.
    Local steering sees these readings only; the final collision guard is geometric.
    """
    sources=('HYBRID','DEPTH','GEOMETRY')

    def __init__(self, worker, source='HYBRID'):
        self.worker,self.source=worker,source
        self.geometry=FrontSensor()
        self.reading=SensorReading()
        self.hold=False
        self.note='WAITING'
        self.last_status=''

    def cycle(self):
        self.source=self.sources[(self.sources.index(self.source)+1)%len(self.sources)]

    def scan(self, drone, manager):
        self.hold=False
        geometric=self.geometry.scan(drone,manager) if self.source!='DEPTH' else SensorReading()
        if self.source=='GEOMETRY':
            self.note='GEOMETRY'
            self.reading=geometric
            return self.reading
        result=self.worker.snapshot()
        reason='WAITING' if not self.worker.error else 'MODEL ERROR'
        valid=False
        travel=0
        if result is not None:
            travel=(drone.p-V(*result.position)).length()
            age=monotonic()-result.captured_at
            turn=abs((drone.yaw-result.yaw+180)%360-180)
            valid=result.valid and 0<=age<=2.0 and travel<=1.0 and turn<=12
            reason='INVALID' if not result.valid else 'STALE / POSE CHANGED'
        if not valid:
            self.hold=self.source=='DEPTH'
            self.note=f'DEPTH HOLD: {reason}' if self.hold else f'GEO FALLBACK: {reason}'
            self.reading=SensorReading(distance=0,sector='--') if self.hold else geometric
            return self.reading
        self.note=f'{self.source} {result.latency_ms:.0f} ms'
        out=SensorReading()
        f,r=axes(result.yaw)
        origin=V(*result.position)+f*.43
        out.points=[(origin+r*x+V(0,y,0)+f*z).tup() for x,y,z in getattr(result,'points',())]
        if self.source=='HYBRID': out.points.extend(geometric.points)
        for name in out.sectors:
            value=max(0,result.sectors[name]-(RADIUS-.43)-travel)
            if value>RANGE: value=inf
            out.sectors[name]=min(value,geometric.sectors[name]) if self.source=='HYBRID' else value
        out.sector=min(out.sectors,key=out.sectors.get)
        out.distance=out.sectors[out.sector]
        if out.distance==inf: out.sector='--'
        self.reading=out
        return out


class Drone:
    def __init__(self):
        self.p, self.v = V(0, REST, 0), V()
        self.yaw, self.target_altitude, self.speed = 0.0, 2.0, 2.0
        self.airborne, self.landing = False, False
        self.path = deque(maxlen=800)

    def takeoff(self):
        self.airborne, self.landing = True, False
        self.target_altitude = max(.8, self.target_altitude)


def map_coordinates(mouse_x, mouse_y, center_x, center_y, size):
    """UI coordinates share mouse/window aspect ratio; no raycast dependency."""
    x, y = (mouse_x-center_x)/size, (mouse_y-center_y)/size
    if not (-.5 <= x <= .5 and -.5 <= y <= .5): return None
    return clamp(x*60,-28,28), clamp(y*60,-28,28)


class ReactiveNavigator:
    """Online local steering from front-camera observations, never a world map.

    No manager/obstacle IDs/bounds are available to this class. It keeps a short
    memory of observed surface points so a turn does not immediately forget a wall.
    Straight flight -> local avoidance -> rejoin original start-goal line.
    This is a local controller, not a complete maze solver or global planner.
    """
    def __init__(self):
        self.start,self.goal,self.rejoin=None,None,None
        self.state='DIRECT'
        self.memory={}
        self.clock,self.next_decision=0.,0.
        self.side=-1
        self.heading=0.
        self.command_speed=0.
        self.stuck=0.
        self.last_progress=None
        self.events=[]

    def begin(self,start,goal):
        self.__init__()
        self.start=V(start.x,goal.y,start.z)
        self.goal=V(*goal.tup())
        self.direction=V(goal.x-start.x,0,goal.z-start.z).unit()
        self.line_right=V(self.direction.z,0,-self.direction.x)
        self.length=(goal-self.start).length()

    def change(self,state,position):
        if state!=self.state:
            self.events.append((state,V(*position.tup())))
            self.events=self.events[-100:]
        self.state=state

    def cross_track(self,p):
        if self.length<1e-6: return 0.
        delta=p-self.start
        return (V(delta.x,0,delta.z)-self.direction*delta.dot(self.direction)).length()

    def rejoin_point(self,p):
        progress=clamp((p-self.start).dot(self.direction)+3.,0,self.length)
        return self.start+self.direction*progress

    def remember(self,reading,drone):
        for point in reading.points:
            p=V(*point)
            if abs(p.y-drone.p.y)>HALF_HEIGHT+.22: continue
            key=(round(p.x/.25),round(p.z/.25))
            self.memory[key]=(p,self.clock)
        self.memory={k:v for k,v in self.memory.items()
                     if self.clock-v[1]<8 and (v[0]-drone.p).length()<12}

    def blocked(self,p,heading,length):
        f=axes(heading)[0]
        endpoint=p+f*length
        if abs(endpoint.x)>28.5 or abs(endpoint.z)>28.5: return True
        margin=RADIUS+STOP+.55
        for obstacle,_ in self.memory.values():
            offset=V(obstacle.x-p.x,0,obstacle.z-p.z)
            projection=offset.dot(f)
            # Allow backing away from a point already inside the inflated margin.
            if projection<=0: continue
            nearest=(offset-f*clamp(projection,0,length)).length()
            if nearest<margin and nearest<offset.length()-.015: return True
        return False

    def command(self,drone,reading,dt,enabled=True):
        self.clock+=dt
        if self.goal is None: return drone.yaw,0.
        if not enabled:
            self.memory.clear(); self.rejoin=None
            self.change('DIRECT',drone.p)
        if self.clock<self.next_decision:
            return self.heading,self.command_speed
        self.next_decision=self.clock+.1
        self.remember(reading,drone)
        if self.last_progress is None or (drone.p-self.last_progress).length()>.4:
            self.last_progress=V(*drone.p.tup()); self.stuck=self.clock
        if enabled and self.clock-self.stuck>25 and self.state!='DIRECT':
            self.change('BLOCKED',drone.p)
            self.command_speed=0
            return drone.yaw,0.
        if self.state=='BLOCKED': return drone.yaw,0.

        # Detect first; do not change course merely because a future obstacle exists.
        if enabled and self.state=='DIRECT' and reading.distance<=BRAKE:
            left,right=reading.sectors['LEFT'],reading.sectors['RIGHT']
            self.side=-1 if left>=right else 1
            self.change('AVOID',drone.p)

        target=self.goal
        if self.state=='REJOIN':
            target=self.rejoin
            if (target-drone.p).length()<.16 and drone.v.length()<.3:
                self.change('DIRECT',drone.p)
                self.rejoin=None
                target=self.goal
        elif self.state=='AVOID': target=self.rejoin_point(drone.p)
        delta=target-drone.p
        distance=V(delta.x,0,delta.z).length()
        nominal=degrees(atan2(delta.x,delta.z)) if distance>.02 else drone.yaw
        lookahead=min(4.5,max(distance,.1))
        blocked=enabled and self.state!='DIRECT' and self.blocked(drone.p,nominal,lookahead)
        if self.state=='REJOIN' and (blocked or reading.distance<STOP+.1):
            self.change('AVOID',drone.p)
            self.rejoin=None
            blocked=True
        if self.state=='AVOID' and not blocked:
            self.rejoin=target
            self.change('REJOIN',drone.p)
        if self.state=='AVOID':
            choices=[]
            for offset in range(-180,181,10):
                heading=nominal+offset
                # Commit to the selected side until the rejoin corridor is clear.
                # Otherwise a wide wall makes the nominal bearing flip sides.
                if axes(heading)[0].dot(self.line_right)*self.side<-.05: continue
                if self.blocked(drone.p,heading,2.7): continue
                turn=abs((heading-drone.yaw+180)%360-180)
                score=abs(offset)+.18*turn+(45 if offset*self.side<0 else 0)
                choices.append((score,heading))
            if choices:
                self.heading=min(choices)[1]
                self.command_speed=min(drone.speed,.9)
            else:
                # Stop and physically turn the front camera to observe another direction.
                self.heading=drone.yaw+self.side*35
                self.command_speed=0.
        else:
            self.heading=nominal
            self.command_speed=min(drone.speed if self.state=='DIRECT' else 1.1,distance*1.6)
        return self.heading,self.command_speed


class FlightController:
    def __init__(self, drone, obstacles, sensor):
        self.drone, self.obstacles, self.sensor = drone, obstacles, sensor
        self.mode, self.avoidance, self.state = 'MANUAL', True, 'NORMAL'
        self.waypoints, self.mission = deque(), 'READY'
        self.collision = False
        self.navigator = ReactiveNavigator()

    def manual(self):
        self.mode, self.mission = 'MANUAL', 'HOLD'
        self.waypoints.clear()
        self.navigator = ReactiveNavigator()

    def land(self):
        self.manual()
        self.drone.landing = self.drone.airborne
        self.drone.target_altitude = REST

    def go(self, target):
        if not all(isfinite(v) for v in target.tup()): raise ValueError('Invalid waypoint.')
        if abs(target.x) > 28 or abs(target.z) > 28 or not .8 <= target.y <= 20:
            raise ValueError('Waypoint: X/Z +/-28, altitude 0.8..20 m.')
        self.waypoints = deque([target])
        self.mode, self.mission = 'AUTO', 'DIRECT'
        self.drone.target_altitude = target.y
        self.drone.takeoff()
        self.navigator.begin(self.drone.p,target)

    def tick(self, dt, forward=0, strafe=0, yaw=0):
        d = self.drone
        self.collision = False
        if not d.airborne:
            d.v = V()
            self.sensor.scan(d, self.obstacles)
            self.state = 'DEPTH HOLD' if getattr(self.sensor,'hold',False) else 'NORMAL'
            return
        desired = V()
        if self.mode == 'AUTO' and self.waypoints:
            final = self.waypoints[0]
            delta = final-d.p
            d.target_altitude = final.y
            if abs(d.p.y-final.y)<.15:
                nav_dt=0 if getattr(self.sensor,'hold',False) else dt
                goal_yaw,speed=self.navigator.command(d,self.sensor.reading,nav_dt,self.avoidance)
                error = (goal_yaw-d.yaw+180)%360-180
                # Brake before turning away. The camera must face the new direction
                # before moving, since there are no side/rear cameras.
                if abs(error)<8 or V(d.v.x,0,d.v.z).length()<.08:
                    d.yaw += clamp(error,-90*dt,90*dt)
                if abs(error)<4:
                    desired=axes(d.yaw)[0]*speed
            if delta.length() < .15 and d.v.length() < .2:
                self.waypoints.popleft()
                self.mission = 'ARRIVED' if not self.waypoints else 'DIRECT'
        elif self.mode == 'MANUAL' and not d.landing:
            d.yaw = (d.yaw+yaw*90*dt)%360
            f, r = axes(d.yaw)
            desired = f*forward+r*(strafe*.8)
            if desired.length() > 1: desired = desired.unit()
            desired = desired*d.speed
        if d.landing: desired = V()
        f, r = axes(d.yaw)
        reading = self.sensor.scan(d, self.obstacles)
        if self.avoidance and getattr(self.sensor,'hold',False):
            d.v=V()
            self.state='DEPTH HOLD'
            if self.mode=='AUTO': self.mission='SENSOR HOLD'
            return
        gap = reading.distance
        vf, vr = d.v.dot(f), d.v.dot(r)
        df, dr = desired.dot(f), desired.dot(r)
        self.state = 'WARNING' if gap < RANGE else 'NORMAL'
        if self.avoidance and gap <= BRAKE and (df > 0 or vf > .001):
            remaining = max(0, gap-STOP)
            limit = min(d.speed, sqrt(2*DECEL*remaining), remaining*3)
            df = min(df, limit)
            self.state = 'STOPPED' if remaining <= .02 else 'BRAKING'
            if self.mode == 'AUTO': self.mission = self.navigator.state
        elif self.mode == 'AUTO' and self.waypoints:
            self.mission = self.navigator.state
        vf = approach(vf, df, (ACCEL if abs(df)>abs(vf) and df*vf>=0 else DECEL)*dt)
        vr = approach(vr, dr, (ACCEL if abs(dr)>abs(vr) and dr*vr>=0 else DECEL)*dt)
        # Safety envelope prevents discrete steps from crossing the 0.5 m threshold.
        if self.avoidance and gap <= BRAKE and vf > 0:
            vf = min(vf, max(0, gap-STOP)/dt)
        target_y = REST if d.landing else clamp(d.target_altitude, .8, 20)
        vy = approach(d.v.y, clamp((target_y-d.p.y)*2.5, -2, 2), 4*dt)
        if abs(vy*dt) > abs(target_y-d.p.y): vy = (target_y-d.p.y)/dt
        d.v = f*vf+r*vr+V(0,vy,0)
        delta = d.v*dt
        fraction = self.obstacles.sweep(d.p, delta) if delta.length() > 1e-10 else 1
        d.p = d.p+delta*fraction
        if fraction < 1:
            d.v = V()
            self.collision = True
            if self.mode == 'AUTO': self.mission = 'BLOCKED'
        d.p.x, d.p.z = clamp(d.p.x,-29,29), clamp(d.p.z,-29,29)
        d.p.y = max(REST, d.p.y)
        if d.landing and d.p.y <= REST+.005:
            d.p.y, d.v, d.airborne, d.landing = REST, V(), False, False
        if not d.path or (d.p-d.path[-1]).length() > .15: d.path.append(V(*d.p.tup()))


# View layer uses composition; the flight/sensor classes above need no graphics runtime.
u = None
PALETTE = {'Box': '#fa916b', 'Wall': '#718cff', 'Cylinder': '#eab95c', 'Sphere': '#a98fee'}


def col(hex_string): return u.color.hex(hex_string)


def line_entity(vertices, parent=None, tint='#38505d', thickness=1):
    return u.Entity(parent=parent or u.scene, model=u.Mesh(vertices=vertices, triangles=[(i,i+1) for i in range(0,len(vertices)-1,2)], mode='line', thickness=thickness), color=col(tint))


def refresh_lines(entity, vertices):
    entity.model.vertices = vertices or [(0,0,0),(0,0,0)]
    entity.model.triangles = [(i,i+1) for i in range(0,len(entity.model.vertices)-1,2)]
    entity.model.generate()


class WorldView:
    def __init__(self, manager):
        from ursina.shaders import lit_with_shadows_shader
        self.shader = lit_with_shadows_shader
        self.manager, self.entities, self.revision = manager, {}, -1
        u.Entity(model='plane', scale=60, color=col('#22323d'), shader=self.shader)
        vertices = []
        for i in range(-30,31):
            vertices.extend([(i,.012,-30),(i,.012,30),(-30,.012,i),(30,.012,i)])
        line_entity(vertices)
        line_entity([(-30,.025,0),(30,.025,0)], tint='#bf6871', thickness=2)
        line_entity([(0,.025,-30),(0,.025,30)], tint='#55bdae', thickness=2)
        for i in range(-25,30,5):
            if i:
                u.Text(str(i), parent=u.scene, position=(i,.04,.4), rotation=(90,0,0), scale=1.8, color=col('#a4bac8'))
                u.Text(str(i), parent=u.scene, position=(.4,.04,i), rotation=(90,0,0), scale=1.8, color=col('#a4bac8'))
        u.Entity(model='cube', position=(0,.016,0), scale=(2.6,.025,2.6), color=col('#2a655f'))
        u.Text('H', parent=u.scene, position=(-.4,.04,.5), rotation=(90,0,0), scale=12, color=col('#8eeee0'))
        light = u.DirectionalLight(shadows=True, shadow_map_resolution=(1024,1024))
        light.look_at(u.Vec3(1,-2,1))
        u.AmbientLight(color=u.color.rgba32(110,125,145,255))
        self.path = line_entity([(0,0,0),(0,0,0)], tint='#4de0c7', thickness=2)
        self.target = u.Entity(model='sphere', scale=.25, color=col('#f8df89'), enabled=False)
        self.reference_line = line_entity([(0,0,0),(0,0,0)],tint='#ffcc70',thickness=2)

    def sync(self, drone, controller):
        if self.revision != self.manager.revision:
            for e in self.entities.values(): u.destroy(e)
            self.entities.clear()
            for o in self.manager.items:
                model = 'sphere' if o.kind == 'Sphere' else cylinder_mesh() if o.kind == 'Cylinder' else 'cube'
                e = u.Entity(model=model, position=o.p.tup(), scale=o.size.tup(), color=col(PALETTE[o.kind]), shader=self.shader, collider='box')
                e.obstacle_id = o.id
                self.entities[o.id] = e
            self.revision = self.manager.revision
        for o in self.manager.items:
            self.entities[o.id].color = col('#ffe9aa' if o.id == self.manager.selected else PALETTE[o.kind])
        points = [p.tup() for p in drone.path]
        refresh_lines(self.path, [v for a,b in zip(points,points[1:]) for v in (a,b)])
        self.target.enabled = bool(controller.waypoints)
        if controller.waypoints: self.target.position = controller.waypoints[0].tup()
        nav=controller.navigator
        reference=[nav.start,nav.goal] if controller.waypoints and nav.start else []
        refresh_lines(self.reference_line,[v.tup() for v in reference])


def cylinder_mesh():
    # Centred unit cylinder; avoids engine-version-dependent primitive origins.
    vertices, triangles = [], []
    for i in range(24):
        a,b = i*2*3.141592653589793/24, (i+1)*2*3.141592653589793/24
        k = len(vertices)
        vertices += [(sin(a)*.5,-.5,cos(a)*.5),(sin(b)*.5,-.5,cos(b)*.5),
                     (sin(b)*.5,.5,cos(b)*.5),(sin(a)*.5,.5,cos(a)*.5), (0,-.5,0),(0,.5,0)]
        triangles += [(k,k+1,k+2),(k,k+2,k+3),(k+4,k+1,k),(k+5,k+3,k+2)]
    flat, normals = [], []
    for a,b,c in triangles:
        va,vb,vc = (u.Vec3(*vertices[i]) for i in (a,b,c))
        normal = (vb-va).cross(vc-va).normalized()
        flat.extend((va,vb,vc))
        normals.extend((normal,normal,normal))
    return u.Mesh(vertices=flat, normals=normals)


class DroneView:
    def __init__(self, drone, shader):
        self.drone = drone
        self.root = u.Entity()
        self.body = u.Entity(parent=self.root)
        self.props = []
        def part(model, scale, position=(0,0,0), tint='#d5e9ee', parent=None, **kw):
            return u.Entity(parent=parent or self.body, model=model, scale=scale, position=position, color=col(tint), shader=shader, **kw)
        part('cube',(.38,.18,.52))
        part('cube',(.24,.08,.3),(0,.12,-.03),'#2c4654')
        for x,z in ((-.43,-.43),(.43,-.43),(-.43,.43),(.43,.43)):
            part('cube',(.61,.065,.075),(x/2,0,z/2),'#3b5361',rotation_y=-degrees(atan2(z,x)))
            part(cylinder_mesh(),(.16,.15,.16),(x,.04,z),'#3ee0c5' if z>0 else '#fa916b')
            hub = u.Entity(parent=self.body, position=(x,.14,z))
            part('cube',(.55,.015,.045),tint='#9bb1bf',parent=hub)
            part('cube',(.045,.015,.55),tint='#9bb1bf',parent=hub)
            self.props.append(hub)
            part('cube',(.05,.18,.05),(x,-.12,z),'#17232e')
        part('sphere',(.13,.13,.12),(0,0,.29),'#102530')
        part('cube',(.06,.05,.025),(0,0,.36),'#5ef4e1')
        self.mount = u.Entity(parent=self.root, position=(0,0,.43))

    def sync(self, dt):
        d = self.drone
        self.root.position, self.root.rotation_y = d.p.tup(), d.yaw
        f,r = axes(d.yaw)
        blend = 1-exp(-8*dt)
        self.body.rotation_x += (d.v.dot(f)*2.2-self.body.rotation_x)*blend
        self.body.rotation_z += (-d.v.dot(r)*2.2-self.body.rotation_z)*blend
        for i,p in enumerate(self.props): p.rotation_y += (1 if i%2 else -1)*(1600 if d.airborne else 0)*dt


class CameraRig:
    names = ['CHASE','TOP','SIDE','FPV','ORBIT']
    def __init__(self, view):
        self.view, self.index, self.angle, self.radius = view, 0, 30., 9.
        u.camera.clip_plane_far = 180
        self.set(0)

    def set(self, index):
        self.index = index%5
        u.camera.parent = u.scene
        u.camera.orthographic = self.index == 1
        u.camera.fov = 26 if self.index == 1 else HFOV if self.index == 3 else 65
        self.sync(1, True)

    def sync(self, dt, snap=False):
        p = self.view.drone.p
        f,r = axes(self.view.drone.yaw)
        if self.index == 3:
            u.camera.position = (p+f*.43).tup()
            u.camera.rotation = (0,self.view.drone.yaw,0)
            return
        if self.index == 0: eye = p-f*7+V(0,3.4,0)
        elif self.index == 1: eye = p+V(0,25,-.001)
        elif self.index == 2: eye = p+r*8+V(0,2,0)
        else:
            if u.held_keys['right mouse']: self.angle += u.mouse.velocity[0]*180
            a = radians(self.angle)
            eye = p+V(sin(a)*self.radius,self.radius*.55,-cos(a)*self.radius)
        t = 1 if snap else 1-exp(-6*dt)
        u.camera.position = u.lerp(u.camera.position,u.Vec3(*eye.tup()),t)
        u.camera.look_at(u.Vec3(*p.tup()))
        u.camera.rotation_z = 0


class SimulatedFrontCamera:
    """Independent 384x216 GPU preview. capture_rgb() is the future depth input."""
    def __init__(self, mount, parent, enabled=True):
        self.buffer, self.cam, self.error = None, None, ''
        self.quad,self.rgb_texture,self.depth_texture=None,None,None
        self.show_depth=False
        self.depth_sequence=-1
        if not enabled:
            self.error = 'Preview disabled (--no-preview)'
            return
        try:
            from panda3d.core import PerspectiveLens
            self.buffer = u.application.base.win.makeTextureBuffer('front-rgb',384,216)
            if self.buffer is None: raise RuntimeError('No render buffer available')
            self.buffer.setClearColor(col('#16222f'))
            lens = PerspectiveLens()
            lens.setAspectRatio(16/9)
            lens.setFov(HFOV, degrees(2*atan2(tan(radians(HFOV/2)),16/9)))
            lens.setNearFar(.05,100)
            self.cam = u.application.base.makeCamera(self.buffer,lens=lens,scene=u.scene)
            for region in self.buffer.getDisplayRegions():
                region.setClearColor(col('#16222f'))
                region.setClearColorActive(True)
            self.cam.reparentTo(mount)
            self.cam.setPos(0,0,0)
            self.cam.setHpr(0,0,0)
            self.rgb_texture=u.Texture(self.buffer.getTexture())
            self.quad=u.Entity(parent=parent,model='quad',texture=self.rgb_texture,scale=(.302,.170),y=-.010)
        except Exception as exc:
            self.error = f'Preview unavailable: {exc}'
            self.close()

    def update_depth(self,result):
        if result and result.sequence!=self.depth_sequence:
            from PIL import Image
            self.depth_texture=u.Texture(Image.frombytes('RGB',(result.width,result.height),result.heatmap))
            self.depth_sequence=result.sequence
        if self.quad is not None:
            self.quad.texture=self.depth_texture if self.show_depth and self.depth_texture else self.rgb_texture

    def capture_rgb(self):
        """Return (width,height,RGB bytes), bottom-up rows. Call at <=10 Hz.
        GPU readback is intentionally opt-in; no model/network/download is used.
        """
        if self.buffer is None: return None
        tex = self.buffer.getTexture()
        u.application.base.graphicsEngine.extractTextureData(tex,u.application.base.win.getGsg())
        return tex.getXSize(),tex.getYSize(),bytes(tex.getRamImageAs('RGB'))

    def close(self):
        if self.cam is not None:
            self.cam.removeNode()
            self.cam = None
        if self.buffer is not None:
            u.application.base.graphicsEngine.removeWindow(self.buffer)
            self.buffer = None


class Minimap:
    def __init__(self, parent, controller):
        self.controller, self.size = controller, .302
        self.root = u.Entity(parent=parent,y=-.016)
        self.bg = u.Entity(parent=self.root,model='quad',scale=self.size,color=col('#12212d'))
        self.layer(self.bg,10)
        vertices=[]
        for i in range(-30,31,5):
            q=i/60*self.size
            vertices += [(q,-self.size/2,-.01),(q,self.size/2,-.01),(-self.size/2,q,-.01),(self.size/2,q,-.01)]
        self.layer(line_entity(vertices,self.root),11)
        self.drone = u.Entity(parent=self.root,model=u.Mesh(vertices=[(0,.008,0),(-.005,-.005,0),(.005,-.005,0)],triangles=[(0,1,2)]),color=col('#58f6d5'),z=-.05,double_sided=True)
        self.target = u.Entity(parent=self.root,model='quad',scale=.007,color=col('#ffe291'),z=-.04,enabled=False)
        self.pending=None
        self.pending_marker=u.Entity(parent=self.root,model='quad',scale=.01,color=col('#ffffff'),z=-.06,enabled=False)
        self.reference_line=line_entity([(0,0,0),(0,0,0)],self.root,'#ffcc70',2)
        self.path = line_entity([(0,0,-.02),(0,0,-.02)],self.root,'#4fae9f',2)
        self.fov = line_entity([(0,0,-.03),(0,0,-.03)],self.root,'#58f6d5')
        for entity,order in ((self.path,13),(self.fov,14),(self.target,15),(self.drone,16)):
            self.layer(entity,order)
        self.layer(self.pending_marker,17)
        self.layer(self.reference_line,12)
        self.markers, self.revision, self.on_pick = [], -1, None

    @staticmethod
    def layer(entity, order):
        entity.set_bin('fixed',order)
        entity.set_depth_test(False)
        entity.set_depth_write(False)

    def project(self,p): return (clamp(p.x,-30,30)/60*self.size,clamp(p.z,-30,30)/60*self.size,-.02)

    def click(self,mouse_x=None,mouse_y=None):
        center=self.root.get_position(relative_to=u.camera.ui)
        xy=map_coordinates(u.mouse.x if mouse_x is None else mouse_x,
                           u.mouse.y if mouse_y is None else mouse_y,
                           center.x,center.y,self.size)
        if xy is None: return False
        if self.on_pick: self.on_pick(*xy)
        return True

    def select(self,x,z):
        self.pending=V(x,0,z)
        self.pending_marker.enabled=True
        self.pending_marker.position=(*self.project(self.pending)[:2],-.06)

    def sync(self):
        c,d,m = self.controller,self.controller.drone,self.controller.obstacles
        if self.revision != m.revision:
            for marker in self.markers: u.destroy(marker)
            self.markers=[]
            for o in m.items:
                e=u.Entity(parent=self.root,model='circle' if o.kind in ('Cylinder','Sphere') else 'quad',position=self.project(o.p),scale=(o.size.x/60*self.size,o.size.z/60*self.size),color=col(PALETTE[o.kind]))
                self.layer(e,12)
                self.markers.append(e)
            self.revision=m.revision
        self.drone.position = (*self.project(d.p)[:2],-.05)
        self.drone.rotation_z = -d.yaw
        ps=[self.project(p) for p in d.path]
        refresh_lines(self.path,[v for a,b in zip(ps,ps[1:]) for v in (a,b)])
        origin=self.project(d.p)
        ends=[d.p+axes(d.yaw+a)[0]*RANGE for a in (-HFOV/2,HFOV/2)]
        refresh_lines(self.fov,[origin,self.project(ends[0]),origin,self.project(ends[1])])
        self.target.enabled=bool(c.waypoints)
        if c.waypoints: self.target.position=(*self.project(c.waypoints[0])[:2],-.04)
        nav=c.navigator
        reference=[nav.start,nav.goal] if c.waypoints and nav.start else []
        refresh_lines(self.reference_line,[self.project(v) for v in reference])


class FlightUI:
    def __init__(self,controller,rig,view,no_preview=False):
        self.c,self.rig = controller,rig
        self.fields=[]
        self.left=u.Entity(parent=u.camera.ui)
        self.right=u.Entity(parent=u.camera.ui)
        self.title=u.Text('DRONE LAB  /  02.2',parent=u.camera.ui,y=.478,scale=1.1,color=col('#6ce6d1'))
        self.hint=u.Text('WASD move  /  QE yaw  /  RF altitude  /  C view  /  SPACE hold',parent=u.camera.ui,y=-.465,scale=.72,color=col('#b9cbd6'))
        self.message=u.Text('Ready. T to take off.',parent=u.camera.ui,y=-.428,scale=.8,color=col('#f2d891'))
        self.panel(self.left,.26,.36,'01  FLIGHT')
        self.flight=self.text(self.left,-.15,.386,'',.82)
        self.button(self.left,-.081,.153,'TAKEOFF',controller.drone.takeoff,.143)
        self.button(self.left,.081,.153,'LAND',controller.land,.143)
        self.button(self.left,-.116,.100,'ALT -',lambda:self.trim('alt',-.5),.071)
        self.button(self.left,-.038,.100,'ALT +',lambda:self.trim('alt',.5),.071)
        self.button(self.left,.040,.100,'SPD -',lambda:self.trim('speed',-.5),.071)
        self.button(self.left,.118,.100,'SPD +',lambda:self.trim('speed',.5),.071)
        self.panel(self.left,-.025,.17,'02  SENSOR')
        self.sensor=self.text(self.left,-.15,.018,'',.75)
        self.avoid=self.button(self.left,0,-.095,'AVOIDANCE ON',self.toggle_avoid,.302)
        self.panel(self.left,-.277,.294,'03  MISSION')
        self.mode=self.text(self.left,-.15,-.180,'',.8)
        self.button(self.left,-.08,-.222,'MANUAL',controller.manual,.142)
        self.button(self.left,.08,-.222,'AUTO',self.auto,.142)
        self.wx=self.field(self.left,-.104,-.29,'X','5',.09)
        self.wy=self.field(self.left,0,-.29,'ALT','2',.09)
        self.wz=self.field(self.left,.104,-.29,'Z','10',.09)
        self.button(self.left,0,-.343,'GO TO WAYPOINT',self.go,.302)
        self.text(self.left,-.15,-.378,'DIRECT -> AVOID -> REJOIN\nGold = original straight line',.65)
        self.panel(self.right,.265,.35,'04  MAP  /  60 x 60 m')
        map_root=u.Entity(parent=self.right,y=.264)
        self.minimap=Minimap(map_root,controller)
        self.minimap.on_pick=self.map_pick
        self.panel(self.right,-.038,.23,'05  FRONT RGB  /  75 DEG')
        preview_root=u.Entity(parent=self.right,y=-.048)
        self.preview=SimulatedFrontCamera(view.mount,preview_root,not no_preview)
        if self.preview.error: self.text(preview_root,-.145,.04,'RGB preview unavailable\nSensor remains active.',.7)
        self.panel(self.right,-.304,.283,'06  OBSTACLE EDITOR')
        self.ox=self.field(self.right,-.104,-.239,'X','0',.09)
        self.oz=self.field(self.right,0,-.239,'Z','8',.09)
        self.sy=self.field(self.right,.104,-.239,'HEIGHT','3',.09)
        self.sx=self.field(self.right,-.104,-.292,'WIDTH','2',.09)
        self.sz=self.field(self.right,0,-.292,'DEPTH','2',.09)
        self.button(self.right,.104,-.292,'NEXT',self.next_obstacle,.09)
        for i,name in enumerate(('Box','Cylinder','Wall','Sphere')):
            self.button(self.right,-.116+i*.078,-.344,name,lambda kind=name:self.add(kind),.073)
        self.button(self.right,-.08,-.393,'DELETE SELECTED',self.delete,.145)
        self.button(self.right,.08,-.393,'CLEAR ALL',self.clear,.145)
        self.selection=self.text(self.right,-.15,-.422,'',.62)
        self.viewbar=u.Entity(parent=u.camera.ui,y=.448)
        for i,name in enumerate(rig.names):
            self.button(self.viewbar,(i-2)*.114,0,f'{i+1} {name}',lambda k=i:rig.set(k),.106)
        self.source_button=self.button(u.camera.ui,-.143,.395,'P: HYBRID',controller.sensor.cycle,.264)
        self.depth_button=self.button(u.camera.ui,.143,.395,'V: RGB / DEPTH',self.toggle_depth,.264)
        self.depth_status=u.Text('',parent=u.camera.ui,position=(-.275,.362,-.1),scale=.66,color=col('#a4cecf'))
        self.layout()

    def text(self,parent,x,y,text,scale=.8):
        return u.Text(text,parent=parent,position=(x,y,-.1),scale=scale,color=col('#cfdee7'))

    def panel(self,parent,y,height,title):
        u.Entity(parent=parent,model='quad',position=(0,y,.1),scale=(.334,height),color=col('#15232f'))
        self.text(parent,-.15,y+height/2-.018,title,.76).color=col('#65d8c6')

    def button(self,parent,x,y,text,callback,width):
        b=u.Button(parent=parent,text=text,position=(x,y,-.12),scale=(width,.036),color=col('#294252'),highlight_color=col('#3e6c7c'),pressed_color=col('#397e79'))
        b.text_entity.scale *= .67
        b.on_click=callback
        return b

    def field(self,parent,x,y,label,default,width):
        self.text(parent,x-width/2,y+.028,label,.59)
        f=u.InputField(parent=parent,default_value=default,position=(x,y,-.15),scale=(width,.032),color=col('#233947'),character_limit=8)
        f.text_field.scale *= .68
        self.fields.append(f)
        return f

    def focused(self): return any(f.active for f in self.fields)
    def blur(self):
        for f in self.fields: f.active=False
    def report(self,text): self.message.text=text

    def trim(self,kind,amount):
        d=self.c.drone
        if kind=='speed': d.speed=clamp(d.speed+amount,.5,5)
        else: d.target_altitude=clamp(d.target_altitude+amount,.8,20)

    def toggle_avoid(self): self.c.avoidance=not self.c.avoidance
    def toggle_depth(self):
        self.preview.show_depth=not self.preview.show_depth
        if self.preview.show_depth and self.preview.depth_texture is None:
            self.report('Depth image not ready. See model status / depth_config.json.')
    def auto(self):
        self.c.mode,self.c.mission='AUTO','HOLD' if not self.c.waypoints else 'EN ROUTE'
    def map_pick(self,x,z):
        self.blur()
        self.wx.text,self.wz.text=f'{x:.1f}',f'{z:.1f}'
        self.minimap.select(x,z)
        self.report('Waypoint coordinates selected. Press GO TO WAYPOINT.')
    def go(self):
        try:
            self.c.go(V(float(self.wx.text),float(self.wy.text),float(self.wz.text)))
            self.minimap.pending_marker.enabled=False
            self.blur()
            self.report('AUTO: straight to waypoint; avoid only when front sensor detects an obstacle.')
        except ValueError as e: self.report(str(e))
    def add(self,kind):
        try:
            self.c.obstacles.add(kind,*[float(f.text) for f in (self.ox,self.oz,self.sx,self.sy,self.sz)],self.c.drone)
            self.blur()
            self.report(f'{kind} added. Click it in the world or use NEXT to select.')
        except ValueError as e: self.report(str(e))
    def next_obstacle(self):
        m=self.c.obstacles
        ids=[o.id for o in m.items]
        if ids: m.selected=ids[(ids.index(m.selected)+1)%len(ids)] if m.selected in ids else ids[0]
    def delete(self): self.c.obstacles.delete()
    def clear(self):
        self.c.obstacles.clear()
        self.report('All obstacles cleared.')
    def layout(self):
        edge=u.window.aspect_ratio/2
        self.left.x,self.right.x=-edge+.187,edge-.187
        self.title.x=-edge+.024
        self.hint.x=-.40
        self.message.x=-.40

    def sync(self):
        self.layout()
        d,c,r=self.c.drone,self.c,self.c.sensor.reading
        status='LANDING' if d.landing else 'AIRBORNE' if d.airborne else 'LANDED'
        self.flight.text=(f'{status}  /  {self.rig.names[self.rig.index]}\n\n'
                          f'Speed       {d.v.length():4.2f} m/s\nSet speed   {d.speed:4.1f} m/s\n'
                          f'Altitude    {d.p.y:4.2f} m\nTarget alt  {d.target_altitude:4.1f} m\n'
                          f'Yaw         {d.yaw%360:5.1f} deg\nX {d.p.x:5.1f}   Z {d.p.z:5.1f}')
        distance='--' if r.distance==inf else f'{r.distance:.2f} m'
        sectors='   '.join(f'{name[0]} {"--" if value==inf else f"{value:.1f}"}' for name,value in r.sectors.items())
        self.sensor.text=f'{c.state}  /  {r.sector}\nNearest: {distance}  ID: {r.obstacle_id or "--"}\n{sectors}\nBrake 2.0 m  /  Stop 0.5 m'
        self.sensor.color=col('#ffbd7c' if c.state in ('BRAKING','STOPPED') else '#cfdee7')
        self.avoid.text=f'AVOIDANCE {"ON" if c.avoidance else "OFF"}'
        self.avoid.color=col('#256658' if c.avoidance else '#77473b')
        self.mode.text=f'{c.mode}  /  {c.mission}'
        self.selection.text=f'Selected: {c.obstacles.selected or "--"}   Objects: {len(c.obstacles.items)}/80'
        if c.collision: self.report('COLLISION HARD STOP - move away or remove obstacle.')
        self.source_button.text=f'P: {c.sensor.source}'
        self.depth_button.text='V: DEPTH VIEW' if self.preview.show_depth else 'V: RGB VIEW'
        self.depth_status.text=f'{c.sensor.note}\nModel: {c.sensor.worker.status}'
        self.preview.update_depth(c.sensor.worker.snapshot())
        self.minimap.sync()


def self_test():
    import unittest
    class PhysicsTests(unittest.TestCase):
        def setup_world(self):
            d,m,s=Drone(),ObstacleManager(),FrontSensor()
            c=FlightController(d,m,s)
            d.p.y=2
            d.takeoff()
            return d,m,s,c
        def test_braking(self):
            for speed in (.5,2,5):
                d,m,s,c=self.setup_world()
                m.add('Wall',0,7,8,5,1,d)
                d.speed=speed
                for _ in range(1800): c.tick(STEP,1)
                self.assertGreaterEqual(6.5-d.p.z-RADIUS,STOP-1e-5)
                self.assertLess(6.5-d.p.z-RADIUS,STOP+.04)
        def test_avoidance_off_collision(self):
            d,m,s,c=self.setup_world()
            m.add('Wall',0,4,8,5,1,d)
            c.avoidance=False
            d.speed=5
            for _ in range(500): c.tick(STEP,1)
            gap=3.5-d.p.z-RADIUS
            self.assertGreaterEqual(gap,-1e-5)
            self.assertLess(gap,.02)
        def test_fov(self):
            d,m,s,c=self.setup_world()
            m.add('Box',0,-3,1,4,1,d)
            self.assertEqual(s.scan(d,m).distance,inf)
            m.clear(); m.add('Box',5,1,1,4,1,d)
            self.assertEqual(s.scan(d,m).distance,inf)
            m.clear(); m.add('Box',0,4,1,4,1,d)
            self.assertEqual(s.scan(d,m).sector,'CENTER')
        def test_yaw_and_strafe(self):
            d,m,s,c=self.setup_world()
            d.yaw=90
            for _ in range(120): c.tick(STEP,1)
            self.assertGreater(d.p.x,1)
            self.assertAlmostEqual(d.p.z,0,places=5)
        def test_sweep_all_axes(self):
            d,m,s,c=self.setup_world()
            m.add('Box',3,0,1,5,1,d)
            self.assertLess(m.sweep(d.p,V(20,0,0)),1)
            m.clear(); m.add('Box',0,3,2,1,2,d)
            self.assertLess(m.sweep(V(0,4,3),V(0,-8,0)),1)
            self.assertLess(m.sweep(V(0,.5,8),V(0,0,-20)),1)
        def test_waypoint_and_land(self):
            d,m,s,c=self.setup_world()
            c.go(V(3,3,4))
            for _ in range(1600): c.tick(STEP)
            self.assertLess((d.p-V(3,3,4)).length(),.2)
            self.assertEqual(c.mission,'ARRIVED')
            c.land()
            for _ in range(1000): c.tick(STEP)
            self.assertFalse(d.airborne)
            self.assertAlmostEqual(d.p.y,REST)
        def test_validation(self):
            d,m,s,c=self.setup_world()
            with self.assertRaises(ValueError): m.add('Box',0,0,2,4,2,d)
            with self.assertRaises(ValueError): c.go(V(float('nan'),2,4))
        def test_grounded_land(self):
            d,m,s=Drone(),ObstacleManager(),FrontSensor()
            c=FlightController(d,m,s)
            c.land()
            c.tick(STEP)
            self.assertFalse(d.landing)
            self.assertFalse(d.airborne)
        def test_sector_and_occlusion(self):
            d,m,s,c=self.setup_world()
            near=m.add('Box',-1.8,4,1,4,1,d)
            m.add('Box',-3,7,1,4,1,d)
            reading=s.scan(d,m)
            self.assertEqual(reading.sector,'LEFT')
            self.assertEqual(reading.obstacle_id,near.id)
            m.clear(); m.add('Box',1.8,4,1,4,1,d)
            self.assertEqual(s.scan(d,m).sector,'RIGHT')
        def test_shapes_and_occlusion(self):
            d,m,s,c=self.setup_world()
            for kind in ('Sphere','Cylinder'):
                m.clear(); m.add(kind,0,4,2,4,2,d)
                self.assertAlmostEqual(s.scan(d,m).distance,3-RADIUS,places=5)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PhysicsTests))
    return result.wasSuccessful()


def main():
    global u
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--no-preview',action='store_true')
    parser.add_argument('--smoke-test',action='store_true',help='Render 90 frames, capture screenshot, then exit.')
    parser.add_argument('--offscreen',action='store_true',help='Render tests without opening a desktop window.')
    parser.add_argument('--no-depth',action='store_true',help='Do not start the optional model worker.')
    parser.add_argument('--sensor',choices=PerceptionSensor.sources,default='HYBRID')
    parser.add_argument('--depth-config',default=str(Path(__file__).with_name('depth_config.json')))
    parser.add_argument('--depth-test-image',help='Run the REAL metric model on an image and exit.')
    args=parser.parse_args()
    if args.self_test: raise SystemExit(0 if self_test() else 1)
    from depth_pipeline import DepthWorker,check_model
    if args.depth_test_image:
        check_model(args.depth_config,args.depth_test_image,Path(__file__).parent/'depth_check')
        return
    import ursina as u
    app=u.Ursina(title='DRONE LAB | 3D Simulator v2.2',borderless=False,fullscreen=False,size=(1440,900),development_mode=False,window_type='offscreen' if args.offscreen else 'onscreen')
    if args.offscreen: u.mouse.enabled=False
    u.window.color=col('#101d29')
    u.window.exit_button.visible=False
    u.window.fps_counter.enabled=True
    worker=DepthWorker(args.depth_config,enabled=not args.no_depth and not args.no_preview)
    d,m=Drone(),ObstacleManager()
    s=PerceptionSensor(worker,args.sensor)
    c=FlightController(d,m,s)
    for spec in [('Box',0,8,2,3,2),('Cylinder',4,11,2,4,2),('Wall',-6,12,6,4,.6),('Sphere',7,5,2,2,2),('Box',-7,-5,3,2,3)]:
        m.add(*spec,d)
    m.selected=None
    world=WorldView(m)
    drone_view=DroneView(d,world.shader)
    drone_view.sync(0)
    rig=CameraRig(drone_view)
    ui=FlightUI(c,rig,drone_view,args.no_preview)

    class Simulation(u.Entity):
        def __init__(self):
            super().__init__()
            self.accumulator,self.hud_timer,self.frames=0.,0.,0
            self.last_capture=0.
        def update(self):
            now=monotonic()
            if worker.ready and ui.preview.buffer and now-self.last_capture>=.25:
                # Capture corresponds to the previous rendered pose, before this tick moves it.
                frame=ui.preview.capture_rgb()
                if frame: worker.submit(frame,d.p.tup(),d.yaw,now)
                self.last_capture=now
            dt=min(u.time.dt,.1)
            focused=ui.focused()
            if not focused and c.mode=='MANUAL' and not d.landing:
                d.target_altitude=clamp(d.target_altitude+(u.held_keys['r']-u.held_keys['f'])*1.5*dt,.8,20)
            if not focused:
                d.speed=clamp(d.speed+(u.held_keys['=']+u.held_keys['+']-u.held_keys['-'])*dt,.5,5)
            self.accumulator+=dt
            while self.accumulator >= STEP:
                c.tick(STEP,0 if focused else u.held_keys['w']-u.held_keys['s'],0 if focused else u.held_keys['d']-u.held_keys['a'],0 if focused else u.held_keys['e']-u.held_keys['q'])
                self.accumulator-=STEP
            drone_view.sync(dt)
            rig.sync(dt)
            self.hud_timer+=dt
            if self.hud_timer >= .05:
                world.sync(d,c); ui.sync(); self.hud_timer=0
            self.frames+=1
            if args.smoke_test and self.frames in (10,20,30,40,50):
                rig.set((self.frames//10)%5)
            if args.smoke_test and self.frames==60:
                d.p.y=2
                d.takeoff()
                rig.set(0)
                rig.sync(1,True)
                center=ui.minimap.root.get_position(relative_to=u.camera.ui)
                ui.wx.active=True
                self.input('left mouse down',(center.x+ui.minimap.size*5/60,center.y+ui.minimap.size*10/60))
                assert ui.wx.text=='5.0' and ui.wz.text=='10.0'
                assert not ui.focused() and ui.minimap.pending_marker.enabled
                before=len(m.items)
                ui.ox.text='12'; ui.oz.text='-12'
                ui.add('Cylinder')
                assert len(m.items)==before+1
                ui.delete()
                assert len(m.items)==before
            if args.smoke_test and self.frames==90:
                from pathlib import Path
                from panda3d.core import Filename
                destination=Path(__file__).resolve().parent/('preview_no_camera.png' if args.no_preview else 'preview.png')
                assert u.application.base.screenshot(namePrefix=Filename.fromOsSpecific(str(destination)).getFullpath(),defaultFilename=False)
                if not args.no_preview:
                    frame=ui.preview.capture_rgb()
                    assert frame and len(frame[2])==384*216*3, ui.preview.error
                    assert len(set(frame[2]))>16, 'RGB camera returned a blank image'
                print('SMOKE PASS: five views, editor, screen-coordinate mouse dispatch (field focused), waypoint marker, RGB buffer.' if not args.no_preview else 'SMOKE PASS: views, editor, minimap mouse dispatch, no-preview mode.')
                ui.preview.close()
                u.application.quit()
        def input(self,key,pointer=None):
            if key=='escape': ui.blur(); return
            if key=='left mouse down' and ui.minimap.click(*(pointer or (u.mouse.x,u.mouse.y))): return
            if ui.focused(): return
            if key=='t': d.takeoff()
            elif key=='g': c.land()
            elif key=='c': rig.set(rig.index+1)
            elif key in ('1','2','3','4','5'): rig.set(int(key)-1)
            elif key=='o': ui.toggle_avoid()
            elif key=='p': s.cycle()
            elif key=='v': ui.toggle_depth()
            elif key=='m': c.manual() if c.mode=='AUTO' else ui.auto()
            elif key=='space': c.manual(); d.v=V(); d.target_altitude=d.p.y; ui.report('HOLD / MANUAL')
            elif key in ('scroll up','scroll down') and rig.index==4:
                rig.radius=clamp(rig.radius+(-1 if key=='scroll up' else 1),3,25)
            elif key=='left mouse down' and hasattr(u.mouse.hovered_entity,'obstacle_id'):
                m.selected=u.mouse.hovered_entity.obstacle_id
    Simulation()
    world.sync(d,c); ui.sync()
    try: app.run()
    finally:
        ui.preview.close()
        worker.close()


if __name__=='__main__': main()
