# Drone Lab 2.2 — bay thẳng, gặp vật cản mới né, quay lại đường thẳng

Bản này thay AUTO A* của v2.1 bằng điều hướng phản ứng theo camera trước.
File nguồn vẫn tên `drone_sim_3d_2.py`. File gốc ở ổ E chưa bị ghi đè.
Giải nén toàn bộ ZIP và chạy file mới, không chỉ chép riêng file Python.

## Hành vi đúng theo yêu cầu mới

1. Chấm waypoint trên minimap: marker trắng và X/Z được cập nhật.
2. Bấm GO: drone lên độ cao mục tiêu, hướng tới waypoint và **bay thẳng**.
3. Chỉ khi cảm biến trước phát hiện obstacle trong vùng phanh 2 m,
   drone giảm tốc, chọn phía có khoảng trống và né cục bộ.
4. Trong khi né, drone tiếp tục quan sát. Nếu phía định đi chưa nằm trước camera,
   drone dừng chuyển động ngang, quay camera sang hướng đó rồi mới đi tiếp.
5. Khi hành lang quay lại đã thông, drone hướng tới một điểm trên đường thẳng
   xuất phát–waypoint. Về gần đường này rồi tiếp tục bay thẳng tới waypoint.

HUD thể hiện:

| Trạng thái AUTO | Ý nghĩa |
|---|---|
| DIRECT | Đang bay thẳng tới waypoint |
| AVOID | Đang dò hướng trống và né vật cản vừa phát hiện |
| REJOIN | Đang trở lại đường thẳng xuất phát–waypoint |
| ARRIVED | Đã tới đích |
| BLOCKED | Không tiến triển; giữ vị trí để người dùng can thiệp |
| SENSOR HOLD | DEPTH thiếu ảnh hợp lệ; dừng chuyển động |

**Đường vàng chỉ là đoạn thẳng tham chiếu ban đầu**, không phải đường vòng tính
sẵn. Đường xanh là quỹ đạo đã bay. Không vẽ trước các đoạn né tương lai.

## Điều hướng lấy thông tin ở đâu?

`ReactiveNavigator` không nhận `ObstacleManager`, không đọc danh sách obstacle,
kích thước, ID hoặc vị trí ẩn. Không còn A*, route queue hay tính lại đường từ map.
GO cũng không kiểm tra bản đồ để đoán waypoint có bị bao kín hay không.

Navigator chỉ nhận pose drone, waypoint và `SensorReading` từ camera trước:

- GEOMETRY: điểm bề mặt thực sự nhìn thấy từ các tia trong FOV 75°.
- DEPTH: các điểm 3D suy ra từ ảnh metric depth và pose lúc chụp.
- HYBRID: kết hợp hai nguồn trên; khi model không hợp lệ, báo rõ GEO FALLBACK.

Các điểm đã thấy được nhớ tối đa 8 giây và trong phạm vi 12 m để tránh vừa quay
camera là quên vật cản. Chỉ đổi hướng khi vào vùng phát hiện gần; nhớ vật ở xa
không tạo ra đường vòng trước. Chọn hướng cục bộ khoảng 10 lần/giây, giữ phía né
đã chọn cho đến khi có thể trở lại đường thẳng để hạn chế lắc trái/phải trước tường.
Điểm quan sát được nới rộng theo bề ngang drone và biên an toàn khi đánh giá khe.

Kiểm tra va chạm cuối cùng vẫn dùng hình học toàn scene như một **hard-stop độc lập**;
nó không cung cấp đường đi cho navigator. MANUAL vẫn do bạn lái, chỉ tự phanh khi
tiến. Tắt AVOIDANCE sẽ bỏ tự né/phanh, vẫn giữ hard-stop.

Đây là điều hướng cục bộ, không bảo đảm giải được mê cung, khe hẹp hoặc đích nằm
trong obstacle. Nếu không dịch chuyển đáng kể trong 25 giây khi né, drone báo
BLOCKED. Bạn có thể về MANUAL, lùi ra hoặc chọn waypoint mới rồi GO. Sensor lấy
mẫu và model depth đều có sai số; collision guard vẫn là lớp bảo vệ cuối.

## Cài và chạy

Cần Python 3.12+. Với cùng môi trường bạn đang dùng:

```powershell
python -m pip install -r requirements-depth.txt
python drone_sim_3d_2.py
```

Để thử riêng thuật toán né bằng camera hình học, không cần PyTorch:

```powershell
python -m pip install -r requirements.txt
python drone_sim_3d_2.py --no-depth --sensor GEOMETRY
```

`run_simulator_v2_2.cmd` dùng `.venv` nếu có, nếu không dùng `python` trong PATH.
Không tự tải model. Cấu hình hiện tại trong `depth_config.json` trỏ tới:

```text
E:/ne_vat_can/Depth-Anything-V2
E:/ne_vat_can/models/depth_anything_v2_metric_indoor_small.pth
```

Giữ `depth_pipeline.py`, `depth_config.json` và script chính cùng thư mục.
Nếu chuyển máy, sửa hai đường dẫn trên. ZIP không chứa checkpoint hoặc repo model.

## Depth và controls

- **P** / nút source: HYBRID → DEPTH → GEOMETRY.
- **V**: đổi RGB / heatmap depth; khi chưa có depth thì hiện RGB kèm trạng thái.
- T/G takeoff/land; WASD di chuyển; QE yaw; RF altitude; +/- tốc độ.
- C hoặc 1–5 đổi view; O avoidance; M mode; Space hủy mission; Esc bỏ focus ô nhập.

Depth Anything V2 metric Small chạy trên worker riêng, tối đa 4 ảnh/giây,
input mặc định 266, CPU 2 thread; có CUDA/MPS thì tự dùng. Ảnh quá 2 giây, dịch
pose hơn 1 m, yaw lệch hơn 12° hoặc chứa dữ liệu không hợp lệ bị loại. DEPTH sẽ
giữ vị trí, HYBRID hiện rõ fallback về geometry. Đây là khoảng cách model ước
lượng; chưa hiệu chuẩn trên world primitive của simulator.

Kiểm tra nạp checkpoint thật trên ảnh:

```powershell
python drone_sim_3d_2.py --depth-test-image "E:\ne_vat_can\test_depth2.jpg"
```

Kết quả ở `depth_check/`: `.npy` metric, heatmap PNG và báo cáo JSON.
Model thật vẫn **chưa được xác minh tại môi trường làm việc này** vì tải thư viện
bị chặn. Việc đổi hướng mới đã được kiểm thử với camera hình học và dữ liệu depth
giả lập; không gọi dữ liệu giả lập là kết quả model.

## Kiểm thử

```powershell
python drone_sim_3d_2.py --self-test
python -m unittest -v test_drone_v22
python drone_sim_3d_2.py --smoke-test --no-depth
```

Đã qua 10 kiểm thử điều khiển gốc và 14 kiểm thử mới, gồm:

- Không đọc map và không tính route khi GO; scene trống bay thẳng.
- Box / tường rộng / cylinder: phát hiện rồi né, trở lại đường thẳng trước đích,
  tới đích không va chạm trong các trường hợp kiểm thử.
- Hai obstacle liên tiếp, thêm obstacle lúc đang bay, tốc độ đặt 5 m/s.
- OFF không tự đổi hướng nhưng hard-stop vẫn hoạt động.
- Click minimap cập nhật field/marker dù field đang focus, theo nhiều tỉ lệ cửa sổ.
- Chiếu điểm metric depth, stale/invalid pose, HYBRID fallback và pixel không hợp lệ.

Trong phiên cập nhật này chưa thử trực tiếp cửa sổ native bằng chuột: môi trường
render bị lỗi truy cập asset. Kiểm thử minimap dùng phương thức UI thật với tọa độ
chuột giả lập. Có thể chạy smoke-test trên máy bạn theo lệnh trên.

Tài liệu model: [Depth Anything V2 metric](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth).
