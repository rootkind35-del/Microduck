# THUYẾT MINH ĐỀ TÀI NGHIÊN CỨU KHOA HỌC SINH VIÊN

**TÊN ĐỀ TÀI:**  
NGHIÊN CỨU ỨNG DỤNG HỌC TĂNG CƯỜNG ĐỂ HUẤN LUYỆN HỆ THỐNG KỸ NĂNG VẬN ĐỘNG VÀ TƯƠNG TÁC CHO ROBOT HAI CHÂN MICRODUCK TRONG MÔI TRƯỜNG MÔ PHỎNG  

**TRỌNG TÂM CÔNG NGHỆ:**  
XÂY DỰNG HỆ THỐNG ĐIỀU PHỐI CÁC MULTI-MODEL KỸ NĂNG ỨNG DỤNG CHO ROBOTICS (KIẾN TRÚC EMBODIED MIXTURE-OF-EXPERTS — NEKO CORE)

---

## 1. Tính cấp thiết của đề tài

Nghiên cứu chế tạo và điều khiển robot hai chân (bipedal robot) luôn là bài toán phức tạp và giàu thách thức nhất trong lĩnh vực kỹ thuật robot (robotics). Khác với robot bánh xe (wheeled robot) hay robot bốn chân (quadruped robot), robot hai chân sở hữu cấu trúc đa giác nâng (support polygon) cực kỳ nhỏ hẹp và trọng tâm liên tục biến thiên, khiến trạng thái cân bằng luôn là cân bằng động bất ổn định (dynamic instability) [1].

Theo trường phái điều khiển kinh điển, bài toán bước đi thường được giải quyết thông qua mô hình con lắc ngược tuyến tính (Linear Inverted Pendulum Model - LIPM), quy hoạch điểm mô-men bằng không (Zero Moment Point - ZMP) hoặc điều khiển dự báo mô hình (Model Predictive Control - MPC) [2]. Tuy nhiên, các phương pháp này đòi hỏi phải giải các bài toán động lực học nghịch đảo (Inverse Dynamics) phi tuyến vô cùng nặng nề theo thời gian thực, đồng thời phụ thuộc chặt chẽ vào các giả thiết đơn giản hóa về mặt sàn và độ cứng khớp [1], [2]. Khi robot di chuyển vào các địa hình phi cấu trúc như sỏi đá nhấp nhô hay cầu thang dốc, sự sai lệch về tiếp xúc khiến các bộ điều khiển kinh điển dễ rơi vào trạng thái mất ổn định hoặc sụp đổ hoàn toàn.

Sự phát triển của Học tăng cường sâu (Deep Reinforcement Learning - DRL), tiêu biểu là thuật toán Tối ưu hóa chính sách cận kề (Proximal Policy Optimization - PPO) [3], đã tạo ra bước chuyển dịch phương pháp luận: chuyển từ việc lập trình phương trình toán học cứng nhắc sang cơ chế để mạng nơ-ron tự tối ưu hóa hành vi thông qua tương tác thử-và-sai (trial-and-error) trong môi trường giả lập vật lý [4]. Tuy nhiên, khi áp dụng DRL cho robot hai chân trong thực tế, các nhà nghiên cứu đang đối mặt với hai rào cản mang tính cốt tử:

1. **Sự bế tắc của mô hình nguyên khối (Monolithic Policy):** Các kỹ năng vận động thực tế có yêu cầu động học hoàn toàn xung đột. Cụ thể, kỹ năng đi bộ trên đường phẳng đòi hỏi sải chân thấp (1.2 − 1.5 cm) với tần số bước cao để tiết kiệm năng lượng; trong khi kỹ năng vượt địa hình gồ ghề (rough terrain) và leo bậc thang cao (high stairs climbing) bắt buộc robot phải nâng cao đùi, nhấc chân chủ động lên tới 5.5 cm (vượt quá 15% chiều cao cơ thể). Việc ép một mạng nơ-ron đơn lẻ học chung tất cả các tác vụ dẫn đến hiện tượng Xung đột đạo hàm (Gradient Conflict) và Quên tai hại (Catastrophic Forgetting), khiến hiệu năng của các kỹ năng đều bị suy thoái.
2. **Cú sốc động học khi chuyển giao chính sách (Kinematic & Actuator Shock):** Trong hệ sinh thái mã nguồn mở hiện hành của robot hai chân Microduck (awesome-microduck / pollen-robotics/microduck_rl) [5], [6], các chính sách được đóng gói độc lập trong các khe nhớ (policy slots). Khi chuyển đổi giữa các kỹ năng, hệ thống thực hiện thao tác tráo đổi cứng (hard-cut switching). Bước nhảy hành động tức thời ($\Delta a = \|a_{t}^+ - a_{t}^-\|$) làm gián đoạn đạo hàm gia tốc góc khớp ($\ddot{q} \to \infty$), gây ra xung dòng điện đột biến $I_{transient} \gg I_{max}$ trong mô hình cơ cấu chấp hành BAM (Better Actuator Models) [7]. Xung lực này dẫn đến sụt áp pin ($V_{in} < 6.0\text{V}$), kích hoạt rơ-le ngắt bảo vệ của servo Dynamixel XL330 hoặc phá hủy hộp số nhựa. Hơn nữa, việc chuyển đổi vô điều kiện giữa lúc chân đang vung lơ lửng (mid-swing phase) làm robot mất điểm tựa động học và ngã quỵ.

Xuất phát từ các vấn đề khoa học nêu trên, đề tài "Nghiên cứu ứng dụng Học tăng cường để huấn luyện hệ thống kỹ năng vận động và tương tác cho robot hai chân Microduck trong môi trường mô phỏng" được triển khai với trọng tâm: Xây dựng hệ thống điều phối đa mô hình kỹ năng (Embodied Mixture-of-Experts - Embodied MoE) mang tên Neko Core. Đề tài lấy robot hai chân Microduck làm nền tảng thực chứng với hai bài toán thách thức bậc nhất: Vượt địa hình gồ ghề ngẫu nhiên và Chinh phục bậc thang cao, nhằm thiết lập một khung điều khiển cắm-và-chạy (plug-and-play), sẵn sàng mở rộng cho các hệ thống robot đa mô hình thế hệ mới.

---

## 2. Mục tiêu của đề tài

### 2.1. Mục tiêu tổng quát
Nghiên cứu, thiết kế và thực chứng thành công khung kiến trúc điều phối đa mô hình kỹ năng (Embodied Mixture-of-Experts Architecture) dựa trên thuật toán Học tăng cường PPO cho robot hai chân Microduck trong môi trường mô phỏng vật lý MuJoCo Warp. Hệ thống đảm bảo khả năng chuyển đổi nóng (hot-swaps) mượt mà, không độ trễ giữa các chính sách chuyên gia (expert policies), triệt tiêu hoàn toàn lực giật gia tốc (zero jerk shock), vượt qua các bài toán địa hình phức tạp và xuất khẩu các mô hình dưới định dạng ONNX tích hợp chuẩn hóa phục vụ chuyển giao thực tế (Sim-to-Real).

### 2.2. Mục tiêu cụ thể
1. **Chuẩn hóa Hợp đồng quan sát nhất thể 61 chiều (Unified 61D Observation Contract):** Thiết lập không gian trạng thái bất biến gồm 48 chiều tự cảm bản thể (proprioception) và 13 chiều thanh ghi lệnh điều khiển (zero-padded command bus). Cấu trúc này cho phép hoán đổi đồ thị mạng nơ-ron trong đúng một chu kỳ điều khiển 20 ms (50 Hz) mà không cần tái cấp phát bộ nhớ RAM/VRAM.
2. **Xây dựng Động cơ hòa trộn động học Quintic S-Curve và Chốt an toàn pha bước (Quintic S-Curve Engine & Gait Phase Latch):** Hiện thực hóa giải thuật hòa trộn đạt độ trơn cấp hai ($C^2$-continuity) với điều kiện biên triệt tiêu vận tốc và gia tốc ở hai đầu mút ($\dot{w} = \ddot{w} = 0$), triệt tiêu lực giật cơ học Jerk và chống sụt áp nguồn pin ($V_{in} \ge 6.85\text{V}$). Tích hợp cơ chế khóa pha tiếp đất, chỉ cho phép hòa trộn khi cả hai bàn chân ở pha tiếp đất kép (double support phase).
3. **Huấn luyện Mô hình Chuyên gia Vượt địa hình gồ ghề (Rough Terrain Locomotion Expert - rough_latest.onnx):** Ứng dụng hàm thưởng tiềm năng bám hướng nội sinh POMDP (Heading-lock POMDP reward) và làm mềm tiếp xúc sàn (solref = [0.04, 1.0]) trên địa hình sỏi đá nhấp nhô BoxRandomGrid, triệt tiêu độ trôi hướng Yaw từ $\pm 35.2^\circ$ xuống $< 1.8^\circ$.
4. **Huấn luyện Mô hình Chuyên gia Leo bậc thang cao (High Stairs Climbing Expert - stairs_latest.onnx):** Xây dựng giáo trình huấn luyện tăng dần (Curriculum Learning) nâng độ nhấc chân chủ động từ 3.5 cm lên 5.5 cm, chinh phục bậc thang chuẩn hình học cao 1.0 cm → 3.5 cm (chiếm 14% chiều cao cơ thể robot) trên mô hình va chạm toàn thân 70 hình học (70-geom full-collision model).
5. **Huấn luyện các mô hình sinh tồn, tư thế và phục hồi động lực học:** Đảm bảo hệ sinh thái kỹ năng hoàn chỉnh gồm: Tự đứng dậy từ ngã úp/ngửa 100% (StandUp), chuyển đổi ngồi - đứng (SitStand), và kỹ năng nhào lộn cuộn mình thoát hiểm khi va chạm mạnh (Roulade Somersault 360°).
6. **Đánh giá thực nghiệm đối chứng (Ablation Study) và Xuất mô hình Sim2Real:** Kiểm thử ma trận 100 lần chạy ngẫu nhiên trên mô phỏng; tích hợp sẵn bộ chuẩn hóa dữ liệu quan sát (Baked Observation Normalizer) vào tệp ONNX để sẵn sàng triển khai trên phần cứng nhúng thời gian thực.

---

## 3. Tổng quan tình hình nghiên cứu liên quan đến đề tài
- **3.1. Học tăng cường sâu trong điều khiển chuyển động robot có chân (Legged Locomotion DRL)**: Schulman et al. (2017) [PPO], Rudin et al. (2022) [Isaac Gym, POMDP blind locomotion].
- **3.2. Rào cản chuyển giao mô phỏng vào thực tế (Sim-to-Real Gap) và Mô hình cơ cấu chấp hành**: Hwangbo et al. (2019) [ANYmal, actuator net], Rhoban (BAM M6 Dynamixel XL330).
- **3.3. Điều phối đa kỹ năng và Sự gián đoạn hành động cơ học**: Peng et al. (2022) [ASE], Merel et al. (2019).
- **3.4. Khoảng trống nghiên cứu (Research Gap)**: Thiếu kiến trúc điều phối đa mô hình thống nhất không sụt áp/giật gián đoạn trên micro-biped; thiếu kỹ năng leo bậc thang hình học cao 14% cơ thể và vượt địa hình sỏi đá có khóa hướng nội sinh.

---

## 4. Đối tượng, phạm vi nghiên cứu, cách tiếp cận và phương pháp nghiên cứu
- **Đối tượng**: PPO Asymmetric Actor-Critic, động lực học bipedal tiếp xúc ma sát, BAM M6 XL330, Quintic Spline $C^2$ và Gait Phase Latch FSM.
- **Phạm vi**: Robot Microduck (~0.74 kg theo mô hình URDF kiểm định, ~0.8 kg vật lý; 25cm, 14 DOF), mô phỏng MuJoCo Warp GPU (mjlab), xuất ONNX chuẩn hóa, không chế tạo cơ khí lại khung robot. Thời gian 10/2026 - 04/2027.
- **Cách tiếp cận**: Physics-Informed & Data-Driven Sim-to-Real Approach kết hợp Task Decomposition (Embodied MoE).
- **Phương pháp**: Nghiên cứu lý thuyết; mô hình hóa toán học & mô phỏng số (MJCF 70 geoms, BAM M6); thực nghiệm song song GPU 4096 envs (DR ma sát, khối lượng, backlash); đo kiểm đối chứng (Ablation Study 100 seeds).

---

## 5. Nội dung nghiên cứu và kế hoạch thực hiện
- Chương 1: Giới thiệu tổng quan & Tính cấp thiết.
- Chương 2: Cơ sở lý thuyết & Nền tảng công nghệ (PPO, MuJoCo Warp, BAM M6, Quintic Spline).
- Chương 3: Thiết kế hệ thống điều phối Multi-Model (Neko Core: 61D contract, Quintic S-Curve, Gait Phase Latch, Reward engineering).
- Chương 4: Huấn luyện 2 mô hình chuyên gia thực chứng (Rough terrain, Stairs) & hệ sinh thái kỹ năng (Standup, Sitstand, Roulade), Ablation study, Baked ONNX.
- Chương 5: Kết luận & Hướng phát triển (mở rộng Quadruped, Humanoid).

---

## 6. Sản phẩm của đề tài
1. Quyển Báo cáo Tổng kết Đề tài NCKH Sinh viên.
2. Bộ Mô hình Điều khiển AI (ONNX Policy Bank): rough_latest.onnx (hoàn thành 100% 2 kịch bản vòng kín: bãi đá cuội 50 khối X→1.614 m và cầu thang 9 bậc X→2.279 m, +119.6 mm, V_min ≥ 6.92 V), stairs expert (curriculum bậc tới 3.5 cm), standup.onnx, sitstand.onnx, roulade.onnx; bộ điều phối Neko Core đạt 96% (24/25) sống sót chuyển tiếp, giảm 99.6% bước nhảy lệnh biên (2.720 → 0.010 rad) — số liệu đối chứng 100 lượt chạy trong `results/neko_core_ablation_summary.json`.
3. Bộ Mã nguồn Điều phối Phân cấp & Kịch bản Kiểm thử (Neko Core Framework).
4. 01 Bài báo khoa học (Manuscript Draft).
5. Hồ sơ kiểm chứng Sim-to-Real sẵn sàng cho daemon robotctl.

