# Tối ưu hóa Thông số Gia công bằng Mạng Nơ-ron Nhân tạo (ANN) và Thuật toán Di truyền (GA)

## Mô tả dự án

Dự án này sử dụng Mạng Nơ-ron Nhân tạo (ANN) kết hợp với Thuật toán Di truyền (GA) để tìm ra bộ thông số vận hành tối ưu (Tốc độ cắt - Speed, Tốc độ tiến dao - Feed, Chiều sâu cắt - DOC) cho một quá trình gia công cơ khí. Mục tiêu là tìm điểm cân bằng hiệu quả giữa hai yếu tố quan trọng nhưng thường mâu thuẫn nhau:

1.  **Độ nhám bề mặt (Surface Roughness - SR):** Cần được **giảm thiểu** để đảm bảo chất lượng bề mặt sản phẩm.
2.  **Tốc độ loại bỏ vật liệu (Material Removal Rate - MRR):** Cần được **tối đa hóa** để tăng năng suất và hiệu quả gia công.


### 1. Mạng Nơ-ron Nhân tạo (Artificial Neural Network - ANN)

* **Vai trò:** ANN được sử dụng như một **mô hình dự đoán** hoặc **mô hình thay thế (surrogate model)**. Nó học mối quan hệ phức tạp, phi tuyến giữa các thông số đầu vào (Speed, Feed, DOC) và các kết quả đầu ra (SR, MRR) từ dữ liệu thực nghiệm có sẵn.
* **Kiến trúc:** Mô hình ANN trong code này là một mạng truyền thẳng (feedforward) tuần tự (Sequential):
    * **Lớp đầu vào (Input Layer):** Ngầm định bởi `input_dim=3` ở lớp ẩn đầu tiên, nhận 3 giá trị: Speed, Feed, DOC.
    * **Lớp ẩn (Hidden Layer):** Một lớp ẩn với 6 nơ-ron và hàm kích hoạt `tanh`. Hàm `tanh` giúp mô hình học các mối quan hệ phi tuyến.
    * **Lớp đầu ra (Output Layer):** Có 2 nơ-ron (tương ứng SR và MRR) với hàm kích hoạt `linear`. Hàm `linear` phù hợp cho bài toán hồi quy vì đầu ra là các giá trị số thực liên tục.
* **Huấn luyện:** Mô hình được huấn luyện bằng thuật toán `adam` để tối thiểu hóa hàm mất mát `mean_squared_error` (MSE) trên dữ liệu huấn luyện (đã được chuẩn hóa). Dữ liệu được chia thành tập huấn luyện và tập kiểm tra để đánh giá hiệu năng của mô hình trên dữ liệu mới.
* **Chuẩn hóa dữ liệu:** Dữ liệu đầu vào và đầu ra được chuẩn hóa về khoảng [0, 1] bằng `MinMaxScaler`. Điều này rất quan trọng để ANN hoạt động hiệu quả.

### 2. Thuật toán Di truyền (Genetic Algorithm - GA)

* **Vai trò:** GA đóng vai trò là **cơ chế tìm kiếm và tối ưu hóa**. Nó khám phá không gian các bộ thông số (Speed, Feed, DOC) khả thi để tìm ra giải pháp (cá thể) tốt nhất dựa trên đánh giá từ mô hình ANN.
* **Nguyên tắc hoạt động:** GA mô phỏng quá trình tiến hóa tự nhiên:
    * **Quần thể (Population):** Một tập hợp các giải pháp tiềm năng (các bộ thông số [Speed, Feed, DOC]), gọi là các **cá thể (Individuals)**.
    * **Hàm thích nghi (Fitness Function):** Đánh giá "chất lượng" của mỗi cá thể. Trong code này, hàm fitness sử dụng dự đoán SR và MRR từ ANN (`predict_outputs` function). Nó kết hợp hai mục tiêu bằng cách sử dụng tham số `bias`:
        `fitness = bias * normalized_SR_score + (1 - bias) * normalized_MRR_score`
        Giá trị SR được chuẩn hóa ngược (giá trị SR thấp -> điểm cao), MRR được chuẩn hóa xuôi (giá trị MRR cao -> điểm cao). `bias` (từ 0 đến 1) cho phép điều chỉnh mức độ ưu tiên giữa việc giảm SR và tăng MRR.
    * **Chọn lọc (Selection):** Các cá thể có độ thích nghi cao hơn có nhiều khả năng được chọn để tạo ra thế hệ tiếp theo (sử dụng phương pháp Bánh xe Roulette - Roulette Wheel Selection).
    * **Lai ghép (Crossover):** Kết hợp "vật liệu di truyền" (giá trị thông số) của hai cá thể cha mẹ để tạo ra cá thể con (sử dụng Lai ghép một điểm cắt - Single-point Crossover).
    * **Đột biến (Mutation):** Thay đổi ngẫu nhiên một phần nhỏ trong thông tin di truyền của cá thể với một xác suất thấp, giúp duy trì sự đa dạng và tránh bị kẹt ở tối ưu cục bộ.
    * **Tiến hóa:** Quá trình chọn lọc, lai ghép, đột biến được lặp lại qua nhiều thế hệ (`max_gen`), giúp quần thể dần hội tụ về các giải pháp ngày càng tốt hơn. Kỹ thuật **Elitism** (giữ lại cá thể tốt nhất của mỗi thế hệ) cũng được sử dụng.

## Cách hoạt động của Code

1.  **Tải và Chuẩn bị dữ liệu:** Đọc dữ liệu từ `data.csv`, tách thành đầu vào (X) và đầu ra (y), chuẩn hóa cả X và y bằng `MinMaxScaler`, sau đó chia thành tập huấn luyện và kiểm tra.
2.  **Huấn luyện ANN:** Xây dựng và huấn luyện mô hình ANN (`create_ann_model`, `ann_model.fit`) trên dữ liệu huấn luyện đã chuẩn hóa. Đánh giá sơ bộ mô hình trên tập kiểm tra.
3.  **Định nghĩa GA:** Xây dựng các hàm cần thiết cho GA:
    * `create_individual`: Tạo cá thể ngẫu nhiên trong phạm vi cho phép.
    * `create_population`: Tạo quần thể ban đầu.
    * `predict_outputs`: **Kết nối GA và ANN**. Lấy cá thể GA -> chuẩn hóa -> dự đoán bằng ANN -> giải chuẩn hóa -> trả về SR, MRR thực tế dự đoán.
    * `calculate_fitness`: Tính độ thích nghi dựa trên SR, MRR dự đoán và giá trị `bias`.
    * `selection`, `crossover`, `mutate`: Thực hiện các toán tử di truyền cơ bản.
    * `optimize`: Hàm điều khiển chính của GA, chạy qua các thế hệ, thực hiện các bước đánh giá, chọn lọc, lai ghép, đột biến.
4.  **Chạy Tối ưu hóa:** Gọi hàm `optimize` nhiều lần với các giá trị `bias` khác nhau (0.0, 0.25, 0.5, 0.75, 1.0) để khám phá các giải pháp tối ưu tương ứng với các mức độ ưu tiên khác nhau giữa SR và MRR.
5.  **Hiển thị Kết quả:** In ra các bộ thông số tối ưu tìm được cho từng giá trị `bias` dưới dạng bảng và văn bản mô tả. (Code cũng bao gồm phần vẽ biểu đồ bằng `matplotlib`, hiện đang được comment).


## Diễn giải Kết quả

Script sẽ in ra kết quả tối ưu hóa cho từng giá trị `bias` đã được thử nghiệm.

* **`bias = 1.0` (100% bias on surface finish):** Kết quả sẽ ưu tiên tìm bộ thông số cho **SR thấp nhất** có thể, bất kể MRR.
* **`bias = 0.0` (0% bias on surface finish):** Kết quả sẽ ưu tiên tìm bộ thông số cho **MRR cao nhất** có thể, bất kể SR.
* **`bias = 0.5` (50% bias on surface finish):** Kết quả tìm kiếm sự **cân bằng** giữa việc giảm SR và tăng MRR.
* Các giá trị `bias` khác (0.25, 0.75) thể hiện các mức độ ưu tiên trung gian khác nhau.

Bảng kết quả và các thông số cuối cùng cho thấy các bộ thông số [Speed, Feed, DOC] khác nhau được đề xuất tùy thuộc vào mục tiêu ưu tiên. Điều này minh họa sự đánh đổi (trade-off) giữa chất lượng bề mặt và năng suất gia công, cung cấp cho người dùng các lựa chọn vận hành dựa trên yêu cầu cụ thể của họ.