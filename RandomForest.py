import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize

# ================== Bước 1: Đọc dữ liệu và xây dựng mô hình ==================
# Đọc dữ liệu từ file CSV
data = pd.read_csv('data.csv')

# Kiểm tra cấu trúc dữ liệu
print("Dữ liệu ban đầu:")
print(data.head())

# Chọn các cột đầu vào và đầu ra
selected_features = ['I', 'Ton', 'Toff', 'Wire Feed']  # Các cột đầu vào
targets = ['MRR', 'SR', 'Overcut']  # Các cột đầu ra

# Tách biến đầu vào (X) và biến đầu ra (y)
X = data[selected_features].values  # Lấy các cột đầu vào
y = data[targets].values            # Lấy các cột đầu ra

# Chia dữ liệu thành tập huấn luyện và tập kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình Random Forest cho từng mục tiêu
models = {}
for i, target in enumerate(targets):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train[:, i])  # Huấn luyện mô hình cho từng mục tiêu
    models[target] = model

# Đánh giá mô hình
for target in targets:
    y_pred = models[target].predict(X_test)
    mse = mean_squared_error(y_test[:, targets.index(target)], y_pred)
    r2 = r2_score(y_test[:, targets.index(target)], y_pred)
    print(f"{target} - Mean Squared Error: {mse}, R²: {r2}")

# ================== Bước 2: Định nghĩa hàm mục tiêu ==================
def objective_function(params, models):

    # Dự đoán giá trị của MRR, SR, Overcut
    params = np.array(params).reshape(1, -1)  # Reshape để phù hợp với mô hình
    mrr = models['MRR'].predict(params)[0]
    sr = models['SR'].predict(params)[0]
    overcut = models['Overcut'].predict(params)[0]
    
    # Định nghĩa hàm mục tiêu (ví dụ: MRR lớn nhất, SR và Overcut nhỏ nhất)
    # Bạn có thể điều chỉnh trọng số (weights) tùy theo mức độ ưu tiên
    weight_mrr = 1.0  # Trọng số cho MRR
    weight_sr = -1.0  # Trọng số cho SR (tối thiểu hóa)
    weight_overcut = -1.0  # Trọng số cho Overcut (tối thiểu hóa)
    
    # Giá trị hàm mục tiêu
    objective_value = (
        weight_mrr * mrr +
        weight_sr * sr +
        weight_overcut * overcut
    )
    return -objective_value  # Nhớ thêm dấu trừ vì optimizer tìm giá trị nhỏ nhất

# ================== Bước 3: Sử dụng thuật toán tối ưu hóa ==================
# Giới hạn giá trị cho các biến đầu vào (giả sử phạm vi dựa trên dữ liệu)
bounds = [
    (1, 10),    # Phạm vi cho I
    (100, 150), # Phạm vi cho Ton
    (20, 60),   # Phạm vi cho Toff
    (1, 15)     # Phạm vi cho Wire Feed
]

# Khởi tạo giá trị ban đầu ngẫu nhiên
initial_guess = [np.mean(bound) for bound in bounds]

# Tối ưu hóa
result = minimize(
    lambda params: objective_function(params, models),
    initial_guess,
    bounds=bounds,
    method='L-BFGS-B'
)

# Hiển thị kết quả tối ưu
optimal_params = result.x
print("\nGiá trị tối ưu cho các biến đầu vào:")
print(f"I: {optimal_params[0]:.2f}, Ton: {optimal_params[1]:.2f}, Toff: {optimal_params[2]:.2f}, Wire Feed: {optimal_params[3]:.2f}")

# Dự đoán giá trị của MRR, SR, Overcut với các tham số tối ưu
optimal_mrr = models['MRR'].predict([optimal_params])[0]
optimal_sr = models['SR'].predict([optimal_params])[0]
optimal_overcut = models['Overcut'].predict([optimal_params])[0]

print("\nKết quả dự đoán với các tham số tối ưu:")
print(f"MRR: {optimal_mrr:.4f}, SR: {optimal_sr:.4f}, Overcut: {optimal_overcut:.4f}")