import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scipy.optimize import minimize

# ================== Bước 1: Đọc dữ liệu và xây dựng mô hình mạng nơ-ron ==================
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

# Chuẩn hóa dữ liệu đầu vào
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Chia dữ liệu thành tập huấn luyện và tập kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Xây dựng mô hình mạng nơ-ron cho từng mục tiêu
models = {}
for i, target in enumerate(targets):
    model = Sequential()
    model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))  # Lớp ẩn với 10 nơ-ron
    model.add(Dense(1, activation='linear'))  # Lớp đầu ra với 1 nơ-ron
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Huấn luyện mô hình
    model.fit(X_train, y_train[:, i], epochs=200, batch_size=8, verbose=0)
    models[target] = model

# Đánh giá mô hình
for target in targets:
    y_pred_scaled = models[target].predict(X_test).flatten()
    y_pred = scaler_y.inverse_transform(np.column_stack([y_pred_scaled] * len(targets)))[:, targets.index(target)]
    y_true = scaler_y.inverse_transform(y_test)[:, targets.index(target)]
    
    mse = np.mean((y_true - y_pred) ** 2)
    print(f"{target} - Mean Squared Error: {mse}")

# ================== Bước 2: Định nghĩa hàm mục tiêu ==================
def objective_function(params, models, scaler_X, scaler_y):
    """
    Hàm mục tiêu để tối ưu hóa.
    params: Các giá trị của I, Ton, Toff, Wire Feed (biến đầu vào).
    models: Các mô hình đã huấn luyện cho MRR, SR, Overcut.
    scaler_X, scaler_y: Scaler để chuẩn hóa và khôi phục dữ liệu.
    """
    # Chuẩn hóa tham số đầu vào
    params_scaled = scaler_X.transform(np.array(params).reshape(1, -1))
    
    # Dự đoán giá trị của MRR, SR, Overcut
    mrr_scaled = models['MRR'].predict(params_scaled)[0]
    sr_scaled = models['SR'].predict(params_scaled)[0]
    overcut_scaled = models['Overcut'].predict(params_scaled)[0]
    
    # Khôi phục giá trị về phạm vi ban đầu
    predictions_scaled = np.column_stack([mrr_scaled, sr_scaled, overcut_scaled])
    predictions = scaler_y.inverse_transform(predictions_scaled)
    mrr, sr, overcut = predictions[0]
    
    # Định nghĩa hàm mục tiêu (ví dụ: MRR lớn nhất, SR và Overcut nhỏ nhất)
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
    lambda params: objective_function(params, models, scaler_X, scaler_y),
    initial_guess,
    bounds=bounds,
    method='L-BFGS-B'
)

# Hiển thị kết quả tối ưu
optimal_params = result.x
print("\nGiá trị tối ưu cho các biến đầu vào:")
print(f"I: {optimal_params[0]:.2f}, Ton: {optimal_params[1]:.2f}, Toff: {optimal_params[2]:.2f}, Wire Feed: {optimal_params[3]:.2f}")

# Dự đoán giá trị của MRR, SR, Overcut với các tham số tối ưu
optimal_mrr, optimal_sr, optimal_overcut = scaler_y.inverse_transform(
    np.column_stack([
        models['MRR'].predict(scaler_X.transform(np.array(optimal_params).reshape(1, -1))),
        models['SR'].predict(scaler_X.transform(np.array(optimal_params).reshape(1, -1))),
        models['Overcut'].predict(scaler_X.transform(np.array(optimal_params).reshape(1, -1)))
    ])
)[0]

print("\nKết quả dự đoán với các tham số tối ưu:")
print(f"MRR: {optimal_mrr:.4f}, SR: {optimal_sr:.4f}, Overcut: {optimal_overcut:.4f}")