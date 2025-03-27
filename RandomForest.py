import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Bước 1: Đọc dữ liệu từ file CSV
data = pd.read_csv('data.csv')

# Kiểm tra cấu trúc dữ liệu
print("Dữ liệu ban đầu:")
print(data.head())

# Bước 2: Tự chọn các trường để huấn luyện và dự đoán
selected_features = ['I', 'Ton', 'Toff', 'Wire Feed', 'MRR', 'SR']  # Các cột để huấn luyện
target_column = 'Overcut'  # Cột để dự đoán

# Tách biến đầu vào (X) và biến đầu ra (y)
X = data[selected_features].values  # Lấy các cột được chọn làm đầu vào
y = data[target_column].values      # Lấy cột Overcut làm đầu ra

# Chia dữ liệu thành tập huấn luyện và tập kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bước 3: Xây dựng mô hình Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)  # n_estimators: Số lượng cây

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Bước 4: Đánh giá mô hình
# Dự đoán trên tập kiểm thử
y_pred = model.predict(X_test)

# Tính các chỉ số đánh giá
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")

# Hiển thị kết quả dự đoán
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("Kết quả dự đoán trên tập kiểm thử:")
print(results)

# ================== PHẦN BỔ SUNG: DỰ ĐOÁN TỪ FILE TEST.CSV ==================
# Bước 5: Đọc dữ liệu kiểm thử từ file test.csv
test_data = pd.read_csv('test.csv')

# Kiểm tra cấu trúc dữ liệu kiểm thử
print("\nDữ liệu kiểm thử:")
print(test_data.head())

# Lấy các cột tương ứng với selected_features từ file test.csv
X_test_input = test_data[selected_features].values

# Dự đoán kết quả
y_test_pred = model.predict(X_test_input)

# Thêm kết quả dự đoán vào DataFrame kiểm thử
test_data['Predicted_Overcut'] = y_test_pred

# Hiển thị kết quả dự đoán
print("\nKết quả dự đoán từ file test.csv:")
print(test_data[['No'] + selected_features + ['Predicted_Overcut']])

# Lưu kết quả vào file CSV
test_data.to_csv('predicted_results_random_forest.csv', index=False)
print("\nKết quả đã được lưu vào file 'predicted_results_random_forest.csv'.")