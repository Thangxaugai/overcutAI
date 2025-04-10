import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import random

# 1. Dữ liệu thực nghiệm (từ bài báo)
data = {
    'Speed': [80, 80, 80, 80, 80, 80, 80, 80, 80, 180, 180, 180, 180, 180, 180, 180, 180, 180, 280, 280, 280, 280, 280, 280, 280, 280, 280],
    'Feed': [0.06, 0.06, 0.06, 0.13, 0.13, 0.13, 0.21, 0.21, 0.21, 0.06, 0.06, 0.06, 0.13, 0.13, 0.13, 0.21, 0.21, 0.21, 0.06, 0.06, 0.06, 0.13, 0.13, 0.13, 0.21, 0.21, 0.21],
    'DOC': [0.5, 0.75, 1, 0.5, 0.75, 1, 0.5, 0.75, 1, 0.5, 0.75, 1, 0.5, 0.75, 1, 0.5, 0.75, 1, 0.5, 0.75, 1, 0.5, 0.75, 1, 0.5, 0.75, 1],
    'SR': [0.34, 0.31, 0.30, 0.75, 0.73, 0.72, 1.51, 1.49, 1.48, 0.43, 0.41, 0.41, 0.76, 0.75, 0.74, 1.51, 1.50, 1.49, 0.50, 0.50, 0.49, 0.90, 0.89, 0.80, 1.68, 1.68, 1.67],
    'MRR': [1203.35, 1807.56, 2413.48, 2607.26, 3916.39, 5229.21, 4211.73, 6326.47, 8447.19, 2707.54, 4067.02, 5430.34, 5866.34, 8811.87, 11765.73, 9476.40, 14234.56, 19006.18, 4211.73, 6326.47, 8447.19, 9125.42, 13707.35, 18302.25, 14741.06, 22142.65, 29565.17]
}

df = pd.DataFrame(data)

# 2. Chuẩn bị dữ liệu
X = df[['Speed', 'Feed', 'DOC']].values
y = df[['SR', 'MRR']].values

# Chuẩn hóa dữ liệu
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_scaled = x_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 3. Xây dựng và huấn luyện mô hình ANN
def create_ann_model():
    model = Sequential()
    model.add(Dense(6, input_dim=3, activation='tanh'))  # 6 nơ-ron trong lớp ẩn với hàm kích hoạt tanh
    model.add(Dense(2, activation='linear'))  # 2 đầu ra: SR và MRR
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Khởi tạo và huấn luyện mô hình
ann_model = create_ann_model()
history = ann_model.fit(X_train, y_train, epochs=5000, batch_size=5, verbose=0, validation_data=(X_test, y_test))

# Đánh giá mô hình
y_pred = ann_model.predict(X_test)
mse = np.mean((y_pred - y_test)**2)
print(f"Mean Squared Error trên tập kiểm tra: {mse:.6f}")

# 4. Thuật toán di truyền
class GeneticAlgorithm:
    def __init__(self, model, scaler_x, scaler_y, pop_size=50, max_gen=100, mutation_rate=0.1):
        self.model = model
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.mutation_rate = mutation_rate
        
        # Phạm vi giá trị cho mỗi thông số
        self.param_ranges = {
            'Speed': [80, 280],    # Tốc độ (m/min)
            'Feed': [0.06, 0.21],  # Tốc độ tiến (mm/rev)
            'DOC': [0.5, 1.0]      # Chiều sâu cắt (mm)
        }
        
    def create_individual(self):
        """Tạo một cá thể (bộ thông số) ngẫu nhiên"""
        return [
            round(random.uniform(self.param_ranges['Speed'][0], self.param_ranges['Speed'][1]), 0),
            round(random.uniform(self.param_ranges['Feed'][0], self.param_ranges['Feed'][1]), 2),
            round(random.uniform(self.param_ranges['DOC'][0], self.param_ranges['DOC'][1]), 2)
        ]
    
    def create_population(self):
        """Tạo quần thể ban đầu"""
        return [self.create_individual() for _ in range(self.pop_size)]
    
    def predict_outputs(self, individual):
        """Dự đoán SR và MRR cho một bộ thông số cụ thể"""
        # Chuẩn hóa thông số đầu vào
        x_normalized = self.scaler_x.transform([individual])
        # Dự đoán đầu ra chuẩn hóa
        y_normalized = self.model.predict(x_normalized, verbose=0)
        # Chuyển đổi ngược để lấy giá trị thực
        y_actual = self.scaler_y.inverse_transform(y_normalized)
        return y_actual[0]  # [SR, MRR]
    
    def fitness(self, individual, bias=0.5):
        """Tính độ thích nghi dựa trên SR và MRR với độ thiên vị bias"""
        sr, mrr = self.predict_outputs(individual)
        
        # Chuẩn hóa đầu ra (giả định giá trị tối đa và tối thiểu từ dữ liệu)
        min_sr, max_sr = min(df['SR']), max(df['SR'])
        min_mrr, max_mrr = min(df['MRR']), max(df['MRR'])
        
        norm_sr = (max_sr - sr) / (max_sr - min_sr)  # Đảo ngược vì chúng ta muốn SR tối thiểu
        norm_mrr = (mrr - min_mrr) / (max_mrr - min_mrr)
        
        # Áp dụng độ thiên vị (bias)
        # bias=1.0 ưu tiên hoàn toàn cho SR, bias=0.0 ưu tiên hoàn toàn cho MRR
        fitness_value = bias * norm_sr + (1 - bias) * norm_mrr
        
        return fitness_value
    
    def selection(self, population, fitnesses):
        """Chọn lọc các cá thể theo phương pháp bánh xe roulette"""
        # Xử lý giá trị đặc biệt trong fitnesses
        fitnesses = np.array(fitnesses)
        
        # Xử lý trường hợp có giá trị âm
        min_fitness = np.min(fitnesses)
        if min_fitness < 0:
            # Dịch chuyển tất cả giá trị lên một mức để đảm bảo không âm
            fitnesses = fitnesses - min_fitness + 1e-10
        
        # Đảm bảo không có giá trị 0
        fitnesses = np.maximum(fitnesses, 1e-10)
        
        # Tính tổng độ thích nghi
        total_fitness = np.sum(fitnesses)
        
        # Tính xác suất lựa chọn và chuẩn hóa để đảm bảo tổng = 1
        selection_probs = fitnesses / total_fitness
        
        # Kiểm tra và sửa nếu tổng không đúng 1 do lỗi số học
        if not np.isclose(np.sum(selection_probs), 1.0):
            selection_probs = selection_probs / np.sum(selection_probs)
        
        # Đảm bảo không có giá trị NaN hoặc inf
        if np.isnan(selection_probs).any() or np.isinf(selection_probs).any():
            selection_probs = np.ones(len(population)) / len(population)
        
        # Kiểm tra lần cuối
        selection_probs = np.nan_to_num(selection_probs)
        selection_probs = selection_probs / np.sum(selection_probs)
        
        # Chọn một cá thể bằng phương pháp bánh xe roulette
        selected_index = np.random.choice(len(population), p=selection_probs)
        return population[selected_index]
        
    def crossover(self, parent1, parent2):
        """Lai ghép hai cá thể để tạo ra con cái"""
        # Lai ghép với một điểm cắt ngẫu nhiên
        crossover_point = random.randint(1, 2)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child
    
    def mutate(self, individual):
        """Đột biến một cá thể với xác suất mutation_rate"""
        mutated_individual = individual.copy()  # Tạo bản sao để tránh thay đổi trực tiếp
        if random.random() < self.mutation_rate:
            # Chọn ngẫu nhiên một thông số để đột biến
            param_index = random.randint(0, 2)
            param_name = list(self.param_ranges.keys())[param_index]
            
            # Đột biến thông số bằng cách chọn một giá trị mới trong phạm vi cho phép
            mutated_individual[param_index] = round(random.uniform(
                self.param_ranges[param_name][0], 
                self.param_ranges[param_name][1]
            ), 2 if param_index > 0 else 0)  # Làm tròn tùy theo loại thông số
        
        return mutated_individual
    
    def optimize(self, bias=0.5):
        """Thực hiện quá trình tối ưu hóa với độ thiên vị bias"""
        population = self.create_population()
        best_individual = None
        best_fitness = -float('inf')
        best_sr = None
        best_mrr = None
        
        for generation in range(self.max_gen):
            # Đánh giá độ thích nghi của mỗi cá thể
            fitnesses = [self.fitness(individual, bias) for individual in population]
            
            # Đảm bảo không có giá trị NaN hoặc Inf trong fitnesses
            fitnesses = np.array(fitnesses)
            valid_indices = ~np.isnan(fitnesses) & ~np.isinf(fitnesses)
            
            if np.any(valid_indices):
                # Chỉ xét các cá thể có fitness hợp lệ
                valid_population = [ind for i, ind in enumerate(population) if valid_indices[i]]
                valid_fitnesses = fitnesses[valid_indices]
                
                if len(valid_population) > 0:
                    # Tìm cá thể tốt nhất trong thế hệ hiện tại
                    current_best_index = np.argmax(valid_fitnesses)
                    current_best = valid_population[current_best_index]
                    current_best_fitness = valid_fitnesses[current_best_index]
                    
                    # Cập nhật cá thể tốt nhất toàn cục
                    if current_best_fitness > best_fitness:
                        best_individual = current_best
                        best_fitness = current_best_fitness
                        best_sr, best_mrr = self.predict_outputs(best_individual)
                    
                    # Hiển thị thông tin về thế hệ hiện tại
                    if generation % 10 == 0:
                        sr, mrr = self.predict_outputs(current_best)
                        print(f"Thế hệ {generation}: Cá thể tốt nhất = {current_best}, SR = {sr:.2f}, MRR = {mrr:.2f}, Độ thích nghi = {current_best_fitness:.4f}")
                    
                    # Tạo thế hệ mới
                    new_population = []
                    
                    # Giữ lại cá thể tốt nhất (elitism)
                    new_population.append(current_best)
                    
                    # Tạo phần còn lại của quần thể mới
                    while len(new_population) < self.pop_size:
                        # Sử dụng chỉ những cá thể hợp lệ cho selection
                        parent1 = self.selection(valid_population, valid_fitnesses)
                        parent2 = self.selection(valid_population, valid_fitnesses)
                        
                        child = self.crossover(parent1, parent2)
                        child = self.mutate(child)
                        
                        new_population.append(child)
                    
                    population = new_population
                else:
                    # Tạo quần thể mới nếu tất cả cá thể đều không hợp lệ
                    population = self.create_population()
            else:
                # Tạo quần thể mới nếu tất cả cá thể đều không hợp lệ
                population = self.create_population()
        
        # Đảm bảo luôn có giá trị trả về
        if best_individual is None:
            best_individual = self.create_individual()
            best_sr, best_mrr = self.predict_outputs(best_individual)
            best_fitness = self.fitness(best_individual, bias)
            
        return best_individual, best_sr, best_mrr, best_fitness

# 5. Chạy thuật toán di truyền với các giá trị bias khác nhau
ga = GeneticAlgorithm(ann_model, x_scaler, y_scaler, pop_size=100, max_gen=100, mutation_rate=0.15)

bias_values = {
    "100% bias cho SR": 1.0,  # Tập trung hoàn toàn vào tối thiểu SR
    "75% bias cho SR": 0.75,  # Thiên vị cao cho SR
    "50% bias (cân bằng)": 0.5,  # Cân bằng giữa SR và MRR
    "25% bias cho SR": 0.25,  # Thiên vị thấp cho SR (ưu tiên MRR)
    "0% bias cho SR": 0.0,    # Tập trung hoàn toàn vào tối đa MRR
}

results = {}

print("\n=== KẾT QUẢ TỐI ƯU HÓA VỚI CÁC GIÁ TRỊ BIAS KHÁC NHAU ===\n")

for label, bias in bias_values.items():
    print(f"Đang tối ưu hóa với {label} (bias = {bias})...")
    best_params, best_sr, best_mrr, best_fitness = ga.optimize(bias)
    
    results[label] = {
        "Tốc độ (m/min)": best_params[0],
        "Tốc độ tiến (mm/rev)": best_params[1],
        "Chiều sâu cắt (mm)": best_params[2],
        "Độ nhám bề mặt (SR)": best_sr,
        "Tốc độ loại bỏ vật liệu (MRR)": best_mrr,
        "Độ thích nghi": best_fitness
    }
    print(f"\nKết quả với {label}:")
    print(f"Thông số tối ưu: Tốc độ = {best_params[0]:.1f} m/min, Tốc độ tiến = {best_params[1]:.2f} mm/rev, Chiều sâu cắt = {best_params[2]:.2f} mm")
    print(f"SR = {best_sr:.4f}, MRR = {best_mrr:.2f}, Độ thích nghi = {best_fitness:.4f}")
    print("-" * 50)

# 6. In kết quả theo định dạng bảng
print("\n" + "="*60)
print("Table 4")
print("Operating parameters at various bias values.")
print("="*60)

# Tạo bảng kết quả cho 100% bias và 50% bias
print("{:<30} {:<30}".format("100% bias on surface finish", "50% bias on surface finish"))
print("{:<5} {:<15} {:<10} {:<5} {:<15} {:<10}".format("", "Speed", str(int(results["100% bias cho SR"]["Speed"])), "", "Speed", str(int(results["50% bias (cân bằng)"]["Speed"]))))
print("{:<5} {:<15} {:<10} {:<5} {:<15} {:<10}".format("", "Feed", str(round(results["100% bias cho SR"]["Feed"], 2)), "", "Feed", str(round(results["50% bias (cân bằng)"]["Feed"], 2))))
print("{:<5} {:<15} {:<10} {:<5} {:<15} {:<10}".format("", "Depth of Cut", str(round(results["100% bias cho SR"]["DOC"], 2)), "", "Depth of Cut", str(round(results["50% bias (cân bằng)"]["DOC"], 2))))

# Tạo bảng kết quả cho 75% bias và 25% bias
print("{:<30} {:<30}".format("75% bias on surface finish", "25% bias on surface finish"))
print("{:<5} {:<15} {:<10} {:<5} {:<15} {:<10}".format("", "Speed", str(int(results["75% bias cho SR"]["Speed"])), "", "Speed", str(int(results["25% bias cho SR"]["Speed"]))))
print("{:<5} {:<15} {:<10} {:<5} {:<15} {:<10}".format("", "Feed", str(round(results["75% bias cho SR"]["Feed"], 2)), "", "Feed", str(round(results["25% bias cho SR"]["Feed"], 2))))
print("{:<5} {:<15} {:<10} {:<5} {:<15} {:<10}".format("", "Depth of Cut", str(round(results["75% bias cho SR"]["DOC"], 2)), "", "Depth of Cut", str(round(results["25% bias cho SR"]["DOC"], 2))))

print("="*60)

# 7. In kết quả về thông số tối ưu cuối cùng (lựa chọn 25% bias)
print("\nKết quả tối ưu cuối cùng:")
print("Thông số tối ưu với bias = 25% (ưu tiên MRR nhưng vẫn quan tâm đến SR):")
print(f"Tốc độ cắt: {int(results['25% bias cho SR']['Speed'])} m/min")
print(f"Tốc độ tiến: {results['25% bias cho SR']['Feed']:.2f} mm/rev")
print(f"Chiều sâu cắt: {results['25% bias cho SR']['DOC']:.2f} mm")
print(f"Đạt được: SR = {results['25% bias cho SR']['SR']:.2f}, MRR = {results['25% bias cho SR']['MRR']:.2f}")

# 7. Biểu diễn trực quan kết quả
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Vẽ biểu đồ SR theo bias
ax1.plot(list(bias_values.values()), [results[label]["Độ nhám bề mặt (SR)"] for label in bias_values.keys()], 'bo-')
ax1.set_xlabel('Độ thiên vị cho SR')
ax1.set_ylabel('Độ nhám bề mặt (SR)')
ax1.set_title('Độ nhám bề mặt tối ưu theo độ thiên vị')
ax1.grid(True)

# Vẽ biểu đồ MRR theo bias
ax2.plot(list(bias_values.values()), [results[label]["Tốc độ loại bỏ vật liệu (MRR)"] for label in bias_values.keys()], 'ro-')
ax2.set_xlabel('Độ thiên vị cho SR')
ax2.set_ylabel('Tốc độ loại bỏ vật liệu (MRR)')
ax2.set_title('MRR tối ưu theo độ thiên vị')
ax2.grid(True)

plt.tight_layout()
plt.savefig('optimization_results.png')
plt.show()