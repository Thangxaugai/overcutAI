import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import random

# 1. Đọc dữ liệu từ file CSV
file_path = 'data.csv'  # Đường dẫn tới file CSV
df = pd.read_csv(file_path)

# 2. Chuẩn bị dữ liệu
X = df[['Speed', 'Feed', 'DOC']].values
y = df[['SR', 'MRR']].values

# Chuẩn hóa dữ liệu
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_scaled = x_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra - sử dụng tỷ lệ nhỏ hơn để huấn luyện nhanh hơn
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 3. Xây dựng và huấn luyện mô hình ANN - giảm epochs và tăng batch_size
def create_ann_model():
    model = Sequential()
    model.add(Dense(6, input_dim=3, activation='tanh'))
    model.add(Dense(2, activation='linear'))
    # Sử dụng học tập với tốc độ cao hơn để hội tụ nhanh hơn
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Khởi tạo và huấn luyện mô hình - giảm epochs xuống để chạy nhanh hơn
print("Đang huấn luyện mô hình ANN...")
ann_model = create_ann_model()
# Giảm epochs xuống và tăng batch_size để tăng tốc độ
history = ann_model.fit(X_train, y_train, epochs=500, batch_size=10, verbose=0, validation_data=(X_test, y_test))

# Đánh giá mô hình
y_pred = ann_model.predict(X_test, verbose=0)
mse = np.mean((y_pred - y_test)**2)
print(f"Mean Squared Error trên tập kiểm tra: {mse:.6f}")

# Tiền tính toán các giá trị min/max cho SR và MRR để tránh tính đi tính lại
min_sr, max_sr = min(df['SR']), max(df['SR'])
min_mrr, max_mrr = min(df['MRR']), max(df['MRR'])

# 4. Thuật toán di truyền tối ưu hóa
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
        
        # Định nghĩa trước các giá trị có thể cho Speed để giảm việc tạo giá trị ngẫu nhiên
        self.speed_values = [80, 180, 280]
        
    def create_individual(self):
        """Tạo một cá thể (bộ thông số) ngẫu nhiên"""
        return [
            random.choice(self.speed_values),  # Chỉ chọn từ các giá trị được định nghĩa trước
            round(random.uniform(self.param_ranges['Feed'][0], self.param_ranges['Feed'][1]), 2),
            round(random.uniform(self.param_ranges['DOC'][0], self.param_ranges['DOC'][1]), 2)
        ]
    
    def create_population(self):
        """Tạo quần thể ban đầu"""
        return [self.create_individual() for _ in range(self.pop_size)]
    
    def predict_outputs_batch(self, population):
        """Dự đoán SR và MRR cho một quần thể (batch prediction)"""
        # Chuẩn hóa thông số đầu vào của toàn bộ quần thể cùng lúc
        x_normalized = self.scaler_x.transform(population)
        # Dự đoán đầu ra chuẩn hóa cho toàn bộ quần thể
        y_normalized = self.model.predict(x_normalized, verbose=0)
        # Chuyển đổi ngược để lấy giá trị thực
        y_actual = self.scaler_y.inverse_transform(y_normalized)
        return y_actual  # [[SR1, MRR1], [SR2, MRR2], ...]
    
    def fitness_batch(self, population, outputs, bias=0.5):
        """Tính độ thích nghi cho toàn bộ quần thể cùng lúc"""
        sr_values = outputs[:, 0]
        mrr_values = outputs[:, 1]
        
        # Chuẩn hóa đầu ra (sử dụng giá trị min/max đã tính trước)
        norm_sr = (max_sr - sr_values) / (max_sr - min_sr)  # Đảo ngược vì chúng ta muốn SR tối thiểu
        norm_mrr = (mrr_values - min_mrr) / (max_mrr - min_mrr)
        
        # Áp dụng độ thiên vị (bias)
        fitness_values = bias * norm_sr + (1 - bias) * norm_mrr
        
        return fitness_values
    
    def selection_tournament(self, population, fitnesses, tournament_size=3):
        """Chọn lọc các cá thể theo phương pháp giải đấu (tournament selection)"""
        # Chọn ngẫu nhiên tournament_size cá thể
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        
        # Chọn cá thể có fitness tốt nhất từ tournament
        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_idx]
    
    def crossover(self, parent1, parent2):
        """Lai ghép hai cá thể để tạo ra con cái"""
        # Lai ghép với một điểm cắt ngẫu nhiên
        crossover_point = random.randint(1, 2)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child
    
    def mutate(self, individual):
        """Đột biến một cá thể với xác suất mutation_rate"""
        mutated_individual = individual.copy()
        if random.random() < self.mutation_rate:
            # Chọn ngẫu nhiên một thông số để đột biến
            param_index = random.randint(0, 2)
            
            if param_index == 0:  # Speed
                mutated_individual[param_index] = random.choice(self.speed_values)
            else:
                param_name = list(self.param_ranges.keys())[param_index]
                mutated_individual[param_index] = round(random.uniform(
                    self.param_ranges[param_name][0], 
                    self.param_ranges[param_name][1]
                ), 2)
        
        return mutated_individual
    
    def optimize(self, bias=0.5):
        """Thực hiện quá trình tối ưu hóa với độ thiên vị bias"""
        population = self.create_population()
        best_individual = None
        best_fitness = -float('inf')
        best_sr = None
        best_mrr = None
        
        # Giảm số lượng thế hệ nếu cần thiết để chạy nhanh hơn
        for generation in range(self.max_gen):
            # Đánh giá độ thích nghi của toàn bộ quần thể cùng lúc
            outputs = self.predict_outputs_batch(population)
            fitnesses = self.fitness_batch(population, outputs, bias)
            
            # Tìm cá thể tốt nhất trong thế hệ hiện tại
            current_best_index = np.argmax(fitnesses)
            current_best = population[current_best_index]
            current_best_fitness = fitnesses[current_best_index]
            current_best_outputs = outputs[current_best_index]
            
            # Cập nhật cá thể tốt nhất toàn cục
            if current_best_fitness > best_fitness:
                best_individual = current_best
                best_fitness = current_best_fitness
                best_sr, best_mrr = current_best_outputs
            
            # Tạo thế hệ mới
            new_population = []
            
            # Elitism: giữ lại cá thể tốt nhất
            new_population.append(current_best)
            
            # Tạo phần còn lại của quần thể mới
            while len(new_population) < self.pop_size:
                # Sử dụng tournament selection thay vì roulette wheel
                parent1 = self.selection_tournament(population, fitnesses)
                parent2 = self.selection_tournament(population, fitnesses)
                
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                
                new_population.append(child)
            
            population = new_population
            
            # Thêm điều kiện dừng sớm nếu đã đạt được kết quả tốt
            if generation > 10 and best_fitness > 0.95:  # Điều chỉnh ngưỡng phù hợp
                break
        
        return best_individual, best_sr, best_mrr, best_fitness

# 5. Chạy thuật toán di truyền với các giá trị bias khác nhau - giảm số lượng quần thể và thế hệ
print("\nỨng dụng thuật toán di truyền để tìm thông số tối ưu...\n")
# Giảm pop_size và max_gen để chạy nhanh hơn
ga = GeneticAlgorithm(ann_model, x_scaler, y_scaler, pop_size=50, max_gen=30, mutation_rate=0.15)

bias_values = {
    "100% bias cho SR": 1.0,
    "75% bias cho SR": 0.75,
    "50% bias (cân bằng)": 0.5,
    "25% bias cho SR": 0.25,
    "0% bias cho SR": 0.0
}

results = {}

# Chạy tối ưu hóa với các giá trị bias khác nhau
for label, bias in bias_values.items():
    print(f"Đang tối ưu hóa với {label} (bias = {bias})...")
    best_params, best_sr, best_mrr, best_fitness = ga.optimize(bias)
    
    results[label] = {
        "Speed": best_params[0],
        "Feed": best_params[1],
        "DOC": best_params[2],
        "SR": best_sr,
        "MRR": best_mrr,
        "Fitness": best_fitness
    }

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

# 8. Biểu diễn trực quan kết quả
bias_values_list = list(bias_values.values())
sr_values = [results[label]["SR"] for label in bias_values.keys()]
mrr_values = [results[label]["MRR"] for label in bias_values.keys()]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(bias_values_list, sr_values, 'bo-')
plt.xlabel('Độ thiên vị cho SR')
plt.ylabel('Độ nhám bề mặt (SR)')
plt.title('Độ nhám bề mặt theo độ thiên vị')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(bias_values_list, mrr_values, 'ro-')
plt.xlabel('Độ thiên vị cho SR')
plt.ylabel('Tốc độ loại bỏ vật liệu (MRR)')
plt.title('MRR theo độ thiên vị')
plt.grid(True)

plt.tight_layout()
plt.savefig('optimization_results.png')