import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from deap import base, creator, tools, algorithms
import random

# Bước 1: Đọc dữ liệu
data = pd.read_csv('data copy.csv')

# Tách dữ liệu thành input (X) và output (y)
X = data[['I', 'Ton', 'Toff', 'Wire Feed']].values
y = data[['MRR', 'SR', 'Overcut']].values

# Chuẩn hóa dữ liệu
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Bước 2: Xây dựng mô hình ANN
model = Sequential([
    Dense(64, input_dim=X_scaled.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(3, activation='linear')  # 3 outputs: MRR, SR, Overcut
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Bước 3: Huấn luyện mô hình
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Bước 4: Đánh giá mô hình
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_original = scaler_y.inverse_transform(y_test)

# Tính các chỉ số đánh giá
mae = mean_absolute_error(y_test_original, y_pred)
r2 = r2_score(y_test_original, y_pred, multioutput='uniform_average')
mse = mean_squared_error(y_test_original, y_pred)

# print(f"Mean Absolute Error: {mae}")
# print(f"R-Square: {r2}")
# print(f"Standard Error (MSE): {mse}")

# Bước 5: Xây dựng hàm mục tiêu cho GA
def objective_function(params):
    params_scaled = scaler_X.transform([params])
    predictions_scaled = model.predict(params_scaled)
    predictions = scaler_y.inverse_transform(predictions_scaled)[0]
    
    mrr = predictions[0]  # MRR cần cao nhất
    sr = predictions[1]   # SR cần thấp nhất
    overcut = predictions[2]  # Overcut cần thấp nhất
    
    # Hàm mục tiêu: Cân bằng giữa MRR, SR và Overcut
    fitness = -mrr + 0.5 * sr + 0.5 * overcut
    return fitness,

# Bước 6: Thiết lập và chạy GA
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Định nghĩa giới hạn cho các thông số đầu vào
toolbox.register("attr_I", random.uniform, min(data['I']), max(data['I']))
toolbox.register("attr_Ton", random.uniform, min(data['Ton']), max(data['Ton']))
toolbox.register("attr_Toff", random.uniform, min(data['Toff']), max(data['Toff']))
toolbox.register("attr_WireFeed", random.uniform, min(data['Wire Feed']), max(data['Wire Feed']))

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_I, toolbox.attr_Ton, toolbox.attr_Toff, toolbox.attr_WireFeed), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", objective_function)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Chạy GA
population = toolbox.population(n=50)
NGEN = 40  # Số thế hệ
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    fits = map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

# Kết quả tối ưu
best_ind = tools.selBest(population, k=1)[0]
optimal_params = best_ind
optimal_fitness = best_ind.fitness.values[0]

print(f"Mean Absolute Error: {mae}")
print(f"R-Square: {r2}")
print(f"Standard Error (MSE): {mse}")

print(f"------")

print("Optimal Parameters:", optimal_params)
print("Optimal Fitness:", optimal_fitness)

