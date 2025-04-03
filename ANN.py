import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from deap import base, creator, tools, algorithms
import random

# Tải dữ liệu
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Chuẩn bị dữ liệu
def prepare_data(data):
    # Đặc trưng đầu vào: I, Ton, Toff, Wire Feed
    X = data[['I', 'Ton', 'Toff', 'Wire Feed']].values
    # Đặc trưng đầu ra: MRR, SR, Overcut
    y = data[['MRR', 'SR', 'Overcut']].values
    
    # Chia tập dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Chuẩn hóa dữ liệu
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y

# Xây dựng mô hình ANN
def build_ann_model(input_dim, neurons=[64, 32, 16], dropout_rate=0.2, learning_rate=0.001):
    model = Sequential()
    
    # Layer đầu vào
    model.add(Dense(int(neurons[0]), activation='relu', input_dim=int(input_dim)))
    model.add(Dropout(dropout_rate))
    
    # Các hidden layer
    for n in neurons[1:]:
        model.add(Dense(int(n), activation='relu'))
        model.add(Dropout(dropout_rate))
    
    # Layer đầu ra (3 đầu ra: MRR, SR, Overcut)
    model.add(Dense(3, activation='linear'))
    
    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    
    return model

# Huấn luyện mô hình ANN
def train_ann_model(model, X_train, y_train, epochs=200, batch_size=16, validation_split=0.2):
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=0
    )
    
    return model, history

# Đánh giá mô hình
def evaluate_model(model, X_test, y_test, scaler_y):
    # Dự đoán
    y_pred_scaled = model.predict(X_test)
    
    # Khôi phục giá trị về thang đo gốc
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_original = scaler_y.inverse_transform(y_test)
    
    # Tính các metrics
    mae = []
    r2 = []
    rmse = []
    
    output_names = ['MRR', 'SR', 'Overcut']
    
    for i in range(3):
        mae.append(mean_absolute_error(y_test_original[:, i], y_pred[:, i]))
        r2.append(r2_score(y_test_original[:, i], y_pred[:, i]))
        rmse.append(np.sqrt(mean_squared_error(y_test_original[:, i], y_pred[:, i])))
        
        print(f"Metrics cho {output_names[i]}:")
        print(f"Mean Absolute Error: {mae[i]:.6f}")
        print(f"R-Square: {r2[i]:.6f}")
        print(f"Standard Error (RMSE): {rmse[i]:.6f}")
        print("-----------------------------------")
    
    return mae, r2, rmse, y_pred, y_test_original

# Tính điểm đánh giá để tối ưu hóa
def calculate_fitness(y_pred, weights=[1.0, -0.5, -0.5]):
    # Tính điểm dựa trên mục tiêu: tối đa hóa MRR, tối thiểu hóa SR và Overcut
    # weights: trọng số cho [MRR, SR, Overcut]
    mrr_score = y_pred[0] * weights[0]  # Tối đa hóa MRR nên trọng số dương
    sr_score = y_pred[1] * weights[1]   # Tối thiểu hóa SR nên trọng số âm
    overcut_score = y_pred[2] * weights[2]  # Tối thiểu hóa Overcut nên trọng số âm
    
    return mrr_score + sr_score + overcut_score

# Tối ưu hóa siêu tham số bằng thuật toán di truyền (GA)
def optimize_ann_with_ga(X_train, y_train, X_test, y_test, scaler_y):
    # Định nghĩa vấn đề tối ưu
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Định nghĩa gen
    # Gen 1-3: Số neurons trong mỗi layer (16-128)
    # Gen 4: Dropout rate (0.1-0.5)
    # Gen 5: Learning rate (0.0001-0.01)
    
    toolbox.register("attr_neurons1", random.randint, 16, 128)
    toolbox.register("attr_neurons2", random.randint, 16, 64)
    toolbox.register("attr_neurons3", random.randint, 8, 32)
    toolbox.register("attr_dropout", lambda: random.uniform(0.1, 0.5))
    toolbox.register("attr_lr", lambda: random.uniform(0.0001, 0.01))
    
    # Tạo cá thể và quần thể
    toolbox.register("individual", tools.initCycle, creator.Individual,
                    (toolbox.attr_neurons1, toolbox.attr_neurons2, toolbox.attr_neurons3, 
                     toolbox.attr_dropout, toolbox.attr_lr), n=1)
    
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Hàm đánh giá
    def evaluate(individual):
        neurons = [individual[0], individual[1], individual[2]]
        dropout_rate = individual[3]
        learning_rate = individual[4]
        
        # Xây dựng và huấn luyện mô hình
        input_dim = X_train.shape[1]
        model = build_ann_model(input_dim, neurons, dropout_rate, learning_rate)
        model, _ = train_ann_model(model, X_train, y_train, epochs=100, batch_size=16, validation_split=0.2)
        
        # Dự đoán
        y_pred_scaled = model.predict(X_test)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        
        # Tính giá trị trung bình của các đầu ra để đánh giá
        avg_mrr = np.mean(y_pred[:, 0])  # Giá trị trung bình của MRR
        avg_sr = np.mean(y_pred[:, 1])   # Giá trị trung bình của SR
        avg_overcut = np.mean(y_pred[:, 2])  # Giá trị trung bình của Overcut
        
        # Tính điểm fitness
        fitness = calculate_fitness([avg_mrr, avg_sr, avg_overcut])
        
        # Giải phóng bộ nhớ
        tf.keras.backend.clear_session()
        
        return (fitness,)
    
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Thuật toán GA
    population = toolbox.population(n=10)
    ngen = 5
    
    print("Bắt đầu tối ưu hóa siêu tham số bằng GA...")
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    final_pop, log = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=ngen, stats=stats, verbose=True)
    
    # Lấy cá thể tốt nhất
    best_individual = tools.selBest(final_pop, k=1)[0]
    best_neurons = [best_individual[0], best_individual[1], best_individual[2]]
    best_dropout = best_individual[3]
    best_lr = best_individual[4]
    
    print(f"Siêu tham số tốt nhất: neurons={best_neurons}, dropout={best_dropout:.3f}, learning_rate={best_lr:.6f}")
    
    # Huấn luyện lại mô hình với siêu tham số tốt nhất
    input_dim = X_train.shape[1]
    best_model = build_ann_model(input_dim, best_neurons, best_dropout, best_lr)
    best_model, _ = train_ann_model(best_model, X_train, y_train, epochs=200, batch_size=16, validation_split=0.2)
    
    return best_model

# Dự đoán với thông số mới
def predict_with_params(model, scaler_X, scaler_y, params):
    # Chuẩn hóa thông số đầu vào
    params_scaled = scaler_X.transform(np.array([params]))
    
    # Dự đoán
    pred_scaled = model.predict(params_scaled)
    
    # Khôi phục về thang đo gốc
    pred = scaler_y.inverse_transform(pred_scaled)
    
    return pred[0]

# Trực quan hóa kết quả
def visualize_results(history, y_test_original, y_pred):
    # Vẽ đồ thị loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Vẽ đồ thị so sánh giá trị thực và dự đoán
    output_names = ['MRR', 'SR', 'Overcut']
    plt.figure(figsize=(18, 6))
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.scatter(y_test_original[:, i], y_pred[:, i])
        plt.plot([min(y_test_original[:, i]), max(y_test_original[:, i])], 
                 [min(y_test_original[:, i]), max(y_test_original[:, i])], 'r--')
        plt.title(f'Actual vs Predicted: {output_names[i]}')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
    
    plt.tight_layout()
    plt.show()

# Hàm chính
def main(file_path):
    # Tải và chuẩn bị dữ liệu
    data = load_data(file_path)
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y = prepare_data(data)
    
    print("Kích thước dữ liệu:")
    print(f"X_train: {X_train_scaled.shape}")
    print(f"X_test: {X_test_scaled.shape}")
    print(f"y_train: {y_train_scaled.shape}")
    print(f"y_test: {y_test_scaled.shape}")
    
    # Xây dựng và huấn luyện mô hình ANN
    input_dim = X_train_scaled.shape[1]
    model = build_ann_model(input_dim)
    model, history = train_ann_model(model, X_train_scaled, y_train_scaled)
    
    # Đánh giá mô hình
    print("\nĐánh giá mô hình ban đầu:")
    mae, r2, rmse, y_pred, y_test_original = evaluate_model(model, X_test_scaled, y_test_scaled, scaler_y)
    
    # Tối ưu hóa siêu tham số bằng GA
    best_model = optimize_ann_with_ga(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, scaler_y)
    
    # Đánh giá mô hình tối ưu
    print("\nĐánh giá mô hình sau khi tối ưu bằng GA:")
    mae_best, r2_best, rmse_best, y_pred_best, y_test_original = evaluate_model(best_model, X_test_scaled, y_test_scaled, scaler_y)
    
    # Thử nghiệm với bộ thông số mới
    print("\nThử nghiệm với bộ thông số mới:")
    new_params = [1, 110, 28, 3]  # I, Ton, Toff, Wire Feed
    predictions = predict_with_params(best_model, scaler_X, scaler_y, new_params)
    
    print(f"Thông số đầu vào: I={new_params[0]}, Ton={new_params[1]}, Toff={new_params[2]}, Wire Feed={new_params[3]}")
    print(f"Dự đoán: MRR={predictions[0]:.6f}, SR={predictions[1]:.6f}, Overcut={predictions[2]:.6f}")
    
    # Tìm bộ thông số tối ưu (tối đa MRR, tối thiểu SR và Overcut)
    print("\nTìm bộ thông số tối ưu:")
    
    # Tạo lưới thông số để tìm kiếm
    I_values = [1, 2, 3]
    Ton_values = np.linspace(100, 120, 5)
    Toff_values = np.linspace(20, 35, 5)
    WF_values = [2, 3, 4, 5]
    
    best_params = None
    best_fitness = float('-inf')
    
    for I in I_values:
        for Ton in Ton_values:
            for Toff in Toff_values:
                for WF in WF_values:
                    params = [I, Ton, Toff, WF]
                    predictions = predict_with_params(best_model, scaler_X, scaler_y, params)
                    fitness = calculate_fitness(predictions)
                    
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_params = params
                        best_pred = predictions
    
    print(f"Bộ thông số tối ưu: I={best_params[0]}, Ton={best_params[1]:.2f}, Toff={best_params[2]:.2f}, Wire Feed={best_params[3]}")
    print(f"Dự đoán với bộ thông số tối ưu: MRR={best_pred[0]:.6f}, SR={best_pred[1]:.6f}, Overcut={best_pred[2]:.6f}")
    
    # Lưu mô hình
    best_model.save('best_ann_model.h5')
    print("Đã lưu mô hình tối ưu vào file 'best_ann_model.h5'")
    
    return best_model, history, y_test_original, y_pred_best

if __name__ == "__main__":
    file_path = 'data.csv'  # Thay thế bằng đường dẫn thực tế đến file dữ liệu
    best_model, history, y_test_original, y_pred = main(file_path)
    
    # Trực quan hóa kết quả
    visualize_results(history, y_test_original, y_pred)