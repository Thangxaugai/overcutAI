import csv
import random

# Hàm tạo dữ liệu ngẫu nhiên
def generate_random_data(num_rows):
    data = []
    for i in range(1, num_rows + 1):
        I = random.randint(1, 12)  # Cột I: từ 1 đến 12
        Ton = random.randint(100, 130)  # Cột Ton: từ 100 đến 130
        Toff = random.randint(24, 60)  # Cột Toff: từ 24 đến 60
        Wire_Feed = random.randint(2, 12)  # Cột Wire Feed: từ 2 đến 12
        
        # Các giá trị MRR, SR, Overcut được tạo ngẫu nhiên trong khoảng hợp lý
        MRR = round(random.uniform(0.01, 0.5), 5)  # MRR: từ 0.01 đến 0.5
        SR = round(random.uniform(1.0, 4.0), 4)  # SR: từ 1.0 đến 4.0
        Overcut = round(random.uniform(0.01, 0.2), 3)  # Overcut: từ 0.01 đến 0.2
        
        # Thêm dòng dữ liệu vào danh sách
        data.append([i, I, Ton, Toff, Wire_Feed, MRR, SR, Overcut])
    
    return data

# Hàm ghi dữ liệu vào file CSV
def write_to_csv(filename, data):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Ghi header
        writer.writerow(["No", "I", "Ton", "Toff", "Wire Feed", "MRR", "SR", "Overcut"])
        # Ghi dữ liệu
        writer.writerows(data)

# Số dòng dữ liệu cần tạo
num_rows = 1000

# Tạo dữ liệu ngẫu nhiên
random_data = generate_random_data(num_rows)

# Ghi dữ liệu vào file CSV
output_filename = "random_data.csv"
write_to_csv(output_filename, random_data)

print(f"Đã tạo {num_rows} dòng dữ liệu ngẫu nhiên và lưu vào file '{output_filename}'.")