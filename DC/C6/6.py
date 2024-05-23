import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import matplotlib.pyplot as plt

# Bước 1: Đọc dữ liệu từ file CSV và bỏ cột 'GallusID'
df = pd.read_csv('C:\\Users\\gatva\\Documents\\Zalo Received Files\\khithaixe.csv')
df.drop(['MODEL'], axis=1, inplace=True)

# Bước 2: Chuẩn bị dữ liệu cho việc huấn luyện mô hình
X = df[['ENGINE_SIZE', 'CYLINDERS', 'FUEL_CONSUMPTION*']]
y = df['CO2_EMISSIONS']  # khí thải

# Sử dụng One-Hot Encoding để chuyển đổi các đặc trưng phân loại thành các đặc trưng số
#X = pd.get_dummies(X, columns=['GallusBreed', 'GallusEggColor', 'GallusCombType', 'GallusClass', 'GallusLegShanksColor', 'GallusBeakColor', 'GallusEarLobesColor', 'GallusPlumage'])

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Bước 3: Xây dựng và huấn luyện mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

# Bước 4: Đánh giá mô hình
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)

# In kết quả đánh giá
print("Mean Absolute Error:", mae)  # Sai số tuyệt đối trung bình
print("Mean Squared Error:", mse)   # Sai số bình phương trung bình
print("Median Absolute Error:", medae) # Sai số tuyệt đối trung vị

# Lấy giá trị của gà có trọng lượng lớn hơn 3000 và màu trứng là 'Brown'
dem = df.loc[(df['CYLINDERS'] ==4) & (df['CO2_EMISSIONS'] > 200)].shape[0]
# Lấy giá trị của gà có tuổi lớn hơn 800 và màu trứng là 'Brown'
dem2 = df.loc[(df['ENGINE_SIZE'] == 2.0) & (df['FUEL_CONSUMPTION*'] > 200)].shape[0]

print("Gà có trọng lượng lớn hơn 3000 và màu trứng là 'Brown':", dem)
print("Gà có tuổi lớn hơn 800 và màu trứng là 'Brown':", dem2)

# Tạo DataFrame từ các giá trị tối thiểu, tối đa và trung bình
summary_table = df[['ENGINE_SIZE', 'CYLINDERS', 'FUEL_CONSUMPTION*', 'CO2_EMISSIONS']].agg(['min', 'max', 'mean'])

# Hiển thị bảng với các thuộc tính nằm theo hàng ngang (chuyển vị)
summary_table_transposed = summary_table.transpose()
print(summary_table_transposed)

# Ghi kết quả vào file
with open('C:\\Users\\gatva\\Documents\\Zalo Received Files\\fileoutput.txt', 'w', encoding='utf-8') as file:
    file.write("Gà có trọng lượng lớn hơn 3000 và màu trứng là 'Brown': " + str(dem) + "\n")
    file.write("Gà có tuổi lớn hơn 800 và màu trứng là 'Brown': " + str(dem2) + "\n\n")
    file.write("Bảng tóm tắt:\n")
    file.write(summary_table_transposed.to_string())

# Bước 5: Biểu diễn các tương quan bằng biểu đồ
# Tạo khung hình và các subplot
fig, axs = plt.subplots(2, 2, figsize=(15, 7))

# Biểu đồ 1: Tuổi vs Kinh nhiệm
axs[0, 0].scatter(df["CO2_EMISSIONS"], df["ENGINE_SIZE"], c='brown')
axs[0, 0].set_title('Tuổi vs Giá nhà')
axs[0, 0].set_xlabel('Tuổi (Age)')
axs[0, 0].set_ylabel('Giá (Price)')

# Biểu đồ 2: Giá nhà với khoảng cách ga tàu
axs[0, 1].scatter(df["CO2_EMISSIONS"], df["FUEL_CONSUMPTION*"], c='green')
axs[0, 1].set_title('Giá nhà với khoảng cách ga tàu')
axs[0, 1].set_xlabel('Giá nhà (Price)')
axs[0, 1].set_ylabel('Khoảng cách tới ga tàu')

# Biểu đồ 3: Giá nhà với số cửa hàng tiện lợi
axs[1, 0].scatter(df["CO2_EMISSIONS"], df["CYLINDERS"], c='yellow')
axs[1, 0].set_title('Giá nhà với số cửa hàng tiện lợi')
axs[1, 0].set_xlabel('Giá nhà ')
axs[1, 0].set_ylabel('Số cửa hàng tiện lợi')

# Biểu đồ 4: Số cửa hàng tiện lợi với khoảng cách ga tàu
axs[1, 1].scatter(df["CYLINDERS"], df["FUEL_CONSUMPTION*"], c='red')
axs[1, 1].set_title('Số cửa hàng tiện lợi với khoảng cách ga tàu')
axs[1, 1].set_xlabel('Số của hàng tiện lợi')
axs[1, 1].set_ylabel('Khoảng cách tới ga tàu')

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()