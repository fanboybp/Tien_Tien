import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import matplotlib.pyplot as plt

# Bước 1: Đọc dữ liệu từ file Abalone.data vào một DataFrame
df=pd.read_csv('C:\\Users\\gatva\\Documents\\Zalo Received Files\\nhakc.csv')
df.drop(df.columns[1:2], axis=1, inplace=True)
# Bước 2: Chuẩn bị dữ liệu cho việc uấn luyện mô hình
X = df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors']]  # Đặc trưng
y = df['price']  # tuổi

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Bước 3: Xây dựng và huấn luyện mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

# Bước 4: Đánh giá mô hình
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)
#lấy giá trị  của ng 30 tuổi có trên lương trên 50k$
dem = df.loc[(df['price'] < 30000) & (df['bedrooms'] >=3)].shape[0]
dem2 = df.loc[(df['price'] < 40000) & (df['bedrooms'] >= 3) & df['bathrooms'] >= 3 ].shape[0]
print("Mean Absolute Error:", mae)  #sai số tuyệt đốitrung bình
print("Mean Squared Error:", mse)   #Sai số bình phương trung bình
print("Median Absolute Error:", medae) #Sai số tuyệt đối trung vị
#lấy giá trị  của ng 30 tuổi có trên lương trên 50k$
print("Số người dưới 30 tuổi mà có thu nhập trên 50000k $ :"+str(dem))
print("Số người dưới 30 tuổi mà có kinh nhiệm trên 5 năm :"+str(dem2))

# Tạo DataFrame từ các giá trị tối thiểu, tối đa và trung bình
summary_table = pd.DataFrame({'Min': df.min(), 'Max': df.max(), 'Mean': df.mean()}, index=['price', 'bedrooms', 'bathrooms','sqft_living','sqft_lot','floors'])

# Hiển thị bảng với các thuộc tính nằm theo hàng ngang (chuyển vị)
summary_table_transposed = summary_table.transpose()
print(summary_table_transposed)
with open('C:\\Users\\gatva\\Documents\\Zalo Received Files\\fileoutput.txt','w',encoding='utf-8') as file:
    file.write("Số người dưới 30 tuổi mà có thu nhập trên 50000k $: " + str(dem) + "\n")
    file.write("Số người dưới 30 tuổi mà có kinh nghiệm trên 5 năm: " + str(dem2) + "\n\n")
    file.write("Bảng tóm tắt:\n")
    file.write(summary_table_transposed.to_string())

# Bước 5: Biểu diễn các tương quan bằng biểu đồ
# Tạo khung hình và các subplot
fig, axs = plt.subplots(2, 2, figsize=(15, 7))

# Biểu đồ 1: Tuổi vs Kinh nhiệm
axs[0, 0].scatter(df["price"], df["bedrooms"], c='brown')
axs[0, 0].set_title('Tuổi vs Kinh nhiệm')
axs[0, 0].set_xlabel('Tuổi (Age)')
axs[0, 0].set_ylabel('Kinh nhiệm (Exprrience)')

# Biểu đồ 2: Tuổi vs thu nhập cá nhân
axs[0, 1].scatter(df["price"], df["bathrooms"], c='green')
axs[0, 1].set_title('Tuổi vs thu nhập cá nhân')
axs[0, 1].set_xlabel('Tuổi (Rings)')
axs[0, 1].set_ylabel('Thu nhập cá nhân (Income)')

# Biểu đồ 3: Tuổi vs chi tiêu hàng tháng
axs[1, 0].scatter(df["price"], df["sqft_living"], c='yellow')
axs[1, 0].set_title('Tuổi vs Trung bình chi tiêu hàng tháng')
axs[1, 0].set_xlabel('Tuổi (Age)')
axs[1, 0].set_ylabel('Trung bình chi tiêu hàng tháng (CCAvg)')

# Biểu đồ 4: Thu nhập với chi tiêu hàng tháng
axs[1, 1].scatter(df["sqft_living"], df["floors"], c='red')
axs[1, 1].set_title('thu nhập cá nhân vs trung bình chi tiêu hàng tháng')
axs[1, 1].set_xlabel('Thu nhập cá nhân (Income)')
axs[1, 1].set_ylabel('Trung bình chi tiêu hàng tháng (CCAvg)')

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()