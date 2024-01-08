import matplotlib.pyplot as plt
import numpy as np

def gamma_correction(input_value, gamma):
    return input_value ** gamma

# Tạo một loạt giá trị đầu vào từ 0 đến 1
input_values = np.linspace(0, 1, 256)

# Tạo các giá trị đầu ra tương ứng với gamma từ 0.1 đến 2.0
gamma_values = [0.1, 0.5, 1.0, 1.5, 2.0]
output_values = []

for gamma in gamma_values:
    output_values.append(gamma_correction(input_values, gamma))

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))

for i, gamma in enumerate(gamma_values):
    plt.plot(input_values, output_values[i], label=f'Gamma = {gamma}')

plt.title('Mối Quan Hệ Giữa Đầu Vào và Đầu Ra của Gamma Correction')
plt.xlabel('Đầu vào (I)')
plt.ylabel('Đầu ra (I\')')
plt.legend()
plt.grid(True)
plt.show()
