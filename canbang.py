from PIL import Image
import numpy as np

def adjust_gamma(image, gamma_r, gamma_g, gamma_b):
    # Chuyển đổi ảnh thành mảng numpy
    np_image = np.array(image)

    # Tách các kênh màu
    red_channel, green_channel, blue_channel = np_image[:, :, 0], np_image[:, :, 1], np_image[:, :, 2]

    # Áp dụng chỉ số gamma cho từng kênh màu
    adjusted_red = np.power(red_channel / 255.0, gamma_r) * 255.0
    adjusted_green = np.power(green_channel / 255.0, gamma_g) * 255.0
    adjusted_blue = np.power(blue_channel / 255.0, gamma_b) * 255.0

    # Ghép các kênh màu đã điều chỉnh lại thành ảnh mới
    adjusted_image = np.stack((adjusted_red, adjusted_green, adjusted_blue), axis=2).astype(np.uint8)

    # Chuyển đổi mảng numpy thành đối tượng Image
    adjusted_image = Image.fromarray(adjusted_image)

    return adjusted_image

# Đường dẫn đến ảnh cần điều chỉnh gamma
image_path = "image/anh.jpg"

# Đọc ảnh gốc
original_image = Image.open(image_path)

# Điều chỉnh gamma theo từng kênh màu (gamma_r, gamma_g, gamma_b)
gamma_adjusted_image = adjust_gamma(original_image, 1.2, 0.8, 0.9)

# Hiển thị ảnh gốc và ảnh đã điều chỉnh gamma
original_image.show()
gamma_adjusted_image.show()
