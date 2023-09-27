import numpy as np
import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def color_balance(image):
    # Tính toán histogram của ảnh
    histogram = [np.zeros(256, dtype=np.int32) for _ in range(3)]
    for i in range(3):
        histogram[i] = cv2.calcHist([image], [i], None, [256], [0, 256])

    # Tính toán histogram tích lũy
    cumulative_histogram = [np.zeros(256, dtype=np.float32) for _ in range(3)]
    for i in range(3):
        cumulative_histogram[i] = np.cumsum(histogram[i])

    # Chuẩn hóa histogram tích lũy
    total_pixels = image.shape[0] * image.shape[1]
    normalized_histogram = [np.zeros(256, dtype=np.float32) for _ in range(3)]
    for i in range(3):
        normalized_histogram[i] = cumulative_histogram[i] / total_pixels

    # Tạo ánh xạ màu mới
    mapping = [np.zeros(256, dtype=np.uint8) for _ in range(3)]
    for i in range(3):
        mapping[i] = np.round(normalized_histogram[i] * 255)

    # Áp dụng ánh xạ màu mới
    result = np.copy(image)
    for i in range(3):
        result[:, :, i] = cv2.LUT(image[:, :, i], mapping[i])

    return result

# Mở hộp thoại chọn tập tin
Tk().withdraw()
filename = askopenfilename()

# Đọc ảnh
image = cv2.imread(filename)

# Cân bằng màu
balanced_image = color_balance(image)

# Hiển thị ảnh gốc và ảnh đã cân bằng màu
cv2.imshow('Original Image', image)
cv2.imshow('Color Balanced Image', balanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
