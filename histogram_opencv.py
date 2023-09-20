import cv2
import matplotlib.pyplot as plt
from tkinter import filedialog
import tkinter as tk

# Hàm rescale
def rescale(frame, scale=1):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

# Hàm mở và xử lý hình ảnh
def open_image():
    global img_path
    img_path = filedialog.askopenfilename()  # Cho phép người dùng chọn đường dẫn
    if img_path:
        img_org = cv2.imread(img_path)
        img = rescale(img_org)
        cv2.imshow('ORG', img)

        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img_equa = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        cv2.imshow('Equalized', img_equa)

        hist_org = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_equa = cv2.calcHist([img_equa], [0], None, [256], [0, 256])

        plot1 = plt.figure(1)
        plt.title('Biểu đồ Histogram')
        plt.plot(hist_org)

        plot2 = plt.figure(2)
        plt.title('Equalized Histogram')
        plt.plot(hist_equa)

        plt.show()

# Khởi tạo cửa sổ Tkinter
root = tk.Tk()
root.title("Open and Process Image")

# Tạo nút để mở file
open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack()

# Khởi chạy giao diện Tkinter
root.mainloop()
