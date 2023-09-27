import cv2
from tkinter import filedialog
import numpy as np
from tkinter import Tk
from tkinter import Scale
from PIL import ImageTk
from PIL import Image
from tkinter import Label
###def nothing(x):
 ###   pass

def brightness(val):
    val = int(val) / 100


    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(float)

    hsv[:, :, 1] = hsv[:, :, 1] * val
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
    hsv[:, :, 2] = hsv[:, :, 2] * val
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255

    hsv = hsv.astype(np.uint8)
    res = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    
    cv2.imshow('image', res)

if __name__ == "__main__":
    # Chọn hình ảnh từ hộp thoại
    file_path = filedialog.askopenfilename()

    if file_path:
        img = cv2.imread(file_path)
        img = cv2.resize(img, (250, 300))  # Điều chỉnh kích thước ảnh

        # Tạo cửa sổ Tkinter để chứa nút "Lưu ảnh"
        root = Tk()

        # Tạo thanh trượt để điều chỉnh độ sáng
        brightness_scale = Scale(root, from_=0, to=255, orient="horizontal", label="Độ sáng",
                                showvalue=1, sliderlength=20, length=300, command=brightness)
        brightness_scale.pack()

        # Hiển thị ảnh gốc
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
        label = Label(root, image=photo)
        label.image = photo
        label.pack()

       
