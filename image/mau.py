import cv2
import numpy as np
import tkinter as tk
from tkinter import colorchooser
from tkinter import filedialog
from PIL import Image, ImageTk

def color_adjustment(image, object_color, target_color):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Chuyển đổi màu từ (R, G, B) sang (H, S, V)
    lower_bound = np.array(object_color[0:3])
    upper_bound = np.array(object_color[3:6])

    # Mở rộng chiều dài của lower_bound và upper_bound để khớp với chiều dài của hsv_image
    lower_bound = lower_bound.reshape(1, 1, 3)
    upper_bound = upper_bound.reshape(1, 1, 3)

    object_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    hsv_image[np.where(object_mask)] = target_color

    result_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return result_image

class ColorAdjustmentApp:
    def __init__(self, root, image_path):
        self.root = root
        self.root.title("Color Adjustment App")

        # Đọc ảnh
        self.input_image = cv2.imread(image_path)

        # Khởi tạo object_color và target_color
        self.object_color = [0, 0, 0, 0, 0, 0]
        self.target_color = [0, 0, 0]

        # Hiển thị ảnh gốc
        self.display_original_image()

        # Button để chọn vật thể và màu sắc
        select_button = tk.Button(root, text="Select Object Color", command=self.select_object_color)
        select_button.pack()

        # Button để chọn màu sắc mong muốn
        choose_color_button = tk.Button(root, text="Choose Target Color", command=self.choose_target_color)
        choose_color_button.pack()

        # Button để thực hiện chỉnh sửa màu sắc
        adjust_button = tk.Button(root, text="Adjust Color", command=self.adjust_color)
        adjust_button.pack()

    def display_original_image(self):
        # Chuyển đổi ảnh OpenCV sang định dạng hỗ trợ bởi Tkinter
        original_image = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2RGB)
        original_image = Image.fromarray(original_image)
        original_image = ImageTk.PhotoImage(original_image)

        # Hiển thị ảnh gốc
        label = tk.Label(root, image=original_image)
        label.image = original_image
        label.pack()

    def select_object_color(self):
        color = colorchooser.askcolor(title="Select Object Color")
        # Lưu trữ màu được chọn trong biến instance để sử dụng sau này
        self.object_color = [int(i) for i in color[0]]

    def choose_target_color(self):
        color = colorchooser.askcolor(title="Choose Target Color")
        # Lưu trữ màu được chọn trong biến instance để sử dụng sau này
        self.target_color = [int(i) for i in color[0]]

    def adjust_color(self):
        # Thực hiện chỉnh sửa màu sắc
        output_image = color_adjustment(self.input_image, self.object_color, self.target_color)

        # Hiển thị ảnh sau khi chỉnh sửa
        adjusted_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        adjusted_image = Image.fromarray(adjusted_image)
        adjusted_image = ImageTk.PhotoImage(adjusted_image)

        # Hiển thị ảnh đã chỉnh sửa, xóa ảnh gốc
        for widget in self.root.winfo_children():
            widget.destroy()

        label = tk.Label(self.root, image=adjusted_image)
        label.image = adjusted_image
        label.pack()

if __name__ == "__main__":
    root = tk.Tk()

    # Mở hộp thoại để chọn ảnh
    file_path = filedialog.askopenfilename()
    app = ColorAdjustmentApp(root, file_path)

    root.mainloop()
