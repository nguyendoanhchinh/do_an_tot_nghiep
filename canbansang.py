import cv2
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Button, PhotoImage
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import messagebox
import numpy as np
from tkinter import Scale

# Khởi tạo biến image
image = None
panelA = None
panelB = None
img_org = None

# Tạo cửa sổ gốc
root = tk.Tk()
root.title("Phần mềm Xử lý ảnh")
# Đặt kích thước cửa sổ
root.geometry("800x400")

# Hàm chức năng Mở
def open_file():
    global image, panelA, panelB, img_org  
    file_path = filedialog.askopenfilename()
  
    if file_path:
        # Kiểm tra đuôi file
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif','.webp')):
            # Tải ảnh gốc
            img_org = Image.open(file_path)
            img_org = img_org.resize((300, 350), Image.LANCZOS)
            
            # Hiển thị ảnh gốc
            photo = ImageTk.PhotoImage(img_org)
            panelA = tk.Label(root, image=photo)
            panelA.image = photo
            panelA.pack(side="left", padx=10, pady=10)
            
           # Lưu ảnh gốc vào biến 'image'
            image = np.array(img_org)
            
            # Tạo khung hiển thị kết quả
            panelB = tk.Label(root)
            panelB.pack(side="right", padx=10, pady=10)
            
        else:
            print("Định dạng ảnh không hỗ trợ.")

# Các chức năng xử lý ảnh ở đây (điều chỉnh độ sáng, độ tương phản, cân bằng màu, gamma, sắc nét, ...)

# Hàm chức năng Thoát
def exit_program():
    root.destroy()

# Tạo menu
menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

# Tạo menu Tệp tin
file_menu = tk.Menu(menu_bar)
menu_bar.add_cascade(label="Tệp tin", menu=file_menu)
file_menu.add_command(label="Mở", command=open_file)
file_menu.add_separator()
file_menu.add_command(label="Thoát", command=exit_program)

# Tạo menu Xử lý ảnh
image_menu = tk.Menu(menu_bar)
menu_bar.add_cascade(label="Xử lý ảnh", menu=image_menu)
# Thêm các chức năng xử lý ảnh vào đây

# Tạo menu Cửa sổ
window_menu = tk.Menu(menu_bar)
menu_bar.add_cascade(label="Cửa sổ", menu=window_menu)
window_menu.add_command(label="Sắp xếp", command=None)  # Thêm chức năng sắp xếp ở đây

# Tạo menu Trợ giúp
help_menu = tk.Menu(menu_bar)
menu_bar.add_cascade(label="Trợ giúp", menu=help_menu)

root.mainloop()
