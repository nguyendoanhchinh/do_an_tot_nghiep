import cv2
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter import Entry, StringVar, Scale,Tk, Label, Button, PhotoImage
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import messagebox
import numpy as np
from tkinter import Scale
from tkinter import Scrollbar
from tkinter import colorchooser
from math import sin, cos, radians
image = img_org = brightness_scale = contrast_scale = gamma_scale = label_org = label_adj = save_button = original_label_width = original_label_height = radius_scale = focus_scale = None

def enable_save_button():
    global save_button
    if save_button and save_button.winfo_exists():
        save_button.config(state="normal")
def center_window(window, width, height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    window.geometry(f"{width}x{height}+{x}+{y}")
def open_file():
    global image, label_org, img_org, original_label_width, original_label_height
    file_path = filedialog.askopenfilename()
    if file_path:
        if file_path.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")
        ):
            img_org = Image.open(file_path)
            original_label_width = label_org.winfo_width()
            original_label_height = label_org.winfo_height()
            img_org = resize_image_to_label(
                img_org, original_label_width, original_label_height
            )

            update_label_size(label_org, original_label_width, original_label_height)
            update_label_size(label_adj, original_label_width, original_label_height)

            display_image(img_org, label_org)
            image = np.array(img_org)
            enable_save_button()
        else:
            messagebox.showwarning("Lỗi", "Định dạng ảnh không được hỗ trợ.")
def resize_image_to_label(img, label_width, label_height):
    aspect_ratio = img.width / img.height  
    new_width = min(img.width, label_width) 
    new_height = int(new_width / aspect_ratio)  
    if new_height > label_height:
        new_height = label_height
        new_width = int(new_height * aspect_ratio)
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

def update_label_size(label, width, height):
    label.config(width=width, height=height)
def display_image(img, label):
    photo = ImageTk.PhotoImage(img)
    label.configure(image=photo)
    label.image = photo
# chức năng lưu ảnh
def enable_save_button():
    global save_button
    if save_button is not None:
        save_button.config(state="normal")
def save_image():
    global label_adj
    if label_adj.image:
        image_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*"),
            ],
        )
        if image_path:
            adjusted_img = ImageTk.getimage(label_adj.image)
            adjusted_img.save(image_path)
def adjust_brightness():
    global img_org, brightness_scale, label_adj, save_button
    destroy_previous_widgets()
    if img_org is None:
        messagebox.showwarning("Lỗi", "Vui lòng mở một ảnh trước!")
        return
    brightness_scale = Scale(image_frame_right,from_=0,to=255,orient="horizontal",label="Độ sáng",showvalue=1,sliderlength=20,length=300,command=update_brightness,)
    brightness_scale.pack()
    save_button = tk.Button(
        image_frame_right, text="Lưu", command=save_image, state="disabled")
    save_button.pack()
    update_brightness()
root = tk.Tk()
root.title("Phần mềm Xử lý ảnh")
window_width = 1000
window_height = 500
center_window(root, window_width, window_height)
header_label = tk.Label(root, text="Phần mềm Xử lý ảnh", font=("Helvetica", 14, "bold"), fg="red")
header_label.pack(side="top", pady=10)
def center_text(widget):
    widget.update_idletasks()
    x = (widget.winfo_width() - root.winfo_reqwidth()) // 2
    root.geometry(f"+{x}+0")
center_text(header_label)
root.configure(bg="pink") 
frame_left = tk.Frame(root, bd=2, relief=tk.SOLID)
frame_right = tk.Frame(root, bd=2, relief=tk.SOLID)
frame_left.pack(side="left", padx=20, pady=20)
frame_right.pack(side="left", padx=20, pady=20)
image_frame_left = tk.Frame(frame_left, bd=2, relief=tk.SOLID)
image_frame_left.pack()

# Tạo một Canvas với kích thước cố định để chứa label_adj và thanh trượt
canvas = tk.Canvas(frame_right, width=65, height=250)
canvas.pack(side="left")

# Thêm thanh trượt vào Canvas
scrollbar = Scrollbar(frame_right, command=canvas.yview)
scrollbar.pack(side="left", fill="y")

# Cấu hình Canvas để sử dụng thanh trượt
canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

# Tạo một khung để chứa label_adj
frame_adj = tk.Frame(canvas)

# Thêm khung vào Canvas
canvas.create_window((0,0), window=frame_adj, anchor="nw")

# Tạo label_adj trong khung
label_adj = tk.Label(frame_adj, width=65, height=250)
label_adj.pack()

menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

file_menu = tk.Menu(menu_bar)
file_menu.add_command(label="Mở", command=open_file)
file_menu.add_separator()
file_menu.add_command(label="Thoát", command=root.destroy)
menu_bar.add_cascade(label="Tệp", menu=file_menu)
edit_menu = tk.Menu(menu_bar)
edit_menu.add_command(label="Điều chỉnh độ sáng ", command=adjust_brightness)
menu_bar.add_cascade(label="Chỉnh sửa", menu=edit_menu)

root.mainloop()
