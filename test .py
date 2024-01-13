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
def enable_save_button():
    global save_button
    if save_button is not None and save_button.winfo_exists():
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
def enable_save_button():
    global save_button
    if save_button is not None and save_button.winfo_exists():
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

def sharpen_image():
    return true

def on_mousewheel(event):
    canvas_right.yview_scroll(int(-1 * (event.delta / 120)), "units")
    canvas_left.yview_scroll(int(-1 * (event.delta / 120)), "units")
    canvas_right.bind_all("<MouseWheel>", on_mousewheel)
    canvas_left.bind_all("<MouseWheel>", on_mousewheel)
def sharpen_image():
    return true
root = tk.Tk()
root.title("Phần mềm Xử lý ánh sáng ảnh")
root.state('zoomed')
header_label = tk.Label(root, text="Phần mềm Xử lý ảnh", font=("Helvetica", 14, "bold"), fg="red")
header_label.pack(side="top", pady=10)
root.configure(bg="pink") 
frame_left = tk.Frame(root, bd=2, relief=tk.SOLID, width=650, height=600)
frame_right = tk.Frame(root, bd=2, relief=tk.SOLID, width=650, height=600)
frame_left.pack(side="left", padx=20, pady=20)
frame_right.pack(side="left", padx=20, pady=20)
frame_left.pack_propagate(False)
frame_right.pack_propagate(False)
canvas_right = tk.Canvas(frame_right)
canvas_right.pack(side=tk.LEFT, expand=True, fill='both')
scrollbar_right = tk.Scrollbar(frame_right, orient="vertical", command=canvas_right.yview)
scrollbar_right.pack(side=tk.RIGHT, fill=tk.Y)
canvas_right.configure(yscrollcommand=scrollbar_right.set)
canvas_right.bind('<Configure>', lambda e: canvas_right.configure(scrollregion=canvas_right.bbox("all")))
image_frame_right = tk.Frame(canvas_right)
canvas_right.create_window((0, 0), window=image_frame_right, anchor="nw")
label_org = tk.Label(frame_left, width=700, height=700)
label_adj = tk.Label(image_frame_right, width=700, height=700)
# Tạo nhãn cho khung mở ảnh
label_frame_left = tk.Label(frame_left, text="Khung mở ảnh", font=("Helvetica", 14, "bold"), anchor="center", justify="center")
label_frame_left.pack(side="top")

# Tạo nhãn cho khung xử lý
label_frame_right = tk.Label(frame_right, text="Khung xử lý", font=("Helvetica", 14, "bold"))
label_frame_right.place(relx=0.5, rely=0.024, anchor='center')


label_org.pack()
label_adj.pack()
menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

file_menu = tk.Menu(menu_bar)
file_menu.add_command(label="Mở", command=open_file)
file_menu.add_separator()
file_menu.add_command(label="Thoát", command=root.destroy)
menu_bar.add_cascade(label="Tệp", menu=file_menu)
edit_menu = tk.Menu(menu_bar)
edit_menu.add_command(label="Làm sắc nét hình ảnh", command=sharpen_image)
menu_bar.add_cascade(label="Chỉnh sửa", menu=edit_menu)
menu_help = tk.Menu(menu_bar)
menu_bar.add_cascade(label="Trợ giúp", menu=menu_help)
exit_menu = tk.Menu(menu_bar)
menu_bar.add_cascade(label="Thoát", menu=exit_menu)
root.mainloop()
