import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, Scale, colorchooser
from tkinter import Tk, Label
from PIL import Image, ImageTk
import numpy as np

# Khởi tạo biến ảnh
img_org = rotated_image_canvas = None
rotation_value = 0
margin_color = (255, 255, 255)
angle_slider = None

def rotate_image():
    global img_org, rotated_image_canvas, angle_slider, rotation_value, margin_color

    destroy_previous_widgets()

    if img_org is None:
        tk.messagebox.showwarning("Lỗi", "Vui lòng mở một ảnh trước!")
        return

    color = colorchooser.askcolor(title="Chọn màu nền")[0]
    margin_color = tuple(map(int, color))

    angle_slider = Scale(root, from_=0, to=360, orient="horizontal", length=200, command=update_rotation)
    angle_slider.pack()

    update_rotation()

def update_rotation(*args):
    global img_org, rotated_image_canvas, angle_slider, rotation_value, margin_color

    if img_org is not None:
        rotation_value = angle_slider.get()
        rotated_img = rotate_image_by_angle(img_org, rotation_value, margin_color)
        display_image(rotated_img, rotated_image_canvas)

def rotate_image_by_angle(image, angle, margin_color):
    if image is not None:
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        image_center = (width / 2, height / 2)
        rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1)
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)
        rotation_matrix[0, 2] += bound_w / 2 - image_center[0]
        rotation_matrix[1, 2] += bound_h / 2 - image_center[1]
        rotated_image = cv2.warpAffine(img_array, rotation_matrix, (bound_w, bound_h), borderMode=cv2.BORDER_CONSTANT, borderValue=margin_color)
        rotated_image_pil = Image.fromarray(rotated_image)
        return rotated_image_pil

def destroy_previous_widgets():
    global angle_slider
    if angle_slider is not None:
        angle_slider.destroy()

def display_image(img, label):
    photo = ImageTk.PhotoImage(img)
    label.configure(image=photo)
    label.image = photo

def open_file():
    global img_org, rotated_image_canvas

    file_path = filedialog.askopenfilename()

    if file_path:
        if file_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")):
            img_org = Image.open(file_path)

            update_label_size(label_org, img_org.width, img_org.height)

            display_image(img_org, label_org)
            rotated_image_canvas = tk.Label(image_frame_left, width=65, height=250)
            rotated_image_canvas.pack()

def update_label_size(label, width, height):
    label.config(width=width, height=height)

# Tạo cửa sổ gốc
root = tk.Tk()
root.title("Phần mềm Xử lý ảnh")
window_width = 1000
window_height = 500
root.geometry(f"{window_width}x{window_height}+300+100")

# Tạo khung chứa các thành phần
frame_left = tk.Frame(root, bd=2, relief=tk.SOLID)
frame_left.pack(side="left", padx=20, pady=20)

# Tạo khung con trong frame_left để hiển thị ảnh gốc
image_frame_left = tk.Frame(frame_left, bd=2, relief=tk.SOLID)
image_frame_left.pack()

# Tạo nhãn cho khung trái
label_org = tk.Label(image_frame_left, width=65, height=250)
label_org.pack()

# Tạo menu
menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

file_menu = tk.Menu(menu_bar)
file_menu.add_command(label="Mở", command=open_file)
file_menu.add_separator()
file_menu.add_command(label="Thoát", command=root.destroy)
menu_bar.add_cascade(label="Tệp", menu=file_menu)

edit_menu = tk.Menu(menu_bar)
edit_menu.add_command(label="Xoay ảnh", command=rotate_image)
menu_bar.add_cascade(label="Chỉnh sửa", menu=edit_menu)

# Main loop
root.mainloop()
