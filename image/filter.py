import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Scale, messagebox
from tkinter import Tk, Label, PhotoImage
from PIL import Image, ImageTk

# Khởi tạo biến ảnh và các thanh trượt
image = None
img_org = None
label_adj = None
radius_scale = None
focus_scale = None
save_button = None 
value = None
radius = None

def enable_save_button():
    global save_button
    if save_button:
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
        if file_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")):
            img_org = Image.open(file_path)

            original_label_width = label_org.winfo_width()
            original_label_height = label_org.winfo_height()

            img_org = resize_image_to_label(img_org, original_label_width, original_label_height)
            update_label_size(label_org, original_label_width, original_label_height)
            update_label_size(label_adj, original_label_width, original_label_height)

            display_image(img_org, label_org)
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

def vignette_effect():
    global img_org, label_adj, save_button, radius_scale, focus_scale, value, radius

    if img_org is None:
        messagebox.showwarning("Lỗi", "Vui lòng mở một ảnh trước!")
        return

    destroy_previous_widgets()

    radius_scale = Scale(
        image_frame_right,
        from_=0,
        to=500,
        orient="horizontal",
        label="Radius",
        resolution=1,
        showvalue=1,
        sliderlength=20,
        length=300,
        command=update_vignette,
    )
    radius_scale.set(0 if radius is None else radius)
    radius_scale.pack()

    focus_scale = Scale(
        image_frame_right,
        from_=0,
        to=10,
        orient="horizontal",
        label="Focus",
        resolution=1,
        showvalue=1,
        sliderlength=20,
        length=300,
        command=update_vignette,
    )
    focus_scale.set(0 if value is None else value)
    focus_scale.pack()

    save_button = tk.Button(
        image_frame_right, text="Lưu", command=save_image, state="disabled"
    )
    save_button.pack()

    update_vignette()

def update_vignette(*args):
    global img_org, label_adj, radius_scale, focus_scale, value, radius

    if img_org is None:
        return

    radius = radius_scale.get()
    value = focus_scale.get()

    if radius == 0 and value == 0:
        # Nếu cả hai đều là 0, hiển thị ảnh gốc
        display_image(img_org, label_adj)
        enable_save_button()
        return

    img_array = np.array(img_org)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Generating vignette mask using Gaussian kernels
    kernel_x = cv2.getGaussianKernel(int(img_array.shape[1] * (0.1 * value + 1)), radius)
    kernel_y = cv2.getGaussianKernel(int(img_array.shape[0] * (0.1 * value + 1)), radius)
    kernel = kernel_y * kernel_x.T

    # Normalizing the kernel
    kernel = kernel / np.linalg.norm(kernel)

    # Generating a mask to image
    mask = 255 * kernel
    mask_imposed = mask[int(0.1 * value * img_array.shape[0]):, int(0.1 * value * img_array.shape[1]):]

    # Applying the mask to each channel in the input image
    output = np.copy(img_array)
    for i in range(3):
        output[:, :, i] = output[:, :, i] * mask_imposed

    # Display the adjusted image
    adjusted_img = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    display_image(adjusted_img, label_adj)
    enable_save_button()

def destroy_previous_widgets():
    global radius_scale, focus_scale
    if radius_scale is not None:
        radius_scale.destroy()
    if focus_scale is not None:
        focus_scale.destroy()

root = tk.Tk()
root.title("Phần mềm Xử lý ảnh")
window_width = 1000
window_height = 500
center_window(root, window_width, window_height)

frame_left = tk.Frame(root, bd=2, relief=tk.SOLID)
frame_right = tk.Frame(root, bd=2, relief=tk.SOLID)
frame_left.pack(side="left", padx=20, pady=20)
frame_right.pack(side="left", padx=20, pady=20)

image_frame_left = tk.Frame(frame_left, bd=2, relief=tk.SOLID)
image_frame_left.pack()

image_frame_right = tk.Frame(frame_right, bd=2, relief=tk.SOLID)
image_frame_right.pack()

label_org = tk.Label(image_frame_left, width=65, height=250)
label_adj = tk.Label(image_frame_right, width=65, height=250)
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

edit_menu.add_command(label="Hiệu ứng mờ viền", command=vignette_effect)
menu_bar.add_cascade(label="Chỉnh sửa", menu=edit_menu)

root.mainloop()
