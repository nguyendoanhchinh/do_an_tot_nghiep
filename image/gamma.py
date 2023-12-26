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
image = img_org = brightness_scale = contrast_scale  = label_org = label_adj = save_button = original_label_width = original_label_height = radius_scale = focus_scale = None
angle_slider = None
rotated_image_canvas = None
rotation_value = 0
margin_color = (255, 255, 255) 
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

def update_brightness(*args):
    global img_org, brightness_scale, label_adj
    if img_org is None:
        return
    brightness_val = brightness_scale.get() / 100
    adjusted_img = img_org.copy()
    adjusted_img = adjusted_img.point(lambda p: p * brightness_val)
    display_image(adjusted_img, label_adj)
    enable_save_button()

def rotate_image():
    global img_org, angle_slider, label_adj, rotated_image_canvas, angle_var, rotated_img, save_button
    destroy_previous_widgets()
    if img_org is None:
        messagebox.showwarning("Lỗi", "Vui lòng mở một ảnh trước!")
        return
    angle_var = StringVar()
    angle_var.trace('w', lambda *args: update_rotation_from_entry(angle_var.get()))
    angle_slider = Scale(image_frame_right, from_=0, to=360, orient="horizontal", label="Góc xoay", showvalue=0, sliderlength=20, length=300, command=update_rotation_from_slider)
    angle_slider.pack()
    angle_entry = Entry(image_frame_right, textvariable=angle_var)
    angle_entry.pack()
    save_button = tk.Button(image_frame_right, text="Lưu", command=lambda: save_image(rotated_img), state="disabled")
    save_button.pack()
    update_rotation_from_slider()
def update_rotation_from_slider(*args):
    global img_org, angle_slider, label_adj, angle_var, rotated_img, save_button
    if img_org is None:
        return
    angle_val = angle_slider.get()
    angle_var.set(str(angle_val)) 
    rotated_img = rotate_and_display(angle_val)
    enable_save_button()
def rotate_and_display(angle_val):
    global img_org, label_adj, image_frame_right, rotated_img, save_button
    angle_rad = radians(float(angle_val))
    width, height = img_org.size
    new_width = abs(width * cos(angle_rad)) + abs(height * sin(angle_rad))
    new_height = abs(width * sin(angle_rad)) + abs(height * cos(angle_rad))

    img_rgba = img_org.convert('RGBA')

    rotated_img = img_rgba.rotate(float(angle_val), resample=Image.BICUBIC, expand=True)
    background = Image.new('RGBA', rotated_img.size, (255, 255, 255, 255))  
    final_img = Image.composite(rotated_img, background, rotated_img)

    # Resize the rotated image to fit the original frame while maintaining aspect ratio
    final_aspect_ratio = final_img.width / final_img.height
    original_aspect_ratio = width / height

    if final_aspect_ratio > original_aspect_ratio:
        new_width = width
        new_height = int(new_width / final_aspect_ratio)
    else:
        new_height = height
        new_width = int(new_height * final_aspect_ratio)

    final_img = final_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    display_image(final_img, label_adj)

    save_button.config(state="normal") 
    return final_img

def update_rotation_from_entry(angle_val):
    global img_org, angle_slider, label_adj, angle_var
    if img_org is None:
        return
    try:
        angle_val = max(0, min(360, float(angle_val)))
    except ValueError:
       
     angle_val = 0
    angle_var.set(str(angle_val))  
    angle_slider.set(angle_val)  
    rotate_and_display(angle_val)

def destroy_previous_widgets():
    global brightness_scale, contrast_scale, gamma_scale, save_button,radius_scale, focus_scale
    if brightness_scale is not None:
        brightness_scale.destroy()
    if save_button is not None and save_button.winfo_exists():
        save_button.destroy()
        
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
image_frame_right = tk.Frame(frame_right, bd=2, relief=tk.SOLID)
image_frame_right.pack()
label_org = tk.Label(image_frame_left, width=65, height=250)
label_adj = tk.Label(image_frame_right, width=75, height=350)
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
edit_menu.add_command(label="Điều chỉnh độ sáng ", command=adjust_brightness)
edit_menu.add_command(label="Xoay ảnh", command=rotate_image)
menu_bar.add_cascade(label="Chỉnh sửa", menu=edit_menu)
root.mainloop()
