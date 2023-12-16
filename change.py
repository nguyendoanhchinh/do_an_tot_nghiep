# import các thư viện
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
from tkinter import Scrollbar
from tkinter import colorchooser

# Khởi tạo biến ảnh
image = img_org = brightness_scale = contrast_scale = gamma_scale = label_org = label_adj = save_button = original_label_width = original_label_height = radius_scale = focus_scale = None
value = 1
radius = 360
angle_slider = None
rotated_image_canvas = None
rotation_value = 0
margin_color = (255, 255, 255) 
def enable_save_button():
    global save_button
    if save_button:
        save_button.config(state="normal")
# hiển thị giao diện chính giữa màn hình
def center_window(window, width, height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    window.geometry(f"{width}x{height}+{x}+{y}")


# Hàm mở tệp
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

            # Tính toán kích thước mới của ảnh để vừa vào nhãn
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


# điều chỉnh kích thước ảnh  trong khung
def resize_image_to_label(img, label_width, label_height):
    aspect_ratio = img.width / img.height  # tính tỷ lệ kích thước khung
    new_width = min(img.width, label_width)  # chiều rộng của khung
    new_height = int(new_width / aspect_ratio)  # chiều cao của khung
    # Nếu chiều cao mới lớn hơn chiều cao của nhãn, thì thu nhỏ ảnh
    if new_height > label_height:
        new_height = label_height
        new_width = int(new_height * aspect_ratio)
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

def update_label_size(label, width, height):
    label.config(width=width, height=height)


# Hàm hiển thị ảnh trong label
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


# Hàm điều chỉnh độ sáng
def adjust_brightness():
    global img_org, brightness_scale, label_adj, save_button
    # Kiểm tra nếu có widgets của chức năng trước đó, thì hủy bỏ chúng
    destroy_previous_widgets()
    if img_org is None:
        messagebox.showwarning("Lỗi", "Vui lòng mở một ảnh trước!")
        return
    # Tạo thanh điều chỉnh độ sáng
    brightness_scale = Scale(image_frame_right,from_=0,to=255,orient="horizontal",label="Độ sáng",showvalue=1,sliderlength=20,length=300,command=update_brightness,)
    brightness_scale.pack()
    # Tạo nút "Lưu"
    save_button = tk.Button(
        image_frame_right, text="Lưu", command=save_image, state="disabled")
    save_button.pack()
    # Gọi hàm cập nhật ảnh đã điều chỉnh
    update_brightness()


# Hàm cập nhật ảnh đã điều chỉnh cho độ sáng
def update_brightness(*args):
    global img_org, brightness_scale, label_adj
    if img_org is None:
        return
    brightness_val = brightness_scale.get() / 100
    adjusted_img = img_org.copy()
    # Điều chỉnh độ sáng
    adjusted_img = adjusted_img.point(lambda p: p * brightness_val)
    # Hiển thị ảnh đã điều chỉnh
    display_image(adjusted_img, label_adj)
    enable_save_button()


# Hàm điều chỉnh độ tương phản
def adjust_contrast():
    global img_org, contrast_scale, label_adj, save_button

    # Kiểm tra nếu có widgets của chức năng trước đó, thì hủy bỏ chúng
    destroy_previous_widgets()
    if img_org is None:
        messagebox.showwarning("Lỗi", "Vui lòng mở một ảnh trước!")
        return
    # Tạo thanh điều chỉnh độ tương phản
    contrast_scale = Scale(image_frame_right,from_=0.1,to=3.0,orient="horizontal",label="Độ tương phản",resolution=0.1,showvalue=1,sliderlength=20,length=300,command=update_contrast,)
    contrast_scale.pack()
    # Tạo nút "Lưu"
    save_button = tk.Button(
        image_frame_right, text="Lưu", command=save_image, state="disabled"
    )
    save_button.pack()
    # Gọi hàm cập nhật ảnh đã điều chỉnh
    update_contrast()
# Hàm cập nhật ảnh đã điều chỉnh cho độ tương phản
def update_contrast(*args):
    global img_org, contrast_scale, label_adj

    if img_org is None:
        return
    contrast_val = contrast_scale.get()
    adjusted_img = img_org.copy()

    # Điều chỉnh độ tương phản
    for x in range(adjusted_img.width):
        for y in range(adjusted_img.height):
            r, g, b = adjusted_img.getpixel((x, y))
            new_r = int((r - 128) * contrast_val + 128)
            new_g = int((g - 128) * contrast_val + 128)
            new_b = int((b - 128) * contrast_val + 128)
            adjusted_img.putpixel((x, y), (new_r, new_g, new_b))
    # Hiển thị ảnh đã điều chỉnh
    display_image(adjusted_img, label_adj)
    enable_save_button()


# Hàm điều chỉnh gamma
def adjust_gamma():
    global img_org, gamma_scale, label_adj, save_button
    # Kiểm tra nếu có widgets của chức năng trước đó, thì hủy bỏ chúng
    destroy_previous_widgets()
    if img_org is None:
        messagebox.showwarning("Lỗi", "Vui lòng mở một ảnh trước!")
        return
    # Tạo thanh điều chỉnh gamma
    gamma_scale = Scale(image_frame_right,from_=0.1,to=5,orient="horizontal",label="Gamma",resolution=0.1,showvalue=1,sliderlength=20,length=300,command=update_gamma,)
    gamma_scale.pack()
    # Tạo nút "Lưu"
    save_button = tk.Button(
        image_frame_right, text="Lưu", command=save_image, state="disabled")
    save_button.pack()

    # Gọi hàm cập nhật ảnh theo giá trị gamma
    update_gamma()

# Hàm cập nhật ảnh theo giá trị gamma
def update_gamma(*args):
    global img_org, gamma_scale, label_adj
    if img_org is None:
        return
    gamma = gamma_scale.get()
    table = np.array(
        [((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    adjusted_img = cv2.LUT(image, table)
    adjusted_img = Image.fromarray(adjusted_img)
    # Tính toán kích thước mới của ảnh để vừa vào nhãn
    adjusted_img = resize_image_to_label(
        adjusted_img, label_adj.winfo_width(), label_adj.winfo_height())
    # Hiển thị ảnh đã điều chỉnh
    display_image(adjusted_img, label_adj)
    enable_save_button()


# Hàm điều làm sắc nét ảnh
def adjust_sharpe():
    return true


# Hàm cân bằng hình ảnh

def equalize_image():
    global img_org, label_adj, save_button

    # Kiểm tra nếu có widgets của chức năng trước đó, thì hủy bỏ chúng
    destroy_previous_widgets()

    if img_org is None:
        messagebox.showwarning("Lỗi", "Vui lòng mở một ảnh trước!")
        return

    # Chuyển đổi ảnh từ PIL Image sang NumPy array
    img_array = np.array(img_org)

    # Chuyển đổi định dạng màu từ RGB sang HSV
    img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    # Tách các kênh màu
    h, s, v = cv2.split(img_hsv)

    # Áp dụng equalizeHist cho kênh độ sáng (Value)
    v_equalized = cv2.equalizeHist(v)

    # Gộp các kênh màu lại
    img_hsv_equalized = cv2.merge([h, s, v_equalized])

    # Chuyển đổi định dạng màu từ HSV sang RGB
    img_equalized = cv2.cvtColor(img_hsv_equalized, cv2.COLOR_HSV2RGB)

    # Hiển thị ảnh đã điều chỉnh
    adjusted_img = Image.fromarray(img_equalized)
    display_image(adjusted_img, label_adj)
    enable_save_button()
      # Cập nhật hàm lưu ảnh
    save_button = tk.Button(
        image_frame_right, text="Lưu", command=save_image, state="normal"
    )
    save_button.pack()

# Hiệu ứng họa tiết
def apply_vignette(radius, focus):
    global img_org, label_adj, save_button

    # Kiểm tra nếu có widgets của chức năng trước đó, thì hủy bỏ chúng
    destroy_previous_widgets()

    if img_org is None:
        messagebox.showwarning("Lỗi", "Vui lòng mở một ảnh trước!")
        return

    # Tính toán kích thước mới của ảnh để vừa vào nhãn
    img_resized = resize_image_to_label(
        img_org, label_adj.winfo_width(), label_adj.winfo_height())
    # Chuyển đổi ảnh thành mảng NumPy
    img_array = np.array(img_resized)
    # Tính toán mask vignette
    rows, cols = img_array.shape[:2]
    mask = np.zeros((rows, cols), dtype=np.uint8)
    center = (cols // 2, rows // 2)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center[1]) ** 2 + (j - center[0]) ** 2)
            mask[i, j] = 255 * (1 - (distance / (radius * focus)))
    # Áp dụng mask lên ảnh
    result_image = cv2.merge([img_array[:,:,0] * (mask / 255.0), img_array[:,:,1] * (mask / 255.0), img_array[:,:,2] * (mask / 255.0)])
    # Chuyển đổi kết quả thành ảnh Image
    result_image = Image.fromarray(result_image.astype('uint8'))
    # Hiển thị ảnh đã điều chỉnh
    display_image(result_image, label_adj)
    enable_save_button()

#Hiệu ứng làm mờ cạnh
def vignette_effect():
    global img_org, label_adj, save_button, radius_scale, focus_scale, value, radius

    if img_org is None:
        messagebox.showwarning("Lỗi", "Vui lòng mở một ảnh trước!")
        return
    destroy_previous_widgets()
    radius_scale = Scale(image_frame_right,from_=0,to=500,orient="horizontal",label="Radius",resolution=1,showvalue=1,sliderlength=20,length=300,command=update_vignette,)
    radius_scale.set(0 if radius is None else radius)
    radius_scale.pack()
    focus_scale = Scale(image_frame_right, from_=0,to=10,orient="horizontal",label="Focus",resolution=1,showvalue=1,sliderlength=20,length=300,command=update_vignette,)
    focus_scale.set(0 if value is None else value)
    focus_scale.pack()
    save_button = tk.Button(
        image_frame_right, text="Lưu", command=save_image, state="disabled")
    save_button.pack()
    update_vignette()

def update_vignette(*args):
    global img_org, label_adj, radius_scale, focus_scale, value, radius
    if img_org is None:
        return
    radius = radius_scale.get()
    value = focus_scale.get()
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
#Xoay ảnh

def rotate_image():
    global img_org, rotated_image_canvas, angle_slider, rotation_value, margin_color, color_button

    destroy_previous_widgets()

    if img_org is None:
        tk.messagebox.showwarning("Lỗi", "Vui lòng mở một ảnh trước!")
        return

    color_button = tk.Button(root, text="Chọn màu nền", command=change_background_color)
    color_button.pack()

    angle_slider = Scale(root, from_=0, to=360, orient="horizontal", length=200, command=update_rotation)
    angle_slider.pack()

    update_rotation()

def change_background_color():
    global margin_color, color_button
    color = colorchooser.askcolor(title="Chọn màu nền")[0]
    margin_color = tuple(map(int, color))
    color_button.configure(bg=rgb_to_hex(margin_color))

def rgb_to_hex(rgb):
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

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
# Hủy bỏ các widgets của chức năng trước đó
def destroy_previous_widgets():
    global brightness_scale, contrast_scale, gamma_scale, save_button,radius_scale, focus_scale

    if brightness_scale is not None:
        brightness_scale.destroy()
    if contrast_scale is not None:
        contrast_scale.destroy()
    if gamma_scale is not None:
        gamma_scale.destroy()
    if radius_scale is not None:
        radius_scale.destroy()
    if focus_scale is not None:
        focus_scale.destroy()
    # Check if save_button exists before destroying
    if save_button is not None and save_button.winfo_exists():
        save_button.destroy()

# Tạo cửa sổ gốc
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
# Tạo khung chứa các thành phần
frame_left = tk.Frame(root, bd=2, relief=tk.SOLID)
frame_right = tk.Frame(root, bd=2, relief=tk.SOLID)
frame_left.pack(side="left", padx=20, pady=20)
frame_right.pack(side="left", padx=20, pady=20)

# Tạo khung con trong frame_left để hiển thị ảnh gốc
image_frame_left = tk.Frame(frame_left, bd=2, relief=tk.SOLID)
image_frame_left.pack()

# Tạo khung con trong frame_right để hiển thị ảnh đã điều chỉnh và các điều khiển
image_frame_right = tk.Frame(frame_right, bd=2, relief=tk.SOLID)
image_frame_right.pack()

# Tạo các nhãn cho khung trái và khung phải
label_org = tk.Label(image_frame_left, width=65, height=250)
label_adj = tk.Label(image_frame_right, width=65, height=250)
label_org.pack()
label_adj.pack()

# Tạo menu
menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

file_menu = tk.Menu(menu_bar)
file_menu.add_command(label="Mở", command=open_file)
file_menu.add_separator()
file_menu.add_command(label="Thoát", command=root.destroy)
menu_bar.add_cascade(label="Tệp", menu=file_menu)

edit_menu = tk.Menu(menu_bar)
edit_menu.add_command(label="Điều chỉnh độ sáng ", command=adjust_brightness)
edit_menu.add_command(label="Điều chỉnh độ tương phản", command=adjust_contrast)
edit_menu.add_command(label="Hiệu chỉnh gamma", command=adjust_gamma)
edit_menu.add_command(label="Hiệu ứng mờ viền", command=vignette_effect)
edit_menu.add_command(label="Cân bằng ánh sáng ảnh", command=equalize_image)
edit_menu.add_command(label="Xoay ảnh", command=rotate_image)
menu_bar.add_cascade(label="Chỉnh sửa", menu=edit_menu)

# Main loop
root.mainloop()
