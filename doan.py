# import các thư viện
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
# Khởi tạo biến ảnh
image = img_org = brightness_scale = contrast_scale = gamma_scale = label_org = label_adj = save_button = original_label_width = original_label_height = radius_scale = focus_scale = radius_scale = focus_x_scale = focus_y_scale =None
value = 1
radius = 360
angle_slider = None
rotated_image_canvas = None
rotation_value = 0
margin_color = (255, 255, 255)
# Hàm này sẽ kích hoạt nút lưu
def enable_save_button():
    global save_button
    if save_button is not None and save_button.winfo_exists():
        save_button.config(state="normal")
# Hàm này sẽ đặt cửa sổ ở giữa màn hình
def center_window(window, width, height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    window.geometry(f"{width}x{height}+{x}+{y}")
# Hàm này sẽ mở tệp ảnh
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
# Hàm này sẽ thay đổi kích thước ảnh để phù hợp với khung nhãn
def resize_image_to_label(img, label_width, label_height):
    aspect_ratio = img.width / img.height
    new_width = min(img.width, label_width)
    new_height = int(new_width / aspect_ratio)
    if new_height > label_height:
        new_height = label_height
        new_width = int(new_height * aspect_ratio)
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

# Hàm này sẽ cập nhật kích thước của khung nhãn
def update_label_size(label, width, height):
    label.config(width=width, height=height)

# Hàm này sẽ hiển thị ảnh trên khung nhãn
def display_image(img, label):
    photo = ImageTk.PhotoImage(img)
    label.configure(image=photo)
    label.image = photo
# chức năng lưu ảnh
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
    brightness_scale = Scale(image_frame_right,from_=-255,to=255,orient="horizontal",label="Độ sáng",showvalue=1,sliderlength=20,length=300,command=update_brightness,)
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
    # Lấy giá trị từ thanh trượt và giới hạn nó trong khoảng [-255, 255]
    brightness_val = max(-255, min(round(brightness_scale.get()), 255))
    adjusted_img = img_org.copy()
    adjusted_img = adjusted_img.point(lambda p: p + brightness_val)
    display_image(adjusted_img, label_adj)
    enable_save_button()

# Hàm điều chỉnh độ tương phản
def truncate(value):
    return max(0, min(255, value))

def adjust_value(value, contrast_val):
    if contrast_val > 0:
        return truncate(int((value - 128) * (contrast_val + 1) + 128))
    else:
        return truncate(int(value / (abs(contrast_val) + 1)))

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
            new_r = adjust_value(r, contrast_val)
            new_g = adjust_value(g, contrast_val)
            new_b = adjust_value(b, contrast_val)
            adjusted_img.putpixel((x, y), (new_r, new_g, new_b))
    # Hiển thị ảnh đã điều chỉnh
    display_image(adjusted_img, label_adj)
    enable_save_button()

def adjust_contrast():
    global img_org, contrast_scale, label_adj, save_button

    # Kiểm tra nếu có widgets của chức năng trước đó, thì hủy bỏ chúng
    destroy_previous_widgets()
    if img_org is None:
        messagebox.showwarning("Lỗi", "Vui lòng mở một ảnh trước!")
        return
    # Tạo thanh điều chỉnh độ tương phản
    contrast_scale = Scale(image_frame_right,from_=-3,to=3.0,orient="horizontal",label="Độ tương phản",resolution=0.1,showvalue=1,sliderlength=20,length=300,command=update_contrast,)
    contrast_scale.pack()
    # Tạo nút "Lưu"
    save_button = tk.Button(
        image_frame_right, text="Lưu", command=save_image, state="disabled"
    )
    save_button.pack()
    # Gọi hàm cập nhật ảnh đã điều chỉnh
    update_contrast()
def adjust_brightness_contrast():
    global img_org, brightness_scale, contrast_scale, label_adj, save_button

    # Kiểm tra nếu có widgets của chức năng trước đó, thì hủy bỏ chúng
    destroy_previous_widgets()
    if img_org is None:
        messagebox.showwarning("Lỗi", "Vui lòng mở một ảnh trước!")
        return
    # Tạo thanh điều chỉnh độ sáng
    brightness_scale = Scale(image_frame_right,from_=-255,to=255,orient="horizontal",label="Độ sáng",showvalue=1,sliderlength=20,length=300,command=update_brightness_contrast,)
    brightness_scale.pack()
    # Tạo thanh điều chỉnh độ tương phản
    contrast_scale = Scale(image_frame_right,from_=-3,to=3.0,orient="horizontal",label="Độ tương phản",resolution=0.1,showvalue=1,sliderlength=20,length=300,command=update_brightness_contrast,)
    contrast_scale.pack()
    # Tạo nút "Lưu"
    save_button = tk.Button(
        image_frame_right, text="Lưu", command=save_image, state="disabled"
    )
    save_button.pack()
    # Gọi hàm cập nhật ảnh đã điều chỉnh
    update_brightness_contrast()

def update_brightness_contrast(*args):
    global img_org, brightness_scale, contrast_scale, label_adj
    if img_org is None:
        return
    # Lấy giá trị từ thanh trượt và giới hạn nó trong khoảng [-255, 255]
    brightness_val = max(-255, min(round(brightness_scale.get()), 255))
    contrast_val = contrast_scale.get()
    adjusted_img = img_org.copy()
    # Điều chỉnh độ sáng
    adjusted_img = adjusted_img.point(lambda p: p + brightness_val)
    # Điều chỉnh độ tương phản
    for x in range(adjusted_img.width):
        for y in range(adjusted_img.height):
            r, g, b = adjusted_img.getpixel((x, y))
            new_r = adjust_value(r, contrast_val)
            new_g = adjust_value(g, contrast_val)
            new_b = adjust_value(b, contrast_val)
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


# Hàm điều chỉnh độ tương phản
def truncate(value):
    return max(0, min(255, value))

def adjust_value(value, contrast_val):
    if contrast_val > 0:
        return truncate(int((value - 128) * (contrast_val + 1) + 128))
    else:
        return truncate(int(value / (abs(contrast_val) + 1)))

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
            new_r = adjust_value(r, contrast_val)
            new_g = adjust_value(g, contrast_val)
            new_b = adjust_value(b, contrast_val)
            adjusted_img.putpixel((x, y), (new_r, new_g, new_b))
    # Hiển thị ảnh đã điều chỉnh
    display_image(adjusted_img, label_adj)
    enable_save_button()

def adjust_contrast():
    global img_org, contrast_scale, label_adj, save_button

    # Kiểm tra nếu có widgets của chức năng trước đó, thì hủy bỏ chúng
    destroy_previous_widgets()
    if img_org is None:
        messagebox.showwarning("Lỗi", "Vui lòng mở một ảnh trước!")
        return
    # Tạo thanh điều chỉnh độ tương phản
    contrast_scale = Scale(image_frame_right,from_=-3,to=3.0,orient="horizontal",label="Độ tương phản",resolution=0.1,showvalue=1,sliderlength=20,length=300,command=update_contrast,)
    contrast_scale.pack()
    # Tạo nút "Lưu"
    save_button = tk.Button(
        image_frame_right, text="Lưu", command=save_image, state="disabled"
    )
    save_button.pack()
    # Gọi hàm cập nhật ảnh đã điều chỉnh
    update_contrast()

# Hàm cân bằng màu
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
    # Bước 1: Thống kê số lượng pixel cho từng mức sáng
    hist, bins = np.histogram(v.flatten(), 256, [0, 256])
    # Bước 2: Tính hàm tích lũy Z
    cdf = hist.cumsum()
    # Bước 3: Tính hàm biến đổi K(i)
    k_transform = (cdf - cdf.min()) / (cdf.max() - cdf.min()) * 255
    # Áp dụng hàm biến đổi K(i) cho kênh độ sáng (Value)
    v_equalized = k_transform[v.astype(np.uint8)]  # Chuyển đổi về kiểu uint8
    # Gộp các kênh màu lại
    img_hsv_equalized = cv2.merge([h, s, v_equalized.astype(np.uint8)])
    # Chuyển đổi định dạng màu từ HSV sang RGB
    img_equalized = cv2.cvtColor(img_hsv_equalized, cv2.COLOR_HSV2RGB)
    # Hiển thị ảnh đã điều chỉnh
    adjusted_img = Image.fromarray(img_equalized)
    display_image(adjusted_img, label_adj)
    enable_save_button()
    # Tạo nút "Lưu" nếu chưa tồn tại
    if not save_button or not save_button.winfo_exists():
        save_button = tk.Button(
            image_frame_right, text="Lưu", command=save_image, state="normal")
        save_button.pack()
    else:
        save_button.config(state="normal")
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
def vignette_effect():
    global img_org, label_adj, save_button, radius_scale, focus_x_scale, focus_y_scale, value, radius
    if img_org is None:
        messagebox.showwarning("Lỗi", "Vui lòng mở một ảnh trước!")
        return
    destroy_previous_widgets()
    # Sửa lại focus_x_scale thành hai thanh scale cho trục x và trục y
    global focus_x_scale
    focus_x_scale = Scale(image_frame_right, from_=0, to=10, orient="horizontal", label="Focus X", resolution=1, showvalue=1, sliderlength=20, length=300, command=update_vignette,)
    focus_x_scale.set(0 if value is None else value)
    focus_x_scale.pack()
    global focus_y_scale
    focus_y_scale = Scale(image_frame_right, from_=0, to=10, orient="horizontal", label="Focus Y", resolution=1, showvalue=1, sliderlength=20, length=300, command=update_vignette,)
    focus_y_scale.set(0 if value is None else value)
    focus_y_scale.pack()
    global radius_scale
    radius_scale = Scale(image_frame_right, from_=0, to=500, orient="horizontal", label="Radius", resolution=1, showvalue=1, sliderlength=20, length=300, command=update_vignette,)
    radius_scale.set(0 if radius is None else radius)
    radius_scale.pack()
    global save_button
    save_button = tk.Button(image_frame_right, text="Lưu", command=save_image, state="disabled")
    save_button.pack()
    update_vignette()
def update_vignette(*args):
    global img_org, label_adj, radius_scale, focus_x_scale, focus_y_scale, value, radius
    if img_org is None:
        return
    radius = radius_scale.get()
    global focus_x_scale
    value = focus_x_scale.get()
    focus_x = focus_x_scale.get()
    focus_y = focus_y_scale.get()
    img_array = np.array(img_org)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)  # Change to HSV color space
    kernel_x = cv2.getGaussianKernel(int(img_array.shape[1] * (0.1 * focus_x + 1)), radius)
    kernel_y = cv2.getGaussianKernel(int(img_array.shape[0] * (0.1 * focus_y + 1)), radius)
    kernel = kernel_y * kernel_x.T
    kernel = kernel / np.linalg.norm(kernel)
    mask = 255 * kernel
    mask_imposed = mask[int(0.1 * focus_y * img_array.shape[0]):, int(0.1 * focus_x * img_array.shape[1]):]
    output = np.copy(img_array)
    output[:, :, 2] = output[:, :, 2] * mask_imposed  # Apply the mask to the Value channel
    adjusted_img = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_HSV2RGB))  # Change back to RGB color space
    display_image(adjusted_img, label_adj)
    enable_save_button()



# Hàm để xoay hình ảnh
def rotate_image():
    # Khai báo các biến toàn cục sẽ được sử dụng trong hàm
    global img_org, angle_slider, label_adj, rotated_image_canvas, angle_var, rotated_img, save_button

    # Hủy các widget trước đó (nếu có)
    destroy_previous_widgets()
    # Kiểm tra xem có hình ảnh nào được mở chưa
    if img_org is None:
        # Nếu không, hiển thị thông báo lỗi và thoát khỏi hàm
        messagebox.showwarning("Lỗi", "Vui lòng mở một ảnh trước!")
        return
    # Tạo một biến StringVar để theo dõi giá trị góc xoay
    angle_var = StringVar()
    # Cập nhật giá trị góc xoay khi biến angle_var thay đổi
    angle_var.trace('w', lambda *args: update_rotation_from_entry(angle_var.get()))

    # Tạo một thanh trượt để điều chỉnh góc xoay
    angle_slider = Scale(image_frame_right, from_=0, to=360, orient="horizontal", label="Góc xoay", showvalue=0, sliderlength=20, length=300, command=update_rotation_from_slider)
    angle_slider.pack()
    # Tạo một hộp nhập liệu để nhập góc xoay
    angle_entry = Entry(image_frame_right, textvariable=angle_var)
    angle_entry.pack()
    # Tạo một nút để lưu hình ảnh đã xoay
    save_button = tk.Button( image_frame_right, text="Lưu",  command=save_image,state="disabled")
    save_button.pack()
    # Tạo một nút để chọn màu nền
    color_button = tk.Button(image_frame_right, text="Chọn màu nền", command=choose_color)
    color_button.pack()
    # Cập nhật hình ảnh xoay
    update_rotation_from_slider()

# Hàm để chọn màu nền
def choose_color():
    global margin_color, angle_slider
    # Hiển thị hộp thoại chọn màu
    color_code = colorchooser.askcolor(title ="Chọn màu")
    # Kiểm tra xem người dùng có chọn màu không
    if color_code[0] is None:  # Người dùng đã hủy bỏ hộp thoại chọn màu
        return
    # Chuyển đổi màu đã chọn thành bộ giá trị RGB
    margin_color = tuple(int(color_code[0][i]) for i in range(3))  # Convert to RGB tuple
    # Cập nhật hình ảnh xoay với màu nền mới
    rotate_and_display(angle_slider.get())

# Hàm để cập nhật góc xoay từ thanh trượt
def update_rotation_from_slider(*args):
    global img_org, angle_slider, label_adj, angle_var, rotated_img, save_button
    # Kiểm tra xem có hình ảnh nào được mở chưa
    if img_org is None:
        return
    # Lấy giá trị góc xoay từ thanh trượt
    angle_val = angle_slider.get()
    # Cập nhật giá trị góc xoay trong hộp nhập liệu
    angle_var.set(str(angle_val)) 
    # Xoay hình ảnh và hiển thị nó
    rotated_img = rotate_and_display(angle_val)
    # Kích hoạt nút lưu
    enable_save_button()

# Hàm để xoay và hiển thị hình ảnh
def rotate_and_display(angle_val):
    global img_org, label_adj, image_frame_right, rotated_img, save_button, margin_color
    # Chuyển đổi góc xoay từ độ sang radian
    angle_rad = radians(float(angle_val))
    # Tính toán kích thước mới của hình ảnh sau khi xoay
    width, height = img_org.size
    new_width = abs(width * cos(angle_rad)) + abs(height * sin(angle_rad))
    new_height = abs(width * sin(angle_rad)) + abs(height * cos(angle_rad))

    # Chuyển hình ảnh gốc sang định dạng RGBA
    img_rgba = img_org.convert('RGBA')

    # Xoay hình ảnh
    rotated_img = img_rgba.rotate(float(angle_val), resample=Image.BICUBIC, expand=True)
    # Tạo một hình ảnh nền mới với màu đã chọn
    background = Image.new('RGBA', rotated_img.size, margin_color + (255,))  
    # Ghép hình ảnh xoay và hình ảnh nền
    final_img = Image.alpha_composite(background, rotated_img)
    # Thay đổi kích thước hình ảnh xoay để phù hợp với khung gốc trong khi duy trì tỷ lệ khung hình
    final_aspect_ratio = final_img.width / final_img.height
    original_aspect_ratio = width / height
    if final_aspect_ratio > original_aspect_ratio:
        new_width = width
        new_height = int(new_width / final_aspect_ratio)
    else:
        new_height = height
        new_width = int(new_height * final_aspect_ratio)
    # Thay đổi kích thước hình ảnh xoay
    final_img = final_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    # Hiển thị hình ảnh xoay
    display_image(final_img, label_adj)
    # Kích hoạt nút lưu
    save_button.config(state="normal") 
    return final_img

# Hàm để cập nhật góc xoay từ hộp nhập liệu
def update_rotation_from_entry(angle_val):
    global img_org, angle_slider, label_adj, angle_var
    # Kiểm tra xem có hình ảnh nào được mở chưa
    if img_org is None:
        return
    try:
        # Chuyển đổi giá trị góc xoay từ chuỗi sang số thực
        angle_val = max(0, min(360, float(angle_val)))
    except ValueError:
        # Nếu không thể chuyển đổi, đặt giá trị góc xoay thành 0
        angle_val = 0
    # Cập nhật giá trị góc xoay trong hộp nhập liệu và thanh trượt
    angle_var.set(str(angle_val))  
    angle_slider.set(angle_val)  
    # Xoay hình ảnh và hiển thị nó
    rotate_and_display(angle_val)
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
    if focus_x_scale is not None:
        focus_x_scale.destroy()
    if focus_y_scale is not None:
        focus_y_scale.destroy()
    # Check if save_button exists before destroying
    if save_button is not None and save_button.winfo_exists():
        save_button.destroy()
def on_mousewheel(event):
    canvas_right.yview_scroll(int(-1*(event.delta/120)), "units")
    canvas_left.yview_scroll(int(-1*(event.delta/120)), "units")

    canvas_right.bind_all("<MouseWheel>", on_mousewheel)
    canvas_left.bind_all("<MouseWheel>", on_mousewheel)


# Tạo cửa sổ gốc
root = tk.Tk()
root.title("Phần mềm Xử lý ánh sáng ảnh")
root.state('zoomed')  # Mở ở chế độ toàn màn hình

header_label = tk.Label(root, text="Phần mềm Xử lý ảnh", font=("Helvetica", 14, "bold"), fg="red")
header_label.pack(side="top", pady=10)

root.configure(bg="pink") 

frame_left = tk.Frame(root, bd=2, relief=tk.SOLID, width=650, height=500)
frame_right = tk.Frame(root, bd=2, relief=tk.SOLID, width=650, height=500)

frame_left.pack(side="left", padx=20, pady=20)
frame_right.pack(side="left", padx=20, pady=20)

# Ngăn các khung từ việc thay đổi kích thước
frame_left.pack_propagate(False)
frame_right.pack_propagate(False)

# Tạo một canvas bên trong khung bên phải
canvas_right = tk.Canvas(frame_right)
canvas_right.pack(side=tk.LEFT, expand=True, fill='both')

# Tạo một thanh cuộn cho canvas bên phải
scrollbar_right = tk.Scrollbar(frame_right, orient="vertical", command=canvas_right.yview)
scrollbar_right.pack(side=tk.RIGHT, fill=tk.Y)

# Cấu hình canvas bên phải
canvas_right.configure(yscrollcommand=scrollbar_right.set)
canvas_right.bind('<Configure>', lambda e: canvas_right.configure(scrollregion=canvas_right.bbox("all")))

# Tạo một khung khác bên trong canvas bên phải
image_frame_right = tk.Frame(canvas_right)

# Thêm khung mới vào canvas bên phải
canvas_right.create_window((0,0), window=image_frame_right, anchor="nw")

label_org = tk.Label(frame_left, width=700, height=700)
label_adj = tk.Label(image_frame_right, width=700, height=700)

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
edit_menu.add_command(label="Điều chỉnh độ sáng và độ tương phản", command=adjust_brightness_contrast)
edit_menu.add_command(label="Hiệu chỉnh gamma", command=adjust_gamma)
edit_menu.add_command(label="Hiệu ứng mờ viền", command=vignette_effect)
edit_menu.add_command(label="Cân bằng màu", command=equalize_image)
edit_menu.add_command(label="Xoay ảnh", command=rotate_image)
menu_bar.add_cascade(label="Chỉnh sửa", menu=edit_menu)
menu_help = tk.Menu(menu_bar)
menu_bar.add_cascade(label="Trợ giúp", menu=menu_help)
exit_menu= tk.Menu(menu_bar)
menu_bar.add_cascade(label="Thoát", menu=exit_menu)
# Main loop
root.mainloop()
