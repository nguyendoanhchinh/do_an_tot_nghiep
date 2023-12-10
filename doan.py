import cv2
import tkinter as tk
from tkinter import filedialog, simpledialog
from tkinter import messagebox
from tkinter import Scale
from PIL import Image, ImageTk
import numpy as np

# Khởi tạo biến image
image = None
img_org = None
save_button = None
# Tạo cửa sổ gốc
root = tk.Tk()
root.title("Phần mềm Xử lý ảnh")
# Đặt kích thước cửa sổ
root.geometry("800x500")

# Tạo khung chứa các thành phần
frame_left = tk.Frame(root, bd=2, relief=tk.SOLID)
frame_right = tk.Frame(root, bd=2, relief=tk.SOLID)
frame_left.pack(side="left", padx=20, pady=20)
frame_right.pack(side="left", padx=10, pady=10)

# Tạo khung con trong frame_left để hiển thị ảnh gốc
image_frame_left = tk.Frame(frame_left, bd=2, relief=tk.SOLID)
image_frame_left.pack()

# Tạo khung con trong frame_right để hiển thị ảnh đã điều chỉnh và các điều khiển
image_frame_right = tk.Frame(frame_right, bd=2, relief=tk.SOLID)
image_frame_right.pack()

# Hàm chức năng Mở
def open_file():
    global image, label, img_org  
    file_path = filedialog.askopenfilename()
  
    if file_path:
        # Kiểm tra đuôi file
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif','.webp')):
            # Tải ảnh gốc
            img_org = Image.open(file_path)
            img_org = img_org.resize((300, 350), Image.Resampling.LANCZOS)
            
            # Hiển thị ảnh gốc
            photo = ImageTk.PhotoImage(img_org)
            try:
                label.configure(image=photo)
                label.image = photo
            except NameError:
                label = tk.Label(image_frame_left, image=photo)
                label.image = photo
                label.pack()
            
           # Lưu ảnh gốc vào biến 'image'
            image = np.array(img_org)
        else:
            messagebox.showwarning("Lỗi", "Định dạng ảnh không hỗ trợ.")
def enable_save_button():
    global save_button
    if save_button:
        save_button.config(state="normal")
# Hàm điều chỉnh độ sáng và độ tương phản
def adjust_brightness_and_contrast():
    global img_org, brightness_scale, contrast_scale

    if img_org is None:
        messagebox.showwarning("Lỗi", "Vui lòng mở một ảnh trước!")
        return

    # Tạo thanh điều chỉnh độ sáng và độ tương phản
    brightness_scale = Scale(image_frame_right, from_=0, to=255, orient="horizontal", label="Độ sáng",
                             showvalue=1, sliderlength=20, length=300, command=update_adjusted_image)
    brightness_scale.pack()

    contrast_scale = Scale(image_frame_right, from_=0.1, to=3.0, orient="horizontal", label="Độ tương phản",
                           resolution=0.1, showvalue=1, sliderlength=20, length=300, command=update_adjusted_image)
    contrast_scale.pack()

    # Gọi hàm cập nhật ảnh đã điều chỉnh
    update_adjusted_image()

# Hàm cập nhật ảnh đã điều chỉnh
def update_adjusted_image(*args):
    global img_org, brightness_scale, contrast_scale, label

    if img_org is None:
        return

    brightness_val = brightness_scale.get() / 100
    contrast_val = contrast_scale.get()

    if brightness_val == 1.0 and contrast_val == 1.0:
        # Nếu cả độ sáng và độ tương phản đều ở giá trị mặc định, hiển thị ảnh gốc
        photo = ImageTk.PhotoImage(img_org)
        label.configure(image=photo)
        label.image = photo
    else:
        adjusted_img = img_org.copy()

        # Điều chỉnh độ sáng
        adjusted_img = adjusted_img.point(lambda p: p * brightness_val)

        # Điều chỉnh độ tương phản
        for x in range(adjusted_img.width):
            for y in range(adjusted_img.height):
                r, g, b = adjusted_img.getpixel((x, y))
                new_r = int((r - 128) * contrast_val + 128)
                new_g = int((g - 128) * contrast_val + 128)
                new_b = int((b - 128) * contrast_val + 128)
                adjusted_img.putpixel((x, y), (new_r, new_g, new_b))

        # Hiển thị ảnh đã điều chỉnh
        photo = ImageTk.PhotoImage(adjusted_img)
        label.configure(image=photo)
        label.image = photo

# Hàm chức năng Hiệu chỉnh gamma
def adjust_gamma():
    global image, img_org, label
    if img_org is not None:
        gamma = simpledialog.askfloat("Gamma", "Nhập giá trị gamma (0.1 đến 5.0):", minvalue=0.1, maxvalue=5.0)
        if gamma is not None:
            table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
            adjusted_img = cv2.LUT(image, table)
            adjusted_img = Image.fromarray(adjusted_img)
            adjusted_img = adjusted_img.resize((300, 350), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(adjusted_img)
            label.configure(image=photo)
            label.image = photo

# Hàm chức năng Tạo độ sắc nét
def sharpen():
    global image, img_org, label
    if img_org is not None:
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        sharpened_img = cv2.filter2D(image, -1, kernel)
        sharpened_img = Image.fromarray(sharpened_img)
        sharpened_img = sharpened_img.resize((300, 350), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(sharpened_img)
        label.configure(image=photo)
        label.image = photo

# Hàm chức năng Cân bằng màu
def balance_color():
    global image, img_org, label
    if img_org is not None:
        balanced_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        balanced_img[:, :, 0] = cv2.equalizeHist(balanced_img[:, :, 0])
        balanced_img = cv2.cvtColor(balanced_img, cv2.COLOR_LAB2BGR)
        balanced_img = Image.fromarray(balanced_img)
        balanced_img = balanced_img.resize((300, 350), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(balanced_img)
        label.configure(image=photo)
        label.image = photo

# Hàm chức năng Khử nhiễu trong ảnh
def image_filter():
    global image, img_org, label
    if img_org is not None:
        # Thực hiện xử lý khử nhiễu ở đây
        # ...
        # Ví dụ: xử dụng GaussianBlur
        denoised_img = cv2.GaussianBlur(image, (5, 5), 0)
        denoised_img = Image.fromarray(denoised_img)
        denoised_img = denoised_img.resize((300, 350), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(denoised_img)
        label.configure(image=photo)
        label.image = photo

# Hàm chức năng Thoát
def exit_program():
    root.destroy()

# Tạo các nhãn cho khung trái và khung phải
label_org = tk.Label(image_frame_left)
label_adj = tk.Label(image_frame_right)

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
image_menu.add_command(label="Điều chỉnh độ sáng và độ tương phản", command=adjust_brightness_and_contrast)
image_menu.add_command(label="Hiệu chỉnh gamma", command=adjust_gamma)
image_menu.add_command(label="Tạo độ sắc nét", command=sharpen)
image_menu.add_command(label="Cân bằng màu", command=balance_color)
image_menu.add_command(label="Khử nhiễu trong ảnh", command=image_filter)

# Tạo menu Cửa sổ
window_menu = tk.Menu(menu_bar)
menu_bar.add_cascade(label="Cửa sổ", menu=window_menu)
window_menu.add_command(label="Sắp xếp", command=None)  # Thêm chức năng sắp xếp ở đây

# Tạo menu Trợ giúp
help_menu = tk.Menu(menu_bar)
menu_bar.add_cascade(label="Trợ giúp", menu=help_menu)

root.mainloop()
