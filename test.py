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
# Khởi tạo biến image
image = None
panelA = None
panelB = None
img_org = None

# Tạo cửa sổ gốc
root = tk.Tk()
root.title("Phần mềm Xử lý ảnh")
# Đặt kích thước cửa sổ
root.geometry("500x400")

# Cho phép cửa sổ thay đổi kích thước
root.resizable(True, True)
# Hàm chức năng Mở
def open_file():
    global image, label, img_org  # Add img_org to store the original image
    file_path = filedialog.askopenfilename()
  
    if file_path:
        # Kiểm tra đuôi file
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            # Load the original image
            img_org = Image.open(file_path)
            img_org = img_org.resize((250, 300), Image.Resampling.LANCZOS)
            
            # Display the original image
            photo = ImageTk.PhotoImage(img_org)
            try:
                label.configure(image=photo)
                label.image = photo
            except NameError:
                label = tk.Label(root, image=photo)
                label.image = photo
                label.pack()
            
            # Store the original image in the 'image' variable
            image = np.array(img_org)
        else:
            print("Định dạng ảnh không hỗ trợ.")


# Hàm chức năng Lưu
def save_file():
    return true;
#Tùy Chỉnh độ sáng ảnh
def adjust_brightness():
    global image, brightness_scale, brightness_label
    if img_org is None:
        messagebox.showwarning("Lỗi", "Vui lòng chọn  ảnh trước !")
        return
    if image is not None:
        brightness_scale = tk.Scale(root, from_=-255, to=255, orient="horizontal", label="Độ sáng", showvalue=1, sliderlength=20, length=300)
        brightness_scale.pack()
        brightness_scale.set(0)  # Đặt giá trị mặc định cho thanh kéo

        brightness_label = tk.Label(root, text="0", font=("Helvetica", 12))
        brightness_label.pack()

        brightness_scale.bind("<B1-Motion>", on_brightness_change)
        brightness_scale.bind("<ButtonRelease-1>", update_brightness)

        update_brightness()

def on_brightness_change(event):
    global image, brightness_scale, brightness_label

    brightness_value = int(brightness_scale.get())
    adjusted_image = cv2.convertScaleAbs(np.array(image), alpha=1, beta=brightness_value)
    cv2.imshow("Brightness Adjustment", adjusted_image)

    # Cập nhật giá trị chỉ số độ sáng
    brightness_label.config(text=str(brightness_value))

def update_brightness(event=None):
    global image, brightness_scale

    brightness_value = int(brightness_scale.get())
    adjusted_image = cv2.convertScaleAbs(np.array(image), alpha=1, beta=brightness_value)
    cv2.imshow("Brightness Adjustment", adjusted_image)
# Hàm chức năng Hiệu chỉnh độ sáng
def adjust_contrast():
    global image, image_tk, image_label
    if image is not None:
        contrast = tk.simpledialog.askfloat("Contrast", "Enter contrast value (0.0 to 3.0):", minvalue=0.0, maxvalue=3.0)
        if contrast is not None:
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)
            image = cv2.addWeighted(image, alpha_c, np.zeros(image.shape, dtype=image.dtype), 0, gamma_c)
            image_pil = Image.fromarray(image)
            image_tk = ImageTk.PhotoImage(image_pil)
            if image_label is None:
                image_label = tk.Label(root, image=image_tk)
                image_label.pack()
            else:
                image_label.config(image=image_tk)

# Hàm chức năng Hiệu chỉnh gamma
def adjust_gamma():
    global image, image_tk, image_label
    if image is not None:
        gamma = tk.simpledialog.askfloat("Gamma", "Enter gamma value (0.1 to 5.0):", minvalue=0.1, maxvalue=5.0)
        if gamma is not None:
            table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
            image = cv2.LUT(image, table)
            cv2.imshow("Image", result)
            cv2.waitKey(1)

# Hàm chức năng Cân bằng màu
def balance_color():
    global img_org

    if img_org is None:
        messagebox.showwarning("Lỗi", "Vui lòng mở một ảnh trước khi cân bằng màu!")
        return

    # Thực hiện cân bằng màu
    if not isinstance(img_org, np.ndarray):
        img_org = np.array(img_org)

    # Tiếp tục với các bước cân bằng màu
    img_yuv = cv2.cvtColor(img_org, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_equa = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    # Tính toán histogram
    hist_org = cv2.calcHist([img_org], [0], None, [256], [0, 256])
    hist_equa = cv2.calcHist([img_equa], [0], None, [256], [0, 256])

    # Tạo cửa sổ
    plt.figure(figsize=(10, 6))
    plt.tight_layout()

    # Hiển thị ảnh gốc
    plt.subplot(2, 2, 1)
    plt.imshow(img_org)
    plt.axis('off')
    plt.title('Ảnh gốc')

    # Hiển thị biểu đồ histogram ảnh gốc
    plt.subplot(2, 2, 2)
    plt.plot(hist_org)
    plt.title('Histogram ảnh gốc')

    # Hiển thị ảnh sau khi cân bằng màu
    plt.subplot(2, 2, 3)
    plt.imshow(img_equa)
    plt.axis('off')
    plt.title('Ảnh sau khi cân bằng màu')

    # Hiển thị biểu đồ histogram ảnh sau khi cân bằng màu
    plt.subplot(2, 2, 4)
    plt.plot(hist_equa)
    plt.title('Histogram sau khi cân bằng')

    # Hiển thị cửa sổ
    plt.show()

    # Tạo cửa sổ Tkinter để chứa nút "Lưu ảnh"
    root = Tk()

    def save_image():
        # Lưu ảnh sau khi cân bằng
        output_path = filedialog.asksaveasfilename(defaultextension='.jpg', filetypes=[('JPEG Files', '*.jpg')])
        if output_path:
            cv2.imwrite(output_path, img_equa)
            messagebox.showinfo("Thông báo", "Đã lưu ảnh sau khi cân bằng vào:\n" + output_path)

    # Tạo nút "Lưu ảnh"
    button_save = Button(root, text="Lưu ảnh", command=save_image)
    button_save.pack()

    # Hiển thị cửa sổ Tkinter
    root.mainloop()

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
file_menu.add_command(label="Lưu", command=save_file)
file_menu.add_separator()

file_menu.add_command(label="Thoát", command=exit_program)

# Tạo menu Xử lý ảnh
image_menu = tk.Menu(menu_bar)
menu_bar.add_cascade(label="Xử lý ảnh", menu=image_menu)
image_menu.add_command(label="Hiệu chỉnh ánh sáng", command=adjust_brightness)
image_menu.add_command(label="Hiệu chỉnh độ tương phản", command=adjust_contrast)
image_menu.add_command(label="Hiệu chỉnh gamma", command=adjust_gamma)
image_menu.add_command(label="Cân bằng màu", command=balance_color)

# Tạo menu Cửa sổ
window_menu = tk.Menu(menu_bar)
menu_bar.add_cascade(label="Cửa sổ", menu=window_menu)
window_menu.add_command(label="Sắp xếp", command=None)  # Thêm chức năng sắp xếp ở đây

# Tạo menu Trợ giúp
help_menu = tk.Menu(menu_bar)
menu_bar.add_cascade(label="Trợ giúp", menu=help_menu)
help_menu.add

root.mainloop()
