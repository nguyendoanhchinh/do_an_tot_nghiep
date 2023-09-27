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
root.geometry("500x500")

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
                label = tk.Label(root, image=photo)
                label.image = photo
                label.pack()
            
           # Lưu ảnh gốc vào biến 'image'
            image = np.array(img_org)
        else:
            print("Định dạng ảnh không hỗ trợ.")



#Tùy Chỉnh độ sáng ảnh


def brightness():
    global img_org

    if img_org is None:
        messagebox.showwarning("Lỗi", "Vui lòng mở một ảnh trước!")
        return

    def adjust_brightness(val):
        val = int(val) / 100

        img = img_org.copy()
        img = img.point(lambda p: p * val)

        img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_pil = Image.fromarray(img_cv2)

        cv2.namedWindow('Ảnh sau khi điều chỉnh độ sáng', cv2.WINDOW_NORMAL)
        cv2.imshow('Ảnh sau khi điều chỉnh độ sáng', img_cv2)
        cv2.resizeWindow('Ảnh sau khi điều chỉnh độ sáng', img_pil.width, img_pil.height)

    brightness_scale = Scale(root, from_=0, to=255, orient="horizontal", label="Độ sáng",
                             showvalue=1, sliderlength=20, length=300, command=adjust_brightness)
    brightness_scale.pack()

    # Hiển thị ảnh gốc ban đầu
    photo = ImageTk.PhotoImage(image=img_org)
    label = Label(root, image=photo)
    label.image = photo
    label.pack()
# Hàm chức năng Hiệu chỉnh độ tương phản
def adjust_contrast():
    return true;
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

# Hàm chức năng Cân bằng sáng
def balance_color():
    global img_org

    if img_org is None:
        messagebox.showwarning("Lỗi", "Vui lòng mở một ảnh trước!")
        return

    # Thực hiện cân bằng màu
    if not isinstance(img_org, np.ndarray):
        img_org = np.array(img_org)

    # Tiếp tục với các bước cân bằng màu
    img_yuv = cv2.cvtColor(img_org, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_equa = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
# Chỉnh kích thước ảnh
    img_org_resized = cv2.resize(img_org, (400, 300))
    img_equa_resized = cv2.resize(img_equa, (400, 300))
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
        nonlocal img_equa  # Sử dụng biến img_equa trong phạm vi hàm này

        # Lưu ảnh sau khi cân bằng
        output_path = filedialog.asksaveasfilename(defaultextension='.jpg', filetypes=[('JPEG Files', '*.jpg')])
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(img_equa, cv2.COLOR_RGB2BGR))  # Chuyển đổi về BGR trước khi lưu
            messagebox.showinfo("Thông báo", "Đã lưu ảnh sau khi cân bằng vào:\n" + output_path)

    # Tạo nút "Lưu ảnh"
    button_save = Button(root, text="Lưu ảnh", command=save_image)
    button_save.pack()

# Khử nhiễu trong ảnh
def image_filter():
    return true;
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
image_menu.add_command(label="Hiệu chỉnh ánh sáng", command=brightness)
image_menu.add_command(label="Hiệu chỉnh độ tương phản", command=adjust_contrast)
image_menu.add_command(label="Hiệu chỉnh gamma", command=adjust_gamma)
image_menu.add_command(label="Cân bằng ảnh", command=balance_color)
image_menu.add_command(label="Khử nhiễu trong ảnh", command=image_filter)
# Tạo menu Cửa sổ
window_menu = tk.Menu(menu_bar)
menu_bar.add_cascade(label="Cửa sổ", menu=window_menu)
window_menu.add_command(label="Sắp xếp", command=None)  # Thêm chức năng sắp xếp ở đây

# Tạo menu Trợ giúp
help_menu = tk.Menu(menu_bar)
menu_bar.add_cascade(label="Trợ giúp", menu=help_menu)
help_menu.add

root.mainloop()
