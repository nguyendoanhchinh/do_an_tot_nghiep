import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import Scale
from PIL import Image, ImageTk
import numpy as np

# Khởi tạo biến ảnh
image = None
img_org = None

# Tạo cửa sổ gốc
root = tk.Tk()
root.title("Phần mềm Xử lý ảnh")
root.geometry("800x500")

# Tạo khung chứa các thành phần
frame_left = tk.Frame(root)
frame_right = tk.Frame(root)
frame_left.pack(side="left", padx=10, pady=10)
frame_right.pack(side="left", padx=10, pady=10)

# Hàm mở tệp
def open_file():
    global image, label_org, img_org
    file_path = filedialog.askopenfilename()

    if file_path:
        # Kiểm tra phần mở rộng của tệp
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')):
            # Tải ảnh gốc
            img_org = Image.open(file_path)
            img_org = img_org.resize((300, 350), Image.LANCZOS)

            # Hiển thị ảnh gốc
            photo = ImageTk.PhotoImage(img_org)
            label_org.configure(image=photo)
            label_org.image = photo

            # Lưu ảnh gốc vào biến 'image'
            image = np.array(img_org)
            
            # Hiển thị các chức năng chỉ sau khi thêm ảnh
            scale_r.pack()
            scale_g.pack()
            scale_b.pack()
            btn_gamma.pack()
        else:
            messagebox.showwarning("Lỗi", "Định dạng ảnh không được hỗ trợ.")

# Hàm điều chỉnh gamma
def adjust_gamma():
    global image, label_adj

    # Lấy giá trị từ thanh trượt
    gamma_r = float(scale_r.get())
    gamma_g = float(scale_g.get())
    gamma_b = float(scale_b.get())

    # Tạo bản sao của ảnh gốc để điều chỉnh gamma
    adjusted_image = np.copy(image)

    # Áp dụng chỉ số gamma cho từng kênh màu
    adjusted_image[:, :, 0] = np.power(adjusted_image[:, :, 0] / 255.0, gamma_r) * 255.0
    adjusted_image[:, :, 1] = np.power(adjusted_image[:, :, 1] / 255.0, gamma_g) * 255.0
    adjusted_image[:, :, 2] = np.power(adjusted_image[:, :, 2] / 255.0, gamma_b) * 255.0

    # Chuyển đổi mảng numpy thành đối tượng Image
    adjusted_image = Image.fromarray(adjusted_image.astype(np.uint8))

    # Hiển thị ảnh đã điều chỉnh gamma
    photo = ImageTk.PhotoImage(adjusted_image)
    label_adj.configure(image=photo)
    label_adj.image = photo

# Tạo các nhãn cho khung trái và khung phải
label_org = tk.Label(frame_left)
label_adj = tk.Label(frame_right)

# Tạo thanh trượt cho hệ màu RGB
scale_r = Scale(frame_right, from_=0, to=5, resolution=0.1, orient="horizontal", label="Gamma R", length=300)
scale_g = Scale(frame_right, from_=0, to=5, resolution=0.1, orient="horizontal", label="Gamma G", length=300)
scale_b = Scale(frame_right, from_=0, to=5, resolution=0.1, orient="horizontal", label="Gamma B", length=300)

# Tạo nút để thực hiện chức năng điều chỉnh gamma
btn_gamma = tk.Button(frame_right, text="Điều chỉnh gamma", command=adjust_gamma)

# Hiển thị khung ảnh trống ban đầu
label_org.pack()
frame_right.pack()
label_adj.pack()

# Hiển thị các thành phần khi chọn thêm ảnh
btn_open = tk.Button(frame_left, text="Mở ảnh", command=open_file)
btn_open.pack()

# Chạy chương trình
root.mainloop()
