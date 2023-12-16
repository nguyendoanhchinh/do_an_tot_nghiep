import cv2
import tkinter as tk
from tkinter import filedialog, Menu, messagebox, Scale
from PIL import Image, ImageTk
import numpy as np

class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Phần mềm Xử lý ảnh")
        self.window_width, self.window_height = 1000, 500
        self.center_window()

        self.image = None
        self.img_org = None
        self.label_org = None
        self.label_adj = None
        self.save_button = None
        self.brightness_scale = None
        self.contrast_scale = None
        self.gamma_scale = None
        self.radius_scale = None
        self.focus_scale = None
        self.current_function = None

        self.create_layout()

    def center_window(self):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - self.window_width) // 2
        y = (screen_height - self.window_height) // 2
        self.root.geometry(f"{self.window_width}x{self.window_height}+{x}+{y}")

    def create_layout(self):
        self.create_menu()
        self.create_frames()
        self.create_labels()

    def create_menu(self):
        menu_bar = Menu(self.root)
        self.root.config(menu=menu_bar)

        file_menu = Menu(menu_bar)
        file_menu.add_command(label="Mở", command=self.open_file)
        file_menu.add_separator()
        file_menu.add_command(label="Thoát", command=self.root.destroy)
        menu_bar.add_cascade(label="Tệp", menu=file_menu)

        edit_menu = Menu(menu_bar)
        edit_menu.add_command(label="Điều chỉnh độ sáng", command=lambda: self.switch_function("adjustment", "brightness"))
        edit_menu.add_command(label="Điều chỉnh độ tương phản", command=lambda: self.switch_function("adjustment", "contrast"))
        edit_menu.add_command(label="Hiệu chỉnh gamma", command=lambda: self.switch_function("adjustment", "gamma"))
        edit_menu.add_command(label="Hiệu ứng mờ viền", command=lambda: self.switch_function("vignette"))
        menu_bar.add_cascade(label="Chỉnh sửa", menu=edit_menu)

    def create_frames(self):
        frame_left = tk.Frame(self.root, bd=2, relief=tk.SOLID)
        frame_right = tk.Frame(self.root, bd=2, relief=tk.SOLID)
        frame_left.pack(side="left", padx=20, pady=20)
        frame_right.pack(side="left", padx=20, pady=20)

        self.image_frame_left = tk.Frame(frame_left, bd=2, relief=tk.SOLID)
        self.image_frame_left.pack()

        self.image_frame_right = tk.Frame(frame_right, bd=2, relief=tk.SOLID)
        self.image_frame_right.pack()

    def create_labels(self):
        self.label_org = tk.Label(self.image_frame_left, width=65, height=250)
        self.label_adj = tk.Label(self.image_frame_right, width=65, height=250)
        self.label_org.pack()
        self.label_adj.pack()

    def open_file(self):
        file_path = filedialog.askopenfilename()

        if file_path:
            if file_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")):
                self.img_org = Image.open(file_path)
                self.display_original_image()
                self.enable_save_button()
            else:
                messagebox.showwarning("Lỗi", "Định dạng ảnh không được hỗ trợ.")

    def display_original_image(self):
        original_label_width = self.label_org.winfo_width()
        original_label_height = self.label_org.winfo_height()

        self.img_org = self.resize_image_to_label(self.img_org, original_label_width, original_label_height)

        self.update_label_size(self.label_org, original_label_width, original_label_height)
        self.update_label_size(self.label_adj, original_label_width, original_label_height)

        self.display_image(self.img_org, self.label_org)
        self.image = np.array(self.img_org)

    def switch_function(self, function, sub_function=None):
        self.destroy_previous_widgets()
        self.current_function = function

        if function == "adjustment":
            self.adjustment_function(sub_function)
        elif function == "vignette":
            self.vignette_function()

    def adjustment_function(self, sub_function):
        scale_range = (0, 255) if sub_function == "brightness" else (0.1, 3.0) if sub_function == "contrast" else (0.1, 10)
        scale_label = "Độ sáng" if sub_function == "brightness" else "Độ tương phản" if sub_function == "contrast" else "Gamma"

        scale = Scale(self.image_frame_right, from_=scale_range[0], to=scale_range[1], orient="horizontal", label=scale_label, resolution=0.1, showvalue=1, sliderlength=20, length=300, command=lambda x: self.update_image(sub_function, x))
        scale.pack()

        self.save_button = tk.Button(self.image_frame_right, text="Lưu", command=self.save_image, state="disabled")
        self.save_button.pack()

        self.update_image(sub_function, scale.get())

    def update_image(self, adjustment_type, value):
        if self.img_org is None:
            return

        value = float(value)

        if adjustment_type == "brightness":
            adjusted_img = self.img_org.point(lambda p: p * (value / 100))
        elif adjustment_type == "contrast":
            adjusted_img = self.img_org.copy()
            adjusted_img = adjusted_img.point(lambda p: p * (value / 100))
        else:  # adjustment_type == "gamma"
            table = np.array([(i / 255.0) ** (1.0 / value) * 255 for i in np.arange(0, 256)]).astype("uint8")
            adjusted_img = cv2.LUT(self.image, table)
            adjusted_img = Image.fromarray(adjusted_img)

        adjusted_img = self.resize_image_to_label(adjusted_img, self.label_adj.winfo_width(), self.label_adj.winfo_height())
        self.display_image(adjusted_img, self.label_adj)
        self.enable_save_button()

    def vignette_function(self):
        self.radius_scale = Scale(self.image_frame_right, from_=1, to=500, orient="horizontal", label="Radius", resolution=1, showvalue=1, sliderlength=20, length=300, command=self.update_vignette)
        self.radius_scale.set(360)
        self.radius_scale.pack()

        self.focus_scale = Scale(self.image_frame_right, from_=1, to=10, orient="horizontal", label="Focus", resolution=1, showvalue=1, sliderlength=20, length=300, command=self.update_vignette)
        self.focus_scale.set(1)
        self.focus_scale.pack()

        self.save_button = tk.Button(self.image_frame_right, text="Lưu", command=self.save_image, state="disabled")
        self.save_button.pack()

        self.update_vignette()

    def update_vignette(self, *args):
        if self.img_org is None:
            return

        radius = self.radius_scale.get()
        focus = self.focus_scale.get()

        img_array = np.array(self.img_org)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        kernel_x = cv2.getGaussianKernel(int(img_array.shape[1] * (0.1 * focus + 1)), radius)
        kernel_y = cv2.getGaussianKernel(int(img_array.shape[0] * (0.1 * focus + 1)), radius)
        kernel = kernel_y * kernel_x.T

        kernel = kernel / np.linalg.norm(kernel)

        mask = 255 * kernel
        mask_imposed = mask[int(0.1 * focus * img_array.shape[0]):, int(0.1 * focus * img_array.shape[1]):]

        output = np.copy(img_array)
        for i in range(3):
            output[:, :, i] = output[:, :, i] * mask_imposed

        adjusted_img = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        self.display_image(adjusted_img, self.label_adj)
        self.enable_save_button()

    def enable_save_button(self):
        if self.save_button:
            self.save_button.config(state="normal")

    def save_image(self):
        if self.label_adj.image:
            image_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("JPEG files", "*.jpg"),
                    ("All files", "*.*"),
                ],
            )
            if image_path:
                adjusted_img = ImageTk.getimage(self.label_adj.image)
                adjusted_img.save(image_path)

    def destroy_previous_widgets(self):
        self.destroy_current_function_widgets()

    def destroy_current_function_widgets(self):
        if self.current_function == "adjustment":
            if self.brightness_scale:
                self.brightness_scale.destroy()
            if self.contrast_scale:
                self.contrast_scale.destroy()
            if self.gamma_scale:
                self.gamma_scale.destroy()
        elif self.current_function == "vignette":
            if self.radius_scale:
                self.radius_scale.destroy()
            if self.focus_scale:
                self.focus_scale.destroy()

    def display_image(self, img, label):
        photo = ImageTk.PhotoImage(img)
        label.configure(image=photo)
        label.image = photo

    def resize_image_to_label(self, img, label_width, label_height):
        aspect_ratio = img.width / img.height
        new_width = min(img.width, label_width)
        new_height = int(new_width / aspect_ratio)
        if new_height > label_height:
            new_height = label_height
            new_width = int(new_height * aspect_ratio)
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def update_label_size(self, label, width, height):
        label.config(width=width, height=height)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessor(root)
    root.mainloop()
