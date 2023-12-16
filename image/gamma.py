
import cv2
import numpy as np
from tkinter import colorchooser, Tk, Scale, Button, Canvas, Label, filedialog
from PIL import Image, ImageTk

class ImageRotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Xoay Ảnh")

        self.angle_label = Label(root, text="Góc xoay:")
        self.angle_label.pack()

        self.angle_slider = Scale(root, from_=0, to=360, orient="horizontal", length=200, command=self.update_rotation)
        self.angle_slider.pack()

        self.margin_color_button = Button(root, text="Chọn màu nền", command=self.choose_margin_color)
        self.margin_color_button.pack()

        self.choose_image_button = Button(root, text="Chọn ảnh", command=self.choose_image)
        self.choose_image_button.pack()

        self.image_path = None
        self.original_image = None
        self.rotated_image = None

        self.rotated_image_label = Label(root, text="Ảnh sau khi xoay:")
        self.rotated_image_label.pack()

        self.rotated_image_canvas = Canvas(root)
        self.rotated_image_canvas.pack()

        self.margin_color = (255, 255, 255)

        self.update_rotation(0)

    def rotate_image(self, angle, margin_color):
        height, width = self.original_image.shape[:2]
        image_center = (width / 2, height / 2)

        rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1)
        
        abs_cos = abs(rotation_matrix[0,0])
        abs_sin = abs(rotation_matrix[0,1])

        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        rotation_matrix[0, 2] += bound_w/2 - image_center[0]
        rotation_matrix[1, 2] += bound_h/2 - image_center[1]

        rotated_image = cv2.warpAffine(self.original_image, rotation_matrix, (bound_w, bound_h), borderMode=cv2.BORDER_CONSTANT, borderValue=margin_color)
        return rotated_image

    def choose_margin_color(self):
        color = colorchooser.askcolor(title="Chọn màu nền")[0]
        self.margin_color = tuple(map(int, color))
        self.update_rotation(self.angle_slider.get())

    def choose_image(self):
        file_path = filedialog.askopenfilename(title="Chọn ảnh", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif")])
        if file_path:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)
            self.update_rotation(self.angle_slider.get())

    def update_rotation(self, angle):
        if self.original_image is not None:
            angle = float(angle)
            self.rotated_image = self.rotate_image(angle, self.margin_color)
            self.display_rotated_image()

    def display_rotated_image(self):
        if self.rotated_image is not None:
            rotated_image_rgb = cv2.cvtColor(self.rotated_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rotated_image_rgb)
            imgtk = ImageTk.PhotoImage(image=pil_image)

            self.rotated_image_canvas.config(width=imgtk.width(), height=imgtk.height())
            self.rotated_image_canvas.create_image(0, 0, anchor="nw", image=imgtk)
            self.rotated_image_canvas.image = imgtk

if __name__ == "__main__":
    root = Tk()
    app = ImageRotator(root)
    root.mainloop()

