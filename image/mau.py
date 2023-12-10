import cv2
import numpy as np

def vignette(image_path, radius=1, strength=1):
    original = cv2.imread(image_path)
    img = original.copy()

    rows, cols = img.shape[:2]

    # generating vignette mask using Gaussian kernels
    kernel_x = cv2.getGaussianKernel(cols, cols//radius)
    kernel_y = cv2.getGaussianKernel(rows, rows//radius)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    mask = cv2.normalize(mask, None, 0, strength, norm_type=cv2.NORM_MINMAX)

    # apply the mask to each channel in the input image
    for i in range(3):
        img[:,:,i] = img[:,:,i] * mask

    return img, original

# test the function
output, original = vignette('test.jpg', radius=2, strength=2)
cv2.imshow('Original', original)
cv2.imshow('Vignette', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
