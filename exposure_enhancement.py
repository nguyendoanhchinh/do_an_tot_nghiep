# 3p
import numpy as np
import cv2
from scipy.spatial import distance
from scipy.ndimage.filters import convolve
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
# project
from utils import get_sparse_neighbor

def create_spacial_affinity_kernel(spatial_sigma: float, size: int = 15):
    """Tạo ma trận kernel (`size` * `size`) sẽ được sử dụng để tính trọng số Gaussian dựa trên độ tương quan không gian.

    Tham số:
        spatial_sigma {float} -- Độ lệch chuẩn không gian.

    Tham số Mặc định:
        size {int} -- Kích thước của kernel. (mặc định: {15})aa
    Returns:
        np.ndarray - Ma trận kernel `size` * `size`
    """
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            kernel[i, j] = np.exp(-0.5 * (distance.euclidean((i, j), (size // 2, size // 2)) ** 2) / (spatial_sigma ** 2))

    return kernel


def compute_smoothness_weights(L: np.ndarray, x: int, kernel: np.ndarray, eps: float = 1e-3):
    """Tính toán trọng số độ mịn được sử dụng để tinh chỉnh vấn đề tối ưu hóa bản đồ chiếu sáng.

    Tham số:
        L {np.ndarray} -- Bản đồ chiếu sáng ban đầu cần được tinh chỉnh.
        x {int} -- Hướng của trọng số. Có thể là x=1 cho chiều ngang hoặc x=0 cho chiều dọc.
        kernel {np.ndarray} -- Ma trận tương quan không gian

    Tham số Mặc định:
        eps {float} -- Hằng số nhỏ để tránh không ổn định trong tính toán. (mặc định: {1e-3})

    Returns:
        np.ndarray - Trọng số độ mịn theo hướng x. Cùng kích thước với `L`.
    """
    Lp = cv2.Sobel(L, cv2.CV_64F, int(x == 1), int(x == 0), ksize=1)
    T = convolve(np.ones_like(L), kernel, mode='constant')
    T = T / (np.abs(convolve(Lp, kernel, mode='constant')) + eps)
    return T / (np.abs(Lp) + eps)


def fuse_multi_exposure_images(im: np.ndarray, under_ex: np.ndarray, over_ex: np.ndarray,
                               bc: float = 1, bs: float = 1, be: float = 1):
    """Thực hiện phương pháp hợp nhất sự chiếu sáng được sử dụng trong bài báo DUAL.

    Tham số:
        im {np.ndarray} -- Hình ảnh đầu vào cần được cải thiện.
        under_ex {np.ndarray} -- Hình ảnh được điều chỉnh vì thiếu sáng. Cùng kích thước với `im`.
        over_ex {np.ndarray} -- Hình ảnh được điều chỉnh vì quá sáng. Cùng kích thước với `im`.
    Tham số Mặc định:
        bc {float} -- Tham số điều khiển ảnh hưởng của đo đạc độ tương phản của Mertens. (mặc định: {1})
        bs {float} -- Tham số điều khiển ảnh hưởng của đo đạc độ bão hòa của Mertens. (mặc định: {1})
        be {float} -- Tham số điều khiển ảnh hưởng của đo đạc độ tốt nghiệp của Mertens. (mặc định: {1})

    Returns:
        np.ndarray -- Hình ảnh được hợp nhất. Cùng kích thước với `im`.
    """
    merge_mertens = cv2.createMergeMertens(bc, bs, be)
    images = [np.clip(x * 255, 0, 255).astype("uint8") for x in [im, under_ex, over_ex]]
    fused_images = merge_mertens.process(images)
    return fused_images
def refine_illumination_map_linear(L: np.ndarray, gamma: float, lambda_: float, kernel: np.ndarray, eps: float = 1e-3):
    """Tinh chỉnh bản đồ chiếu sáng dựa trên bài toán tối ưu hóa được mô tả trong hai bài báo.
       Hàm này sử dụng trình giải nhanh hơn được trình bày trong bài báo LIME.
    Tham số:
        L {np.ndarray} -- Bản đồ chiếu sáng cần được tinh chỉnh.
        gamma {float} -- Hệ số sửa gamma.
        lambda_ {float} -- Hệ số để cân bằng các thành phần trong vấn đề tối ưu hóa.
        kernel {np.ndarray} -- Ma trận tương quan không gian.
    Tham số Mặc định:
        eps {float} -- Hằng số nhỏ để tránh không ổn định trong tính toán (mặc định: {1e-3}).
    Returns:
        np.ndarray -- Bản đồ chiếu sáng được tinh chỉnh. Cùng hình dạng với `L`.
    """
    # Tính toán trọng số độ mịn
    wx = compute_smoothness_weights(L, x=1, kernel=kernel, eps=eps)
    wy = compute_smoothness_weights(L, x=0, kernel=kernel, eps=eps)
    n, m = L.shape
    L_1d = L.copy().flatten()
    # Tính toán ma trận Laplacian không đồng đều không gian năm điểm
    row, column, data = [], [], []
    for p in range(n * m):
        diag = 0
        for q, (k, l, x) in get_sparse_neighbor(p, n, m).items():
            weight = wx[k, l] if x else wy[k, l]
            row.append(p)
            column.append(q)
            data.append(-weight)
            diag += weight
        row.append(p)
        column.append(p)
        data.append(diag)
    F = csr_matrix((data, (row, column)), shape=(n * m, n * m))
    # Giải hệ thống tuyến tính
    Id = diags([np.ones(n * m)], [0])
    A = Id + lambda_ * F
    L_refined = spsolve(csr_matrix(A), L_1d, permc_spec=None, use_umfpack=True).reshape((n, m))
    # Sửa gamma
    L_refined = np.clip(L_refined, eps, 1) ** gamma
    return L_refined

def correct_underexposure(im: np.ndarray, gamma: float, lambda_: float, kernel: np.ndarray, eps: float = 1e-3):
    """Chỉnh sửa sự thiếu sáng bằng thuật toán dựa trên retinex được trình bày trong các bài báo DUAL và LIME.

    Tham số:
        im {np.ndarray} -- Hình ảnh đầu vào cần được chỉnh sửa.
        gamma {float} -- Hệ số sửa gamma.
        lambda_ {float} -- Hệ số để cân bằng các thành phần trong vấn đề tối ưu hóa (trong DUAL và LIME).

    Tham số Mặc định:
        kernel {np.ndarray} -- Ma trận tương quan không gian.
        eps {float} -- Hằng số nhỏ để tránh không ổn định trong tính toán (mặc định: {1e-3})

    Returns:
        np.ndarray -- Hình ảnh được chỉnh sửa vì thiếu sáng. Cùng hình dạng với `im`.
    """

    # Ước lượng đầu tiên của bản đồ chiếu sáng
    L = np.max(im, axis=-1)
    # Tinh chỉnh chiếu sáng
    L_refined = refine_illumination_map_linear(L, gamma, lambda_, kernel, eps)

    # Chỉnh sửa thiếu sáng của hình ảnh
    L_refined_3d = np.repeat(L_refined[..., None], 3, axis=-1)
    im_corrected = im / L_refined_3d
    return im_corrected

# TODO: Resize hình ảnh nếu quá lớn, tối ưu hóa mất quá nhiều thời gian

def enhance_image_exposure(im: np.ndarray, gamma: float, lambda_: float, dual: bool = True, sigma: int = 3,
                           bc: float = 1, bs: float = 1, be: float = 1, eps: float = 1e-3):
    """Nâng cao hình ảnh đầu vào, sử dụng phương pháp DUAL hoặc LIME. Để biết thêm thông tin, vui lòng xem các bài báo gốc.

    Tham số:
        im {np.ndarray} -- Hình ảnh đầu vào cần được chỉnh sửa.
        gamma {float} -- Hệ số sửa gamma.
        lambda_ {float} -- Hệ số để cân bằng các thành phần trong vấn đề tối ưu hóa (trong DUAL và LIME).

    Tham số Mặc định:
        dual {bool} -- Biến boolean để chỉ định phương pháp nâng cao sẽ được sử dụng (hoặc DUAL hoặc LIME) (mặc định: {True})
        sigma {int} -- Độ lệch chuẩn không gian cho trọng số Gaussian dựa trên độ tương quan không gian. (mặc định: {3})
        bc {float} -- Tham số điều khiển ảnh hưởng của đo đạc độ tương phản của Mertens. (mặc định: {1})
        bs {float} -- Tham số điều khiển ảnh hưởng của đo đạc độ bão hòa của Mertens. (mặc định: {1})
        be {float} -- Tham số điều khiển ảnh hưởng của đo đạc độ tốt nghiệp của Mertens. (mặc định: {1})
        eps {float} -- Hằng số nhỏ để tránh không ổn định trong tính toán (mặc định: {1e-3})

    Returns:
        np.ndarray -- Hình ảnh được nâng cao về chiếu sáng. Cùng hình dạng với `im`.
    """
    # Tạo kernel tương quan không gian
    kernel = create_spacial_affinity_kernel(sigma)

    # Chỉnh sửa thiếu sáng
    im_normalized = im.astype(float) / 255.
    under_corrected = correct_underexposure(im_normalized, gamma, lambda_, kernel, eps)

    if dual:
        # Chỉnh sửa quá sáng và hợp nhất nếu chọn phương pháp DUAL
        inv_im_normalized = 1 - im_normalized
        over_corrected = 1 - correct_underexposure(inv_im_normalized, gamma, lambda_, kernel, eps)
        # Hợp nhất hình ảnh
        im_corrected = fuse_multi_exposure_images(im_normalized, under_corrected, over_corrected, bc, bs, be)
    else:
        im_corrected = under_corrected

    # Chuyển đổi thành 8 bit và trả về
    return np.clip(im_corrected * 255, 0, 255).astype("uint8")
