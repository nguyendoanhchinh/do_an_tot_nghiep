def get_sparse_neighbor(p: int, n: int, m: int):
    """Trả về một từ điển, trong đó các khóa là chỉ mục của 4 hàng xóm của `p` trong ma trận thưa,
       và giá trị là các bộ (i, j, x), trong đó `i`, `j` là chỉ mục của hàng xóm trong ma trận bình thường,
       và x là hướng của hàng xóm.

    Tham số:
        p {int} -- chỉ mục trong ma trận thưa.
        n {int} -- số hàng trong ma trận gốc .
        m {int} -- số cột trong ma trận gốc.

    Trả về:
        dict -- từ điển chứa các chỉ mục của 4 hàng xóm của `p`.
    """
    i, j = p // m, p % m
    d = {}
    if i - 1 >= 0:
        d[(i - 1) * m + j] = (i - 1, j, 0)
    if i + 1 < n:
        d[(i + 1) * m + j] = (i + 1, j, 0)
    if j - 1 >= 0:
        d[i * m + j - 1] = (i, j - 1, 1)
    if j + 1 < m:
        d[i * m + j + 1] = (i, j + 1, 1)
    return d
