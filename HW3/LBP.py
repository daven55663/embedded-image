import numpy as np

def lbp_feature_extraction(image_gray):
    # 定義LBP的半徑和鄰居點數量
    radius = 1  # LBP半徑
    n_points = 8 * radius  # LBP鄰居點數量
    
    # 取得影像的高度和寬度
    height, width = image_gray.shape
    # 初始化LBP結果影像
    lbp_result = np.zeros((height, width), dtype=np.uint8)
    
    # 定義鄰居位置相對於中心像素的偏移量
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, 1), (1, 1), (1, 0),
        (1, -1), (0, -1)
    ]
    
    # 遍歷影像的每個像素（忽略邊界）
    for i in range(radius, height - radius):
        for j in range(radius, width - radius):
            center_pixel = image_gray[i, j]
            binary_pattern = 0
            for k, (dy, dx) in enumerate(offsets):
                neighbor_pixel = image_gray[i + dy, j + dx]
                if neighbor_pixel >= center_pixel:
                    binary_pattern |= (1 << k)
            lbp_result[i, j] = binary_pattern
    
    # 將LBP結果轉換為8位元格式以便顯示
    lbp_result = np.uint8(255 * (lbp_result / lbp_result.max()))
    
    # 回傳LBP結果
    return lbp_result