import cv2
import numpy as np

# 讀取灰階影像
def sobel_edge_detection(image_gray):
    # 使用Sobel運算子計算X方向上的梯度
    sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    # 使用Sobel運算子計算Y方向上的梯度
    sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)

    # 計算兩個方向的梯度幅度(將X和Y方向的梯度進行加權計算)
    abs_sobelx = cv2.convertScaleAbs(sobel_x)
    abs_sobely = cv2.convertScaleAbs(sobel_y)
    sobel_combined = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)

    # 將梯度幅度影像轉換為8位元格式
    sobel_combined = np.uint8(sobel_combined)

        
    # 回傳Sobel結果
    return sobel_combined

