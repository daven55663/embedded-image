import numpy as np
from scipy.ndimage import label
import cv2

# 改良版 Labeling 函式，新增紅色標記膨脹
def labeling_with_overlay(lbp_image, top_indices, original_image, min_area=500, kernel_size=3):
    
    # 標記區域
    labeled_image, num_features = label(lbp_image)
    print(f"找到的區域數量: {num_features}")

    # 創建一個過濾小面積區域的掩碼
    filtered_mask = np.zeros_like(labeled_image, dtype=np.uint8)

    for region_label in range(1, num_features + 1):
        region = (labeled_image == region_label)
        if np.sum(region) >= min_area:
            filtered_mask[region] = 255

    # 創建膨脹 kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 在原始影像上疊加紅色標記，並對紅色標記進行膨脹
    overlay_image = original_image.copy()
    for idx in top_indices:
        target_region = (labeled_image == idx).astype(np.uint8)  # 將布林轉成 uint8 格式
        expanded_region = cv2.dilate(target_region, kernel, iterations=1)  # 膨脹處理
        overlay_image[expanded_region == 1] = [0, 0, 255]  # 標記為紅色

    return overlay_image, filtered_mask
