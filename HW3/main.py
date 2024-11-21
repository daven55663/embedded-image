import cv2
import numpy as np
from Sobel import sobel_edge_detection
from LBP import lbp_feature_extraction
from histogram_function import calculate_histogram, find_top_three
from Labeling import labeling_with_overlay

# 主程式
if __name__ == "__main__":
    # 載入原始影像
    image_path = 'mtest.jpg'
    image = cv2.imread(image_path)

    if image is None:
        print("無法讀取影像")
    else:
        # HSV 過濾道路部分
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([0, 0, 0])
        upper_yellow = np.array([180, 200, 100])
        mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
        road_hsv_filtered = cv2.bitwise_and(image, image, mask=mask)

        # Sobel 邊緣檢測
        image_gray = cv2.cvtColor(road_hsv_filtered, cv2.COLOR_BGR2GRAY)
        sobel_result = sobel_edge_detection(image_gray)

        # LBP 特徵提取
        lbp_result = lbp_feature_extraction(sobel_result)

        # 計算直方圖並找出前三大值
        lbp_hist = calculate_histogram(lbp_result)
        top_indices = find_top_three(lbp_hist)

        # 呼叫 Labeling，將結果疊加到原始影像，並放大紅色標記
        road_highlighted, filtered_mask = labeling_with_overlay(
            lbp_result, top_indices, image, min_area=500, kernel_size=3
        )

        # 儲存與顯示結果
        cv2.imshow("Sobel", sobel_result)
        cv2.imshow("LBP", lbp_result)
        cv2.imshow("Road Highlighted", road_highlighted)
        cv2.imshow("HSV Filtered Road", road_hsv_filtered)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
