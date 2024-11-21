import numpy as np

def l1_norm_distance(hist1, hist2, threshold=1000):
    # 計算 L1 範數距離
    distance = np.sum(np.abs(hist1 - hist2))
    print(f"L1範數距離: {distance}")
    
    # 如果距離小於閾值，則認為兩個直方圖相似
    return distance < threshold
