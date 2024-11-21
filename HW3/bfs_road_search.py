import numpy as np
from collections import deque

def bfs_road_search(image, similarity_threshold=30, min_area=500):
    # 初始化BFS所需的數據結構
    height, width = image.shape
    visited = np.zeros((height, width), dtype=bool)
    queue = deque()
    road_pixels = []

    # 在影像最下方找到最常見的紋理並作為起始點
    start_row = height - 1
    start_col = np.argmax(image[start_row, :])
    queue.append((start_row, start_col))
    visited[start_row, start_col] = True

    # 定義八個可能的移動方向（上下左右及對角線）
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]

    # 開始BFS搜索
    while queue:
        x, y = queue.popleft()
        road_pixels.append((x, y))

        # 循環檢查所有可能的方向
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy

            # 確保新位置在影像範圍內且未被訪問過
            if 0 <= new_x < height and 0 <= new_y < width and not visited[new_x, new_y]:
                # 比較紋理值是否相近（增加相似度閾值來過濾不相干的區域）
                if abs(int(image[new_x, new_y]) - int(image[x, y])) <= similarity_threshold:
                    queue.append((new_x, new_y))
                    visited[new_x, new_y] = True
    
    # 過濾掉面積小於指定值的區域
    if len(road_pixels) < min_area:
        return []
    
    # 返回找到的道路區域的所有像素座標
    return road_pixels

