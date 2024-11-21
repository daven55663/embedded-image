import numpy as np
import cv2
import matplotlib.pyplot as plt

def calculate_histogram(image):
    # 計算灰度影像的直方圖
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.flatten()
    return hist

def find_top_three(hist):
    # 找出直方圖中前三大值的索引
    top_indices = np.argsort(hist)[-3:]  # 找出前三大值的索引
    return top_indices

def plot_histogram(hist, output_path='histogram.png'):
    # 繪製直方圖並儲存
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.savefig(output_path)
    plt.close()
