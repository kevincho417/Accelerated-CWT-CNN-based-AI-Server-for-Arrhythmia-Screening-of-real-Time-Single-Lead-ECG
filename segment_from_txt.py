import biosppy.signals.ecg as ecg
import numpy as np
def segment(path):
    # 讀取txt檔案
    with open(path, 'r') as f:
        # 讀取所有行，提取第一個數字
        numbers = []
        for line in f.readlines():
            num_list = line.split(',')  #拆分為數字列表
            first_num = int(num_list[0])  #提取第一個元素
            numbers.append(first_num)  #添加到數字列表中

    ecg_signal = np.array(numbers).astype(np.float32)

    # 設定取樣頻率
    fs = 250

    # 檢測R波峰值
    rpeaks, = ecg.hamilton_segmenter(ecg_signal, sampling_rate=fs)
    rpeaks, = ecg.correct_rpeaks(ecg_signal, rpeaks, sampling_rate=fs, tol=0.05)

    seg_list = []  # 創建一個空的列表

    for i in range(len(rpeaks)):
        seg_start = rpeaks[i] -184   # 設定每個R波峰值附近的心電圖訊號段的起點
        seg_end = rpeaks[i] + 184  # 設定每個R波峰值附近的心電圖訊號段的終點
        seg = ecg_signal[seg_start:seg_end]  # 切割心電圖訊號
        
        if len(seg) == 368:
            seg_list.append(seg)  # 添加seg到列表中
    
    return  seg_list # 返回包含所有seg的列表