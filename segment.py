import biosppy.signals.ecg as ecg
import numpy as np
def segment(path):
    # 讀取txt檔案
    

    ecg_signal = np.array(path)

    # 設定取樣頻率
    fs = 512

    # 檢測R波峰值
    rpeaks, = ecg.hamilton_segmenter(ecg_signal, sampling_rate=fs)
    rpeaks, = ecg.correct_rpeaks(ecg_signal, rpeaks, sampling_rate=fs, tol=0.05)
    
    return  rpeaks # 返回包含所有seg的列表

