import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- 設定 ---
# 包含所有 .csv 和 annotations.txt 檔案的來源資料夾
SOURCE_DATA_DIR = 'mitbih_raw_data'
# 輸出資料夾
OUTPUT_DIR = 'dataset'

# 心電圖訊號參數
SAMPLING_RATE = 360  # MIT-BIH 資料庫的標準取樣率為 360 Hz
SECONDS_BEFORE = 0.36
SECONDS_AFTER = 0.36

# 計算取樣點數
SAMPLES_BEFORE = int(SECONDS_BEFORE * SAMPLING_RATE) # 約 130
SAMPLES_AFTER = int(SECONDS_AFTER * SAMPLING_RATE)  # 約 130
TOTAL_SAMPLES = SAMPLES_BEFORE + 1 + SAMPLES_AFTER

# MIT-BIH 註釋到目標類別的對應關係
ANNOTATION_MAP = {
    'N': ['N', 'L', 'R', 'e', 'j'],
    'S': ['A', 'a', 'J', 'S'],
    'V': ['V', 'E'],
    'F': ['F'],
    'Q': ['/', 'f', 'Q']
}

# 將上述對應關係反轉，以便快速查找每個註釋屬於哪個目標類別
CLASS_MAP = {annotation: aami_class for aami_class, annotations in ANNOTATION_MAP.items() for annotation in annotations}


def preprocess_mitbih_record(record_name):
    """
    處理單一 MIT-BIH 記錄，切割心跳並將結果 (儲存路徑, 心跳數據) 回傳。
    這個函式將會由多個獨立的 CPU 核心平行執行。
    """
    signal_file = os.path.join(SOURCE_DATA_DIR, f"{record_name}.csv")
    annotation_file = os.path.join(SOURCE_DATA_DIR, f"{record_name}annotations.txt")

    try:
        # ============================ 核心修正部分 ============================
        # 為了處理欄位名稱可能存在的變異 (例如, 'MLII' vs ' MLII')，
        # 我們直接讀取整個 CSV，然後按位置選取第二欄 (索引為 1)，這通常是 MLII 導聯。
        df_signal = pd.read_csv(signal_file)
        # .iloc[:, 1] 選取所有列的第二欄
        signal = df_signal.iloc[:, 1].values
        # ====================================================================
        
        r_peaks = []
        annotations = []
        with open(annotation_file, 'r') as f:
            next(f) # 略過標頭
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        r_peak_sample = int(parts[1])
                        annotation_type = parts[2]
                        if annotation_type in CLASS_MAP:
                            r_peaks.append(r_peak_sample)
                            annotations.append(annotation_type)
                    except (ValueError, IndexError):
                        continue
    except FileNotFoundError:
        return [] # 如果找不到檔案，回傳空列表
    except Exception as e:
        print(f"處理記錄 {record_name} 時發生錯誤: {e}")
        return []

    # --- 2. 切割心跳 ---
    heartbeats_to_save = []
    file_counters = {class_name: 0 for class_name in ANNOTATION_MAP.keys()}

    for r_peak, ann in zip(r_peaks, annotations):
        start_idx = r_peak - SAMPLES_BEFORE
        end_idx = r_peak + SAMPLES_AFTER + 1
        
        if start_idx >= 0 and end_idx <= len(signal):
            heartbeat = signal[start_idx:end_idx]
            if len(heartbeat) == TOTAL_SAMPLES:
                target_class = CLASS_MAP[ann]
                file_counters[target_class] += 1
                file_number = file_counters[target_class]
                # 檔名現在包含記錄名稱以確保唯一性
                save_path = os.path.join(OUTPUT_DIR, target_class, f"{record_name}_{file_number}.npy")
                heartbeats_to_save.append((save_path, heartbeat))
                
    return heartbeats_to_save


def main():
    """
    主函式，掃描來源資料夾，並使用多核心處理所有記錄。
    """
    print(f"開始預處理資料集...")
    print(f"來源資料夾: '{SOURCE_DATA_DIR}'")
    print(f"目標資料夾: '{OUTPUT_DIR}'")

    # --- 1. 尋找所有記錄 ---
    csv_files = glob.glob(os.path.join(SOURCE_DATA_DIR, '*.csv'))
    record_names = []
    for csv_file in csv_files:
        record_name = os.path.basename(csv_file).replace('.csv', '')
        # 檢查對應的註釋檔案是否存在
        if os.path.exists(os.path.join(SOURCE_DATA_DIR, f"{record_name}annotations.txt")):
            record_names.append(record_name)

    if not record_names:
        print(f"錯誤：在 '{SOURCE_DATA_DIR}' 中找不到任何匹配的 (.csv, annotations.txt) 檔案對。")
        print("請確認您的檔案都放在正確的資料夾中。")
        return

    print(f"找到 {len(record_names)} 個記錄需要處理: {sorted(record_names)}")

    # --- 2. 建立輸出資料夾 ---
    for class_name in ANNOTATION_MAP.keys():
        class_path = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(class_path, exist_ok=True)

    # --- 3. 使用多核心處理所有記錄 ---
    num_workers = cpu_count()
    print(f"將使用 {num_workers} 個 CPU 核心進行平行處理...")

    all_heartbeats_to_save = []
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(preprocess_mitbih_record, record_names), total=len(record_names), desc="處理記錄進度"))
    
    # 將所有子程序的回傳結果合併到一個列表中
    for result_list in results:
        all_heartbeats_to_save.extend(result_list)
        
    if not all_heartbeats_to_save:
        print("警告：處理完成，但沒有產生任何有效的心跳數據。")
        return
        
    # --- 4. 循序寫入檔案 ---
    # 為了避免多核心同時寫入檔案系統的潛在問題，我們在收集完所有數據後，
    # 於主程序中進行寫入，這樣也能提供一個更清晰的進度條。
    print(f"\n總共切割出 {len(all_heartbeats_to_save)} 個心跳，開始寫入檔案...")
    for save_path, heartbeat in tqdm(all_heartbeats_to_save, desc="寫入檔案進度"):
        np.save(save_path, heartbeat)

    print("\n處理完成！")
    print(f" - 總共成功儲存了 {len(all_heartbeats_to_save)} 個心跳。")
    print(f"您整理好的資料集位於 '{OUTPUT_DIR}' 資料夾中。")

if __name__ == "__main__":
    main()

