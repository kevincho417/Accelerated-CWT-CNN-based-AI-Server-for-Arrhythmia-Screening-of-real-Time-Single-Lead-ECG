import serial
import time
import threading
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- 請在此設定您的 COM Port ---
SERIAL_PORT = 'COM25'
BAUD_RATE = 115200
# ---------------------------------

# --- 繪圖設定 ---
# 圖表上最多顯示的資料點數量
MAX_PLOT_POINTS = 200
# -----------------

# 使用 deque 作為固定長度的資料緩衝區，效能較佳
data_queue = deque(maxlen=MAX_PLOT_POINTS)
# 用於控制子執行緒結束的旗標
is_running = True

def parse_line(line_str: str):
    """
    穩健地解析一行字串，尋找並回傳第一個有效的數值。
    """
    # 將 Tab 字元替換為空格，然後以空格分割
    parts = line_str.replace('\t', ' ').split()
    for part in parts:
        try:
            # 嘗試將部分轉換為浮點數並回傳
            return float(part)
        except ValueError:
            # 如果轉換失敗，表示此部分不是數字，繼續尋找下一個
            continue
    # 如果整行都找不到有效數值，則回傳 None
    return None

def serial_reader_thread():
    """
    在一個獨立的執行緒中持續讀取序列埠資料。
    這樣做可以避免讀取資料時（blocking I/O）凍結圖形介面。
    """
    global is_running
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            print(f"成功連接到序列埠 {SERIAL_PORT}。開始讀取資料...")
            while is_running:
                try:
                    line = ser.readline()
                    if line:
                        decoded_line = line.decode('utf-8').strip()
                        print(f"收到資料: {decoded_line}") # 仍然在終端機印出原始資料
                        
                        # 解析數值
                        value = parse_line(decoded_line)
                        if value is not None:
                            data_queue.append(value) # 將有效數值加入佇列
                except UnicodeDecodeError:
                     print(f"收到無法解碼的資料 (bytes): {line}")
            print("讀取執行緒已結束。")

    except serial.SerialException as e:
        print(f"\n錯誤：無法開啟序列埠 {SERIAL_PORT}。")
        print(f"詳細錯誤訊息: {e}")
        is_running = False # 通知主執行緒結束
    except Exception as e:
        print(f"\n讀取執行緒發生未預期的錯誤: {e}")
        is_running = False


def update_plot(frame, line, ax):
    """
    Matplotlib 動畫的更新函式，會被週期性呼叫。
    """
    # 設定線條的資料
    line.set_data(np.arange(len(data_queue)), data_queue)
    
    # 自動調整圖表的顯示範圍
    ax.relim()
    ax.autoscale_view()
    return line,

def main():
    """
    主函式，負責啟動執行緒與設定繪圖。
    """
    global is_running
    
    # 啟動讀取序列埠的子執行緒
    reader_thread = threading.Thread(target=serial_reader_thread)
    reader_thread.daemon = True # 設定為守護執行緒，主程式結束時會自動關閉
    reader_thread.start()

    # 等待一小段時間，確保執行緒有時間嘗試連線
    time.sleep(1.5) 
    
    # 如果子執行緒因為連線失敗而結束，主程式也跟著結束
    if not is_running:
        print("無法建立序列埠連線，程式即將結束。")
        return

    # --- 設定 Matplotlib 圖表 ---
    fig, ax = plt.subplots()
    # 初始化一條空的線
    line, = ax.plot([], [])
    
    # 設定圖表樣式
    ax.set_title("即時訊號監控")
    ax.set_xlabel("樣本點")
    ax.set_ylabel("訊號值")
    ax.grid(True)

    # 建立動畫
    # interval=50 表示每 50 毫秒更新一次圖表
    ani = animation.FuncAnimation(fig, update_plot, fargs=(line, ax), interval=50, blit=True)
    
    print("正在開啟繪圖視窗...")
    # 顯示圖表 (這會是一個 blocking call，直到視窗被關閉)
    plt.show()

    # 當繪圖視窗被關閉後，執行以下程式碼
    print("繪圖視窗已關閉，正在結束程式...")
    is_running = False # 通知子執行緒可以結束了
    reader_thread.join(timeout=2) # 等待子執行緒最多 2 秒
    print("程式已結束。")


if __name__ == '__main__':
    main()

