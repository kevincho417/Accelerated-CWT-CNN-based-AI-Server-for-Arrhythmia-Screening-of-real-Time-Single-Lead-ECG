import time
import json
import serial  # 用於讀取序列埠 (USB Port)
import requests # 用於發送 HTTP 請求

# --- 您需要根據實際情況修改的設定 ---

# 您的 FastAPI 伺服器位址
SERVER_URL = "http://192.168.50.28:7000/submit" 

# ECG 裝置在您電腦上的 COM Port 名稱
# Windows 上可能是 'COM3', 'COM4' 等
# Linux 上可能是 '/dev/ttyUSB0', '/dev/ttyACM0' 等
# 您需要在裝置管理員中找到正確的 Port 名稱
SERIAL_PORT = 'COM25'  # <--- 請修改成您裝置的 Port

# 序列埠的鮑率 (Baud Rate)，需要與您的裝置設定相符
BAUD_RATE = 115200

# 每次收集多少筆數據後再一次性發送到伺服器
SAMPLES_PER_CHUNK = 512

# --- 主程式 ---

def main():
    """
    主函式，負責連接 USB 裝置並將資料傳送到伺服器。
    """
    print(f"正在嘗試連接到序列埠: {SERIAL_PORT}...")
    
    try:
        # 建立序列埠連線
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print("成功連接到序列埠。")
    except serial.SerialException as e:
        print(f"錯誤：無法開啟序列埠 {SERIAL_PORT}。")
        print(f"請檢查：")
        print(f"1. 裝置是否已正確連接？")
        print(f"2. Port 名稱 ('{SERIAL_PORT}') 是否正確？")
        print(f"3. 是否有其他程式正在使用此 Port？")
        print(f"詳細錯誤訊息: {e}")
        return

    data_buffer = []

    try:
        while True:
            # 從序列埠讀取一行資料
            line = ser.readline()
            
            if line:
                try:
                    # =========================== 修正 4：處理解碼錯誤 ===========================
                    # 在解碼時加入 errors='ignore'，讓程式忽略無法解碼的位元組
                    decoded_line = line.decode('utf-8', errors='ignore').strip()
                    # ========================================================================
                    
                    # 如果解碼後是空字串，則跳過此次迴圈
                    if not decoded_line:
                        continue

                    # 新的解析邏輯：尋找 'B' 字元，並提取其後面的數值
                    parts = decoded_line.replace('\t', ' ').split()
                    
                    try:
                        # 尋找 'B' 在列表中的索引位置
                        b_index = parts.index('B')
                        
                        # 檢查 'B' 是否是列表中的最後一個元素
                        if b_index < len(parts) - 1:
                            # 取得 'B' 後面的部分
                            value_str = parts[b_index + 1]
                            # 嘗試將其轉換為浮點數
                            data_point = float(value_str)
                            data_buffer.append(data_point)
                        else:
                            print(f"警告：在資料行 '{decoded_line}' 中, 'B' 後面沒有數值。")
                    
                    except ValueError:
                        # 如果 .index('B') 找不到 'B'，或 float() 轉換失敗，都會觸發此例外
                        print(f"警告：無法從資料行 '{decoded_line}' 中解析出 'B' 後的數值。")

                except Exception as e:
                    # 捕捉其他潛在的錯誤
                    print(f"處理資料時發生未預期的錯誤: {e}")
                    continue

            # 當 buffer 中的資料量達到設定的 chunk 大小
            if len(data_buffer) >= SAMPLES_PER_CHUNK:
                print(f"已收集 {len(data_buffer)} 筆資料，準備發送到伺服器...")
                
                try:
                    # 將資料透過 HTTP POST 請求以 JSON 格式發送到伺服器
                    response = requests.post(SERVER_URL, json=data_buffer, timeout=5)
                    
                    # 檢查伺服器的回應
                    if response.status_code == 200:
                        print("成功發送資料到伺服器。")
                    else:
                        print(f"錯誤：伺服器回應狀態碼 {response.status_code}")
                        print(f"回應內容: {response.text}")

                    # 清空 buffer 以便收集下一批資料
                    data_buffer = []

                except requests.exceptions.RequestException as e:
                    print(f"錯誤：無法連線到伺服器 {SERVER_URL}。請檢查網路連線與伺服器狀態。")
                    print(f"詳細錯誤訊息: {e}")
                    # 發生網路錯誤時，不清空 buffer，稍後重試
                    time.sleep(2) # 等待 2 秒後重試

    except KeyboardInterrupt:
        print("\n程式已由使用者手動中斷。")
    except Exception as e:
        print(f"發生未預期的錯誤: {e}")
    finally:
        # 確保在程式結束時關閉序列埠連線
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("已關閉序列埠連線。")


if __name__ == "__main__":
    main()

