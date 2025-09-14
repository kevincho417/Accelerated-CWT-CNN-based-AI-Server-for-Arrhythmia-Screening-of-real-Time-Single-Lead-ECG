# 即時心電圖 (ECG) 監控儀表板運行指南
## 這是一個基於 FastAPI 的即時心電圖 (ECG) 監控應用程式。它透過 WebSocket 接收即時 ECG 數據，連接到 Triton Inference Server 進行心搏分類，並將結果儲存到 MongoDB 中，同時在網頁儀表板上視覺化波形與分析結果。

## 系統架構
本系統由以下幾個核心組件構成：

### 前端 (data_display.html)：使用者介面，運行在瀏覽器中。透過 WebSocket 與後端進行雙向通訊，以顯示即時波形和歷史數據。

### 後端 (app.py)：基於 FastAPI 的 Python 伺服器，是整個系統的中樞。它負責：

提供前端網頁。

接收來自感測器的 HTTP POST 請求。

管理 WebSocket 連線，向所有客戶端廣播即時數據。

對 ECG 數據進行預處理（濾波、R 波偵測、訊號切分）。

與 Triton Inference Server 通訊，執行模型推論。

將分析結果存入 MongoDB 資料庫。

Triton Inference Server：一個獨立的推論伺服器，用於託管並執行 CWT_CNN 模型，對切分後的心搏訊號進行分類。

MongoDB：一個 NoSQL 資料庫，用於儲存每一次心搏的波形數據、預測結果和時間戳。

前置準備 (Prerequisites)
在運行此應用程式之前，請確保以下環境和服務已經準備就緒：

Python 環境:

建議使用 Python 3.8 或更高版本。

MongoDB 資料庫:

確保 MongoDB 正在運行。

應用程式預設會嘗試連接到 mongodb://192.168.50.28:27017。如有需要，請修改 app.py 中的 MONGO_DETAILS 變數。

NVIDIA Triton Inference Server:

確保 Triton 伺服器正在運行。

伺服器上必須已經成功載入名為 CWT_CNN 的模型。

模型輸入名稱應為 input__0，輸出名稱為 output__0。

應用程式預設會嘗試連接到 192.168.50.28:8001。如有需要，請修改 trt_CNN.py 中的 --url 預設參數。

專案檔案:

確保以下檔案位於同一個專案目錄下：

app.py (主應用程式)

trt_CNN.py (Triton 客戶端)

data_display.html (前端頁面)

segment.py (R 波偵測模組，需自行準備)

tfa_morlet_112m.py (訊號濾波模組，需自行準備)
