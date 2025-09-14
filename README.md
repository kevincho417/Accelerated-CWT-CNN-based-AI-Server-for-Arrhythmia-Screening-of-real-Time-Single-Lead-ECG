# 即時心電圖 (ECG) 監控儀表板運行指南

這是一個基於 FastAPI 的即時心電圖 (ECG) 監控應用程式。它透過 WebSocket 接收即時 ECG 數據，連接到 Triton Inference Server 進行心搏分類，並將結果儲存到 MongoDB 中，同時在網頁儀表板上視覺化波形與分析結果。

## 系統架構

本系統由以下幾個核心組件構成：

1.  **前端 (data_display.html)**：使用者介面，運行在瀏覽器中。透過 WebSocket 與後端進行雙向通訊，以顯示即時波形和歷史數據。
2.  **後端 (app.py)**：基於 FastAPI 的 Python 伺服器，是整個系統的中樞。它負責：
    * 提供前端網頁。
    * 接收來自感測器的 HTTP POST 請求。
    * 管理 WebSocket 連線，向所有客戶端廣播即時數據。
    * 對 ECG 數據進行預處理（濾波、R 波偵測、訊號切分）。
    * 與 Triton Inference Server 通訊，執行模型推論。
    * 將分析結果存入 MongoDB 資料庫。
3.  **Triton Inference Server**：一個獨立的推論伺服器，用於託管並執行 `CWT_CNN` 模型，對切分後的心搏訊號進行分類。
4.  **MongoDB**：一個 NoSQL 資料庫，用於儲存每一次心搏的波形數據、預測結果和時間戳。

## 前置準備 (Prerequisites)

在運行此應用程式之前，請確保以下環境和服務已經準備就緒：

1.  **Python 環境**:
    * 建議使用 Python 3.8 或更高版本。

2.  **MongoDB 資料庫**:
    * 確保 MongoDB 正在運行。
    * 應用程式預設會嘗試連接到 `mongodb://192.168.50.28:27017`。如有需要，請修改 `app.py` 中的 `MONGO_DETAILS` 變數。

3.  **NVIDIA Triton Inference Server**:
    * 確保 Triton 伺服器正在運行。
    * 伺服器上必須已經成功載入名為 `CWT_CNN` 的模型。
    * 模型輸入名稱應為 `input__0`，輸出名稱為 `output__0`。
    * 應用程式預設會嘗試連接到 `192.168.50.28:8001`。如有需要，請修改 `trt_CNN.py` 中的 `--url` 預設參數。

4.  **專案檔案**:
    * 確保以下檔案位於同一個專案目錄下：
        * `app.py` (主應用程式)
        * `trt_CNN.py` (Triton 客戶端)
        * `data_display.html` (前端頁面)
        * `segment.py` (R 波偵測模組，**需自行準備**)
        * `tfa_morlet_112m.py` (訊號濾波模組，**需自行準備**)

## 安裝步驟

1.  **建立虛擬環境 (建議)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

2.  **安裝 Python 依賴套件**
    您可以將以下內容儲存為 `requirements.txt` 檔案，然後執行 `pip install -r requirements.txt`。

    **requirements.txt:**
    ```
    fastapi
    uvicorn[standard]
    numpy
    motor
    tritonclient[grpc]
    ```

    **安裝指令:**
    ```bash
    pip install -r requirements.txt
    ```

## 運行應用程式

1.  **啟動 FastAPI 伺服器**
    在您的終端機中，切換到專案目錄，並執行以下指令：
    ```bash
    uvicorn app:app --host 0.0.0.0 --port 7000 --reload
    ```
    * `--host 0.0.0.0`: 使伺服器可以被區域網路中的其他裝置訪問。
    * `--port 7000`: 指定伺服器運行的埠號。
    * `--reload`: 開發模式，當程式碼變動時會自動重啟伺服器。

2.  **訪問儀表板**
    伺服器成功啟動後，在您的瀏覽器中開啟以下網址：
    `http://<YOUR_SERVER_IP>:7000`
    (如果是在本機運行，可以使用 `http://127.0.0.1:7000` 或 `http://localhost:7000`)

3.  **發送 ECG 數據**
    儀表板本身不產生數據。您需要另一個程式或裝置，將 ECG 數據以 HTTP POST 請求的方式發送到以下端點：
    * **URL**: `http://<YOUR_SERVER_IP>:7000/submit`
    * **Method**: `POST`
    * **Body (JSON)**: 一個包含 ECG 原始數據點的陣列，例如 `[20, 22, 21, ..., 25]`。

    當後端收到數據後，會進行處理，並透過 WebSocket 將結果即時更新到所有已連接的儀表板頁面上。

## 檔案功能說明

* `app.py`:
    * **核心邏輯**：接收 HTTP 請求，管理 WebSocket 連線。
    * **數據流**：調用 `tfa_morlet_112m` 進行濾波，`segment` 進行 R 波偵測，再將切分好的心搏數據傳遞給 `trt_CNN` 進行推論。
    * **資料庫互動**：使用 `motor` 異步函式庫將結果寫入 MongoDB。

* `trt_CNN.py`:
    * **Triton 客戶端**：封裝了與 Triton Inference Server 的 gRPC 通訊邏輯。
    * `CNN_processing` **函式**：接收一個或多個 ECG 心搏片段，以非同步方式發送推論請求，並回傳模型的預測標籤列表 (`['F', 'Q', 'N', 'V', 'S']`)。

* `data_display.html`:
    * **使用者介面**：使用 HTML, CSS 和 JavaScript 建立的單頁應用程式。
    * **即時繪圖**：使用 `Plotly.js` 繪製即時 ECG 波形。
    * **WebSocket 通訊**：
        * `/ws`：接收即時波形、R 波位置和預測結果，用於更新主圖表。
        * `/ws1`：定期從後端獲取歷史紀錄，並更新歷史表格。
