# Real-Time ECG Monitoring and AI Analysis System (即時心電圖監控與 AI 分析系統)

## 專案簡介

本專案是一個完整的即時心電圖 (ECG) 監控與分析解決方案。它能從硬體設備即時採集 ECG 訊號，透過後端伺服器進行訊號處理與 R 波偵測，並呼叫 NVIDIA Triton Inference Server 上的深度學習模型進行心搏分類，最終將結果即時呈現在網頁儀表板上，同時將分析結果儲存於 MongoDB 資料庫。

## 系統架構

系統由五個核心組件構成，協同運作完成從數據採集到視覺化呈現的完整流程。


---

## 核心組件詳解

### 1. 後端伺服器 (`app.py`)

-   **框架**: FastAPI
-   **功能**:
    -   **API 端點**:
        -   `POST /submit`: 接收由 `client_app.py` 傳來的原始 ECG 數據塊。
        -   `GET /`: 提供前端儀表板的 HTML 頁面。
        -   `POST /clear-table`: 清空 MongoDB 中的歷史紀錄。
    -   **即時通訊**:
        -   `WebSocket /ws`: 向所有前端客戶端廣播即時的 ECG 波形、R 波位置和預測結果。
        -   `WebSocket /ws1`: 定期從 MongoDB 讀取歷史數據並推送給前端。
    -   **數據處理**:
        -   管理數據緩衝區，確保數據流的連續性。
        -   呼叫 `filter_fir1` 進行訊號濾波。
        -   呼叫 `segment` 進行 R 波偵測與心搏切分。
    -   **AI 推論調度**: 呼叫 `trt_CNN.py` 中的 `CNN_processing2` 函式，將切分好的心搏樣本送至 Triton Server 進行推論。
    -   **資料庫操作**: 使用 `motor` 非同步驅動程式，將分析結果（波形、預測、時間戳）存入 MongoDB。

### 2. 前端儀表板 (`data_display.html`)

-   **技術**: HTML, JavaScript, CSS, [Plotly.js](https://plotly.com/javascript/)
-   **功能**:
    -   **即時圖表**: 透過 `/ws` WebSocket 接收數據，使用 Plotly.js 繪製一個可平滑滾動的 ECG 波形圖。
    -   **狀態顯示**: 顯示連線狀態、最新預測結果和心搏總數等即時資訊。
    -   **歷史紀錄**: 透過 `/ws1` WebSocket 接收數據，將歷史預測結果動態呈現在表格中。
    -   **互動功能**:
        -   點擊「查看波形」按鈕可彈出一個視窗，顯示特定心搏的詳細波形。
        -   點擊「清空歷史紀錄」按鈕可觸發後端的數據清除功能。

### 3. 數據採集客戶端 (`client_app.py`)

-   **功能**:
    -   使用 `pyserial` 函式庫連接指定的序列埠（COM Port），讀取 ECG 設備數據。
    -   將讀取的數據收集到一個緩衝區，當數量達到 `SAMPLES_PER_CHUNK` (512) 時，打包成 JSON 陣列。
    -   使用 `requests` 函式庫透過 HTTP POST 請求將數據批次傳送到後端伺服器的 `/submit` 端點。
    -   包含完善的錯誤處理機制，應對序列埠連接失敗、數據解碼錯誤和網路請求異常等情況。

### 4. Triton 推論客戶端 (`trt_CNN.py`)

-   **功能**:
    -   作為 NVIDIA Triton Inference Server 的 gRPC 客戶端。
    -   定義了目標模型名稱 (`CWT_CNN`)、輸入 (`input__0`) 與輸出 (`output__0`) 的格式。
    -   `CNN_processing` 函式接收多個心搏訊號段（`seg_list`），將它們轉換為符合模型輸入的格式 (`[1, 1, 368]`, FP32)。
    -   以**非同步**方式將請求發送到 Triton Server，並透過回呼函式處理返回的結果。
    -   解析模型的輸出，使用 `np.argmax` 找到最大機率的索引，並將其對應到預設的標籤 (`['F','Q' , 'N','V', 'S']`)。

---

## 環境設置與執行指南

### 前置需求

1.  **Python**: 3.8+
2.  **MongoDB**: 一個正在運行的 MongoDB 實例。
3.  **NVIDIA Triton Inference Server**:
    -   已安裝並正在運行。
    -   已成功部署名為 `CWT_CNN` 的模型，且模型的輸入/輸出名稱與 `trt_CNN.py` 中設定的一致。
4.  **ECG 硬體設備**: 一個能透過序列埠輸出數據的 ECG 設備。

### 安裝步驟

1.  **複製專案**:
    ```bash
    git clone <your-repository-url>
    cd <your-project-directory>
    ```

2.  **建立虛擬環境** (建議):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **安裝 Python 依賴套件**:
    ```bash
    pip install fastapi "uvicorn[standard]" motor numpy pyserial requests tritonclient[grpc]
    ```

### 組態設定

在執行前，請根據您的環境修改以下檔案中的設定：

1.  **`app.py`**:
    -   `MONGO_DETAILS`: 修改為您的 MongoDB 連線位址。

2.  **`client_app.py`**:
    -   `SERVER_URL`: 修改為 `app.py` 伺服器的 IP 位址與埠號。
    -   `SERIAL_PORT`: 修改為您的 ECG 設備所連接的正確 COM Port 名稱。
    -   `BAUD_RATE`: 確保鮑率與您的設備設定相符。

3.  **`trt_CNN.py`**:
    -   `--url` (或 `default` 值): 修改為您的 Triton Server 的 IP 位址與 gRPC 埠號 (預設 8001)。

### 執行順序

請務必按以下順序啟動各個服務：

1.  **啟動 MongoDB 資料庫**。

2.  **啟動 NVIDIA Triton Inference Server** 並確認 `CWT_CNN` 模型已載入成功。

3.  **啟動後端 FastAPI 伺服器**:
    在專案根目錄下執行：
    ```bash
    uvicorn app:app --host 0.0.0.0 --port 7000 --reload
    ```

4.  **啟動數據採集客戶端**:
    ```bash
    python client_app.py
    ```

5.  **打開前端儀表板**:
    在瀏覽器中開啟 `http://<您的伺服器IP>:7000`。您應該能看到即時更新的 ECG 波形與分析結果。
