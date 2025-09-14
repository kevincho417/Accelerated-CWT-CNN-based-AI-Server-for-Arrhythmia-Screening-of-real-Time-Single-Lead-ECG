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

# Demo video


https://github.com/user-attachments/assets/73958386-3d68-4f42-95ba-b1a6f683f9d3
## Demo 影片流程詳解

影片中的操作流程可以分為以下幾個關鍵階段：

---

### **第一階段：系統初始狀態 (0:00 - 0:08)**

-   **畫面呈現**:
    -   瀏覽器開啟了系統的前端儀表板 (`data_display.html`)。
    -   左側的「即時心電圖波形」圖表為空白。
    -   下方的「即時狀態」顯示為「連線中...」，「最新預測」為 `-`，「心搏總數」為 `0`。
    -   右側的「歷史紀錄」表格為空。
-   **技術背景**:
    -   此時，前端頁面已成功載入，並透過 JavaScript 建立好了與後端 `app.py` 伺服器的兩個 WebSocket 連線 (`/ws` 用於即時數據，`/ws1` 用於歷史紀錄)。
    -   「連線狀態」顯示為「已連線」表示 WebSocket 握手成功。
    -   系統正處於待命狀態，等待 `client_app.py` 開始傳送 ECG 數據。

---

### **第二階段：數據接收與即時分析 (0:09 - 0:37)**

-   **畫面呈現**:
    -   **0:09**: ECG 波形首次出現在左側圖表中，並開始向左平滑滾動。
    -   **0:10**: 「即時狀態」面板開始更新。「最新預測」顯示出第一個 AI 模型預測結果（如 `N`），「心搏總數」開始計數。
    -   **0:10**: 「歷史紀錄」表格中幾乎同時出現了第一筆分析紀錄，包含時間戳和預測結果。
    -   隨著時間推移，圖表持續更新，狀態面板和歷史紀錄不斷有新的數據加入，系統進入了連續監控模式。
    -   **0:34**: 影片中出現了一段明顯的雜訊干擾，展示了系統處理真實世界非理想訊號的能力。
-   **技術背景**:
    1.  **數據採集**: `client_app.py` 已啟動，它從序列埠 (`COM25`) 讀取 ECG 數據，每收集 512 筆數據點就打包成一個 JSON 陣列，透過 HTTP POST 請求發送到 `app.py` 的 `/submit` 端點。
    2.  **後端處理**: `app.py` 伺服器接收到數據後，執行一連串操作：
        -   將數據進行數位濾波。
        -   偵測 R 波峰 (心搏的關鍵特徵)，並以每個 R 波峰為中心，前後截取出 368 個點的波形片段。
        -   呼叫 `trt_CNN.py` 模組，將這些波形片段發送到 NVIDIA Triton Server 進行 AI 推論，獲得預測標籤 (如 `N`, `V` 等)。
    3.  **數據廣播**:
        -   後端透過 `/ws` WebSocket，將**濾波後的連續波形**和**預測結果**廣播給前端。前端的 JavaScript 接收到數據後，更新 Plotly.js 圖表，實現波形滾動效果。
        -   後端將每一筆分析結果（包含波形片段、預測標籤、時間戳）存入 MongoDB 資料庫。
        -   後端的 `/ws1` WebSocket 連線會定期查詢 MongoDB，並將最新的歷史紀錄推送給前端，前端動態更新右側的表格。

---

### **第三階段：互動式波形詳圖查詢 (0:38 - 0:42)**

-   **畫面呈現**:
    -   使用者在「歷史紀錄」表格中點擊了其中一筆紀錄旁的「查看波形」按鈕。
    -   一個彈出式視窗 (Modal) 出現，裡面使用 Plotly.js 精確繪製了該次心搏的 368 點詳細波形圖。視窗標題顯示了該次心搏的預測結果和發生時間。
    -   使用者可以關閉彈出視窗，返回主介面。
-   **技術背景**:
    -   「查看波形」按鈕在前端被點擊時，觸發了 `showWaveformModal` JavaScript 函式。
    -   這個函式使用了當初透過 `/ws1` WebSocket 傳來並儲存在前端的 `seg_data` (368 點的波形數據)，在彈出視窗內的 `modal-chart` 容器中繪製出詳細圖形。
    -   這個功能允許使用者對任何一次被 AI 標記的心搏進行視覺化確認，非常有利於數據的複查與分析。

---

### **第四階段：歷史紀錄管理 (0:43 - 0:49)**

-   **畫面呈現**:
    -   使用者點擊了頁面右上角的紅色「清空歷史紀錄」按鈕。
    -   瀏覽器彈出一個標準的確認對話框，詢問是否確定要刪除。
    -   使用者確認後，瀏覽器顯示一個提示框，告知成功刪除的紀錄筆數。
    -   右側的「歷史紀錄」表格被瞬間清空。
-   **技術背景**:
    -   點擊按鈕觸發了前端的 `fetch` API，向後端 `app.py` 的 `/clear-table` 端點發送一個 HTTP POST 請求。
    -   後端 `app.py` 收到此請求後，執行對 MongoDB 資料庫的 `delete_many({})` 操作，即刪除 `ecg_results` 集合中的所有文件。
    -   刪除成功後，後端回傳一個包含刪除數量的 JSON 確認訊息。
    -   前端接收到成功的訊息後，透過 JavaScript 將表格的 `innerHTML` 設為空，完成介面的更新。

# 伺服器回應紀錄範例

以下為伺服器處理請求並將資料傳入進行Rpeak分析後，進行推論，並將結果傳入 MongoDB 的範例輸出：


<img width="564" height="173" alt="server response" src="https://github.com/user-attachments/assets/6c583d5a-5192-44f9-b927-13b8c0b1c4b8" />

## 結論

此 Demo 影片完整展示了一個功能齊全的即時監控系統。透過 FastAPI、WebSocket、Plotly.js、MongoDB 和 Triton Inference Server 等技術的整合，成功打造了一個從硬體數據採集、後端 AI 分析、數據持久化到前端即時視覺化與互動的閉環解決方案。

