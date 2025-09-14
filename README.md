# MIT-BIH 心律不整分類模型：完整訓練與部屬流程

這份文件將引導您完成使用提供的 Python 程式碼，對 MIT-BIH 心律不整數據集進行數據預處理、模型訓練、評估及匯出為 ONNX 格式的完整流程。

## 專案介紹

本專案旨在建立一個能夠對心電圖 (ECG) 信號進行心律不整分類的深度學習模型。專案採用了混合式卷積神經網絡 (`HybridCWTCNN`)，該網絡結合了固定的連續小波變換 (CWT) 層來提取時頻特徵，以及一個可訓練的標準 CNN 來進行最終分類。

### 專案檔案結構

-   `data_preprosessing.py`: 負責將原始的 MIT-BIH 數據轉換為模型可用的 `.npy` 格式。
-   `my.txt`: 包含 CWT 卷積層的預定義權重，用於特徵提取。
-   `HybridCWTCNN.py`: 定義了主要的混合模型架構。
-   `training.py`: 主訓練腳本，負責載入數據、訓練模型、評估性能並儲存結果。
-   `onnx_ex.py`: 將訓練完成的 PyTorch 模型轉換為通用的 ONNX 格式，便於跨平台部署。
-   `static_CNN.py`: (備用模型) 提供一個不含 CWT 層的標準 CNN 模型架構。

## 環境建置

在開始之前，請確保您已安裝所有必要的 Python 套件。建議使用 `pip` 進行安裝：

```bash
pip install torch torchvision pandas numpy tqdm scikit-learn matplotlib seaborn
```
## 資料獲取
```
https://www.kaggle.com/datasets/mondejar/mitbih-database
```
