專案工作流程總結：從混合式 PyTorch 模型到 Triton 部署與優化
本文件記錄了建立、訓練、部署及優化一個混合式卷積神經網路 (CNN) 的完整流程。專案的核心目標是將一個預先計算好的小波轉換 (CWT) 特徵層與一個可訓練的 CNN 分類器結合，並最終將其部署到 Triton Inference Server。

階段一：建立與偵錯混合式神經網路模型
此階段的目標是建立一個 PyTorch 模型，其中一部分權重是固定的 (來自 CWT)，另一部分是可訓練的。

需求分析：

輸入檔案:

static_CNN.py: 包含基礎 CNN 架構的 Python 腳本。

my.txt: 一個文字檔，以 PyTorch 張量 (Tensor) 的字串格式，定義了 conv5 到 conv116 層的固定權重。

核心任務: 修改 static_CNN.py，使其在初始化時自動從 my.txt 載入權重，並將這些層設定為不可訓練（凍結權重）。

初步實現:

撰寫了一個 parse_weights 函數，使用正則表達式 (regex) 讀取並解析 my.txt，將權重字串轉換為 PyTorch 張量。

修改了 ConvNet 模型的 __init__ 方法，使其在迴圈中動態建立 conv5 到 conv116 層，載入解析出的權重，並設定 param.requires_grad = False 來實現權重凍結。

問題與偵錯:

問題: 執行時遇到 ValueError: optimizer got an empty parameter list。

根本原因: 當時的模型版本中，所有的層都被凍結了，導致優化器 (Optimizer) 找不到任何可以訓練的參數。

解決方案: 將 static_CNN.py 中原本可訓練的層（如 conv1-4 和 fc1-2）重新整合回 ConvNet 模型中，確保模型中同時包含可訓練與不可訓練的部分。

階段二：整合訓練腳本與 ONNX 匯出
此階段的目標是將模型定義與訓練流程結合，並將訓練好的模型轉換為 ONNX 格式以便部署。

整合訓練流程:

輸入檔案:

HybridCWTCNN.py: 包含已修正的混合式模型定義。

training.py: 包含資料載入 (NpySignalDataset)、訓練迴圈、評估及繪圖的完整腳本。

任務: 將兩者合併為一個獨立、可執行的完整訓練腳本。

實現: 建立了一個新的 Python 腳本，其中包含了 HybridCWTCNN.py 的模型類別和 training.py 的所有輔助函數與主執行邏輯。

匯出為 ONNX 格式:

目標: 將訓練好的 PyTorch 模型 (.pth) 轉換為標準的 .onnx 格式，以供 Triton Server 使用。

挑戰與偵錯:

問題 1: google.protobuf.runtime_version.VersionError。

原因: Conda 環境中 onnx 和 protobuf 函式庫版本不相容，是典型的環境衝突。

解決方案: 提供了一系列 pip uninstall 和 pip install 指令來強制重新安裝，以解決版本依賴問題。

問題 2: ModuleNotFoundError: No module named 'onnx'。

原因: 即使 pip 顯示已安裝，但 Python 直譯器卻找不到。這通常是 Conda 環境路徑損壞或 pip 與 python 指令不匹配所致。

解決方案: 提供了更強制的安裝指令 python -m pip install --force-reinstall ...，並提供了重建 Conda 環境作為最終手段。

階段三：Triton Inference Server 部署與配置
此階段的目標是讓 ONNX 模型成功在 Triton Server 上載入並提供服務。

Triton 模型庫配置:

檔案結構: 建立了符合 Triton 要求的模型庫結構。

設定檔: 提供了 config.pbtxt 檔案來描述模型的輸入、輸出規格。

問題與偵錯:

問題: Triton Server 啟動時回報 UNAVAILABLE: Invalid argument: model expects shape [-1,5] but configuration specifies shape [1,5]。

原因: ONNX 模型在匯出時已正確設定了動態軸 (Dynamic Axes)，允許批次大小 (batch_size) 可變（表示為 -1）。然而，config.pbtxt 中卻將批次大小寫死為 1，導致模型與設定不匹配。

解決方案: 修改 config.pbtxt，將輸入和輸出的 dims 中的 1 改為 -1，以正確反映模型的動態批次處理能力。

程式碼片段

# 修正後的 config.pbtxt 關鍵部分
output [
  {
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ -1, 5 ]  # -1 代表此維度為動態
  }
]
階段四：訓練流程優化與自動化
此階段的目標是改進訓練腳本，使其功能更完善，並解決資料不平衡問題。

結果自動化與視覺化:

需求: 自動儲存評估結果，並將混淆矩陣以更直觀的百分比形式呈現。

實現:

腳本現在會自動建立一個以時間戳命名的 results/ 資料夾。

評估後的準確率、分類報告 (classification_report)、原始預測結果 (.npz) 都會自動儲存。

修改了繪圖函數，使混淆矩陣的數值正規化為百分比，並將圖片自動儲存至結果資料夾。

處理資料不平衡:

需求 1: 使用相同數量的樣本進行訓練與評估，以避免模型偏向多數類別。

初步實現: 引入了資料下採樣 (Downsampling) 邏輯。腳本會自動找到樣本數最少的類別，並將所有其他類別的樣本數隨機減少至與其相同。

需求 2 (優化): 下採樣策略過於激進，會損失太多資料。改為使用最少類別樣本數的三倍作為所有類別取樣的上限。

最終實現:

當 SAMPLES_PER_CLASS 設為 None 時，腳本會計算出 sample_limit = min_class_count * 3。

對於每個類別，實際取樣的數量為 min(該類別總樣本數, sample_limit)。

這個「帶有上限的部分平衡」策略，在緩解資料不平衡和最大化資料利用率之間取得了更好的平衡。
