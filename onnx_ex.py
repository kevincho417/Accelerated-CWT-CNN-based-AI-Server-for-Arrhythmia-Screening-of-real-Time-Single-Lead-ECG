import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import re
import numpy as np

# --- 常數設定 ---
MODEL_INPUT_LENGTH = 368  # 您的模型在訓練時使用的標準輸入長度
ONNX_FILENAME = 'model.onnx' # 使用正確的 .onnx 副檔名

# =============================================================================
# 1. 權重解析函數 (直接從您的訓練腳本中複製過來)
# =============================================================================
def parse_weights(file_path):
    print(f"正在從 {file_path} 解析權重...")
    weights = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"讀取權重檔案時發生錯誤: {e}")
        return weights

    pattern = re.compile(r"self\.(conv\d+)\s*\.weight\.data\s*=\s*(torch\.tensor\(.*?\))", re.DOTALL)
    matches = pattern.findall(content)
    
    if not matches:
        print("警告：在權重檔案中沒有找到任何匹配的權重。")
        return weights

    temp_globals = {'torch': torch}
    for layer_name, tensor_str in matches:
        try:
            exec(f"result = {tensor_str}.type(torch.FloatTensor)", temp_globals)
            weights[layer_name] = temp_globals['result']
        except Exception as e:
            print(f"無法解析層 {layer_name} 的權重：{e}")
            
    print(f"成功解析 {len(weights)} 個層的權重。")
    return weights

# =============================================================================
# 2. 完整的模型定義 (直接從您的訓練腳本中複製過來)
# =============================================================================
class ConvNet(nn.Module):
    def __init__(self, frozen_weights_path='my.txt'):
        super(ConvNet, self).__init__()
        
        # Part A: 可訓練層
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout(0.1)
        self.flatten = nn.Flatten()
        
        dummy_input = torch.randn(1, 1, MODEL_INPUT_LENGTH)
        flattened_size = self._get_conv_output_size(dummy_input)

        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, 5) # 假設是 5 分類

        # Part B: 載入並凍結 CWT 權重
        frozen_weights = parse_weights(frozen_weights_path)
        for i in range(5, 117):
            layer_name = f'conv{i}'
            conv_layer = nn.Conv1d(1, 2, kernel_size=61, padding=30, bias=False)
            if layer_name in frozen_weights:
                print(f"正在為 {layer_name} 載入並凍結權重。")
                conv_layer.weight.data = frozen_weights[layer_name]
                conv_layer.weight.requires_grad = False
            else:
                print(f"警告: 在 {frozen_weights_path} 中找不到 {layer_name} 的權重。此層將被凍結。")
                conv_layer.weight.requires_grad = False
            setattr(self, layer_name, conv_layer)

    def _get_conv_output_size(self, shape):
        x = self.pool1(F.relu(self.conv1(shape)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        return int(np.prod(x.size()))

    def forward(self, x):
        x_cnn = F.relu(self.conv1(x))
        x_cnn = self.dropout1(x_cnn)
        x_cnn = self.pool1(x_cnn)
        x_cnn = F.relu(self.conv2(x_cnn))
        x_cnn = self.dropout2(x_cnn)
        x_cnn = self.pool2(x_cnn)
        x_cnn = F.relu(self.conv3(x_cnn))
        x_cnn = self.dropout3(x_cnn)
        x_cnn = self.pool3(x_cnn)
        x_cnn = F.relu(self.conv4(x_cnn))
        x_cnn = self.pool4(x_cnn)
        x_cnn = self.dropout4(x_cnn)
        x_cnn = self.flatten(x_cnn)
        x_cnn = F.relu(self.fc1(x_cnn))
        output = self.fc2(x_cnn)
        return output

def export_model_to_onnx():
    """
    初始化 ConvNet 模型，載入權重，並將其匯出為 ONNX 格式。
    """
    print("Initializing model and preparing for ONNX export...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 建立模型實例
    #    模型在 __init__ 中會自動載入和凍結權重
    model = ConvNet(frozen_weights_path='my.txt')
    
    # 如果您已經訓練並儲存了可訓練部分的權重，請在這裡載入
    # trained_model_path = 'hybrid_cwt_cnn_model.pth' # 這是您之前訓練儲存的檔案
    # if os.path.exists(trained_model_path):
    #     print(f"正在從 {trained_model_path} 載入已訓練的權重...")
    #     model.load_state_dict(torch.load(trained_model_path))
    # else:
    #     print(f"警告：找不到已訓練的權重檔案 '{trained_model_path}'。將匯出未經訓練的模型。")

    model.to(device)
    model.eval() # 關鍵步驟：設定為評估模式
    
    print(f"Model initialized on device: {device}")

    # 2. 建立一個符合模型輸入形狀的虛擬張量
    #    形狀為 (batch_size, channels, signal_length)
    dummy_input = torch.randn(1, 1, MODEL_INPUT_LENGTH, device=device)
    
    print(f"Starting ONNX export to '{ONNX_FILENAME}'...")

    # 3. 將模型匯出為 ONNX 格式
    try:
        torch.onnx.export(
            model,
            dummy_input,
            ONNX_FILENAME,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input__0'],
            output_names=['output__0'],
            # 為 Triton 設定動態軸，允許批次大小和訊號長度可變
            dynamic_axes={
                'input__0': {0: 'batch_size', 2: 'signal_length'},
                'output__0': {0: 'batch_size'}
            }
        )
        print(f"✅ 模型已成功匯出至 '{ONNX_FILENAME}'!")
        
    except Exception as e:
        print(f"❌ ONNX 匯出時發生錯誤: {e}")
        print("請確保您已按照指示修復 Conda 環境。")


if __name__ == '__main__':
    export_model_to_onnx()