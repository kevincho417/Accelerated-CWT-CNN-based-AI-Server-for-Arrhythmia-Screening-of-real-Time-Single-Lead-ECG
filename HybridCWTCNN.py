import torch
from torch import nn
import torch.nn.functional as F
import re
import numpy as np # 新增 numpy 以便計算 flattened size

# =============================================================================
# 1. 解析 my.txt 以提取權重 (這部分不變)
# =============================================================================
def parse_weights(file_path):
    """
    解析 my.txt 檔案以提取卷積層的權重。
    """
    weights = {}
    # 嘗試不同的編碼格式來開啟檔案
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252']
    content = None
    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            print(f"成功以 {encoding} 編碼讀取檔案。")
            break
        except UnicodeDecodeError:
            print(f"以 {encoding} 編碼讀取失敗，嘗試下一個...")
    
    if content is None:
        raise ValueError("無法使用任何嘗試過的編碼格式讀取檔案。")

    # 使用正則表達式尋找所有權重張量
    # 注意：這個正則表達式假設了 my.txt 中權重定義的格式
    pattern = re.compile(r"self\.(conv\d+)\s*\.weight\.data\s*=\s*(torch\.tensor\(.*?\)\.type\(torch\.cuda\.FloatTensor\))", re.DOTALL)
    matches = pattern.findall(content)

    temp_globals = {'torch': torch}
    for match in matches:
        layer_name = match[0]
        tensor_str = match[1]
        
        try:
            # 使用 exec 來執行 torch.tensor(...) 字串，以建立張量物件
            # 注意：使用 exec/eval 可能存在安全風險，但在此情況下，我們信任 my.txt 的來源
            exec(f"result = {tensor_str}", temp_globals)
            weights[layer_name] = temp_globals['result']
        except Exception as e:
            print(f"無法解析層 {layer_name} 的權重：{e}")
            
    return weights

# =============================================================================
# 2. 修正後的 ConvNet 模型
# =============================================================================
class ConvNet(nn.Module):
    def __init__(self, frozen_weights):
        super(ConvNet, self).__init__()
        self._cuda = True
        self.img_select = torch.linspace(0, 367, 112, dtype=int)

        # ---------------------------------------------------------------------
        # Part A: 新增原始的可訓練層 (根據 static_CNN.py 推斷)
        # 這些層的 requires_grad 預設為 True，所以它們是可訓練的。
        # 注意：這裡的通道數 (in/out channels) 和大小是根據常見的 CNN 架構推斷的，
        # 您可能需要根據您的具體資料維度進行調整。
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
        
        # 為了計算 flattened size，我們需要一個虛擬輸入
        # 假設輸入維度是 (batch_size, 1, 112)，這是根據 self.img_select 推斷的
        dummy_input = torch.randn(1, 1, 112)
        flattened_size = self._get_conv_output_size(dummy_input)

        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, 2) # 假設是二分類問題，所以輸出是 2
        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        # Part B: 載入並凍結來自 my.txt 的層
        for i in range(5, 117):
            layer_name = f'conv{i}'
            conv_layer = nn.Conv1d(1, 2, kernel_size=61, padding=30, bias=False)
            
            if layer_name in frozen_weights:
                print(f"正在為 {layer_name} 載入凍結的權重。")
                conv_layer.weight.data = frozen_weights[layer_name]
                # 關鍵步驟：凍結權重，使其不可訓練
                conv_layer.weight.requires_grad = False
            else:
                # 如果在 my.txt 中找不到，我們也將其凍結，因為它們沒有預訓練的權重
                print(f"警告：在 my.txt 中找不到 {layer_name} 的權重。此層將被凍結且不進行訓練。")
                conv_layer.weight.requires_grad = False

            setattr(self, layer_name, conv_layer)
        # ---------------------------------------------------------------------

    def _get_conv_output_size(self, shape):
        x = self.pool1(self.conv1(shape))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.pool4(self.conv4(x))
        return int(np.prod(x.size()))

    def forward(self, x):
        # 這是您原始 static_CNN.py 中的前向傳播路徑
        # 它只使用了我們設定為可訓練的層 (conv1-4, fc1-2)
        x = x[:, self.img_select, :]
        
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.dropout3(x)
        x = self.pool3(x)

        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = self.dropout4(x)

        x = self.flatten(x)
        
        x = F.relu(self.fc1(x))
        # 在訓練時，通常 CrossEntropyLoss 會包含 Softmax，所以這裡可以先移除
        # x = F.softmax(self.fc2(x), dim=1) 
        x = self.fc2(x)

        # 注意：凍結的 conv5-116 層沒有在這個 forward 函數中使用。
        # 如果您需要使用它們，您需要修改上面的 forward 邏輯。

        return x

# =============================================================================
# 3. 訓練迴圈 (這部分不變)
# =============================================================================
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    """
    訓練模型的函數。
    """
    model.train()  # 將模型設定為訓練模式
    print("\n--- 開始訓練 ---")
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向傳播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向傳播和優化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'週期 [{epoch+1}/{num_epochs}], 步驟 [{i+1}/{len(train_loader)}], 損失: {loss.item():.4f}')
    print("--- 訓練完成 ---")


if __name__ == '__main__':
    # 解析 my.txt 中的權重
    frozen_weights = parse_weights('my.txt')

    # 初始化模型並載入凍結的權重
    model = ConvNet(frozen_weights)
    
    # 將模型移至 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 打印出模型中哪些參數是可訓練的
    print("\n--- 模型參數狀態 ---")
    for name, param in model.named_parameters():
        print(f"層: {name}, 是否可訓練: {param.requires_grad}")
    print("--------------------")

    # 定義損失函數和優化器
    # 現在 trainable_params 將不再是空的！
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    # 將 filter 物件轉換為 list，以檢查是否為空
    trainable_params_list = list(trainable_params)

    if not trainable_params_list:
        print("錯誤：沒有找到任何可訓練的參數。請檢查模型定義。")
    else:
        optimizer = torch.optim.Adam(trainable_params_list, lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # =============================================================================
        # ... 在此處準備您的資料集和 DataLoader ...
        #
        # 範例 (您需要用您的真實資料取代這裡的假資料):
        print("\n正在使用假資料進行範例訓練...")
        # 假設輸入資料維度是 [batch_size, 112, 1] -> 經過 forward 會變成 [batch_size, 1, 112]
        dummy_train_data = torch.randn(256, 112, 1) 
        dummy_train_labels = torch.randint(0, 2, (256,)) # 假設是2分類
        
        train_dataset = torch.utils.data.TensorDataset(dummy_train_data, dummy_train_labels)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=64,
                                                   shuffle=True)
        # =============================================================================

        # 訓練模型
        train_model(model, train_loader, criterion, optimizer, device, num_epochs=5)