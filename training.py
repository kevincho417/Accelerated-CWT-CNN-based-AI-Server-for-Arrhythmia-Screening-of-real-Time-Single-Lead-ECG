import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset, Subset
import os
import numpy as np
from tqdm import tqdm
import re
import datetime
from collections import Counter

# 引入繪圖和評估工具
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
from itertools import cycle

# --- 全域設定 ---
DATA_DIR = 'dataset'
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
MODEL_INPUT_LENGTH = 368
# --- 新增設定：設定每個類別要使用的樣本數 ---
# 設定為 None 可自動使用數量最少的類別的樣本數
SAMPLES_PER_CLASS = None 

# --- 建立結果儲存資料夾 ---
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
RESULTS_DIR = f'results_{TIMESTAMP}'
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
print(f"所有結果將儲存於: {RESULTS_DIR}")


# =============================================================================
# 1. 權重解析函數 (維持不變)
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
# 2. 混合式 CWT-CNN 模型 (維持不變)
# =============================================================================
class ConvNet(nn.Module):
    def __init__(self, frozen_weights_path='my.txt'):
        super(ConvNet, self).__init__()
        
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
        self.fc2 = nn.Linear(512, 5)

        frozen_weights = parse_weights(frozen_weights_path)
        for i in range(5, 117):
            layer_name = f'conv{i}'
            conv_layer = nn.Conv1d(1, 2, kernel_size=61, padding=30, bias=False)
            if layer_name in frozen_weights:
                conv_layer.weight.data = frozen_weights[layer_name]
                conv_layer.weight.requires_grad = False
            else:
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

# =============================================================================
# 3. 資料集類別 (維持不變)
# =============================================================================
class NpySignalDataset(Dataset):
    def __init__(self, root_dir, target_length):
        self.root_dir = root_dir
        self.target_length = target_length
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.file_paths = []
        self.labels = []
        for cls_name in self.classes:
            class_dir = os.path.join(root_dir, cls_name)
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.npy'):
                    self.file_paths.append(os.path.join(class_dir, file_name))
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        signal = np.load(file_path).astype(np.float32)
        label = self.labels[idx]

        if len(signal) > self.target_length:
            signal = signal[:self.target_length]
        elif len(signal) < self.target_length:
            padding = self.target_length - len(signal)
            signal = np.pad(signal, (0, padding), 'constant')
        
        signal = np.expand_dims(signal, axis=0)
        return torch.from_numpy(signal), torch.tensor(label, dtype=torch.long)

# =============================================================================
# 4. 訓練與評估函數 (維持不變)
# =============================================================================
def train_model(model, train_loader, criterion, optimizer, device, num_epochs):
    model.train()
    history = {'loss': [], 'accuracy': []}
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100 * correct / total:.2f}%'})
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        print(f"Epoch {epoch+1} - Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    return history

def evaluate_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

# =============================================================================
# 5. 繪圖與儲存函數 (維持不變)
# =============================================================================
def plot_and_save_confusion_matrix_percentage(labels, preds, class_names, output_dir):
    cm = confusion_matrix(labels, preds)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_percentage, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (in Percentage)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    save_path = os.path.join(output_dir, "confusion_matrix_percentage.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"百分比混淆矩陣圖已儲存至: {save_path}")
    plt.close()

def plot_and_save_multiclass_roc(labels, probs, class_names, output_dir):
    num_classes = len(class_names)
    bin_labels = label_binarize(labels, classes=range(num_classes))
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(bin_labels[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    plt.figure(figsize=(12, 9))
    plt.plot(fpr["macro"], tpr["macro"],
             label=f'Macro-average ROC curve (AUC = {roc_auc["macro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)
    colors = cycle(plt.cm.get_cmap('tab10').colors)
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {class_names[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Multi-class Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right"); plt.grid(True)
    save_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC 曲線圖已儲存至: {save_path}")
    plt.close()

# =============================================================================
# 6. 主執行區塊 (已修改)
# =============================================================================
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"將使用裝置: {device}")

    print(f"從 '{DATA_DIR}' 載入資料...")
    if not os.path.exists(DATA_DIR):
        print(f"錯誤: 資料夾 '{DATA_DIR}' 不存在。請先執行資料預處理腳本。")
        exit()
        
    # 載入完整資料集
    full_dataset = NpySignalDataset(root_dir=DATA_DIR, target_length=MODEL_INPUT_LENGTH)
    print(f"完整資料集載入完成。總共有 {len(full_dataset)} 個樣本。")
    print(f"原始類別分佈: {Counter(full_dataset.labels)}")
    
    # --- 新增：資料下採樣 (Downsampling) 邏輯 ---
    labels = np.array(full_dataset.labels)
    class_counts = Counter(labels)
    
    if SAMPLES_PER_CLASS is None:
        # 自動尋找樣本數最少的類別
        min_samples = min(class_counts.values())
        print(f"自動偵測到最少樣本數為: {min_samples}。將以此數量為基準進行下採樣。")
    else:
        min_samples = SAMPLES_PER_CLASS
        print(f"已手動設定每個類別的樣本數為: {min_samples}。")

    balanced_indices = []
    for class_idx in sorted(class_counts.keys()):
        # 取得該類別所有樣本的索引
        class_indices = np.where(labels == class_idx)[0]
        # 檢查是否有足夠的樣本
        if len(class_indices) < min_samples:
            print(f"警告：類別 {class_idx} 只有 {len(class_indices)} 個樣本，少於目標的 {min_samples}。將使用所有可用樣本。")
            chosen_indices = class_indices
        else:
            # 從中隨機選取 min_samples 個樣本
            chosen_indices = np.random.choice(class_indices, size=min_samples, replace=False)
        balanced_indices.extend(chosen_indices)
    
    np.random.shuffle(balanced_indices) # 打亂索引順序
    
    # 根據平衡後的索引建立一個新的 Subset
    balanced_dataset = Subset(full_dataset, balanced_indices)
    print(f"下採樣完成。平衡後的資料集總共有 {len(balanced_dataset)} 個樣本。")
    
    # --- 使用平衡後的資料集進行後續操作 ---
    train_size = int(0.8 * len(balanced_dataset))
    test_size = len(balanced_dataset) - train_size
    train_dataset, test_dataset = random_split(balanced_dataset, [train_size, test_size])
    
    print(f"訓練集大小: {len(train_dataset)}")
    print(f"測試集大小: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print("\n--- 初始化模型 ---")
    model = ConvNet(frozen_weights_path='my.txt').to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    # 訓練模型
    train_history = train_model(model, train_loader, criterion, optimizer, device, num_epochs=NUM_EPOCHS)

    # --- 評估與儲存結果 ---
    print("\n--- 正在評估模型並儲存結果 ---")
    labels, preds, probs = evaluate_model(model, test_loader, device)
    
    results_path = os.path.join(RESULTS_DIR, "evaluation_results.npz")
    np.savez(results_path, labels=labels, predictions=preds, probabilities=probs)
    print(f"評估原始資料已儲存至: {results_path}")
    
    accuracy = np.mean(labels == preds)
    report = classification_report(labels, preds, target_names=full_dataset.classes, digits=4)
    accuracy_text = f"Test Accuracy: {accuracy:.4f}\n\n"
    print(accuracy_text)
    print("Classification Report:\n", report)

    report_path = os.path.join(RESULTS_DIR, "test_report.txt")
    with open(report_path, 'w') as f:
        f.write(accuracy_text)
        f.write("Classification Report:\n")
        f.write(report)
    print(f"準確率與分類報告已儲存至: {report_path}")

    print("\n--- 正在繪製並儲存結果圖表 ---")
    class_names = full_dataset.classes
    plot_and_save_confusion_matrix_percentage(labels, preds, class_names, RESULTS_DIR)
    plot_and_save_multiclass_roc(labels, probs, class_names, RESULTS_DIR)

    model_save_path = os.path.join(RESULTS_DIR, 'hybrid_cwt_cnn_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"\n模型已儲存至 {model_save_path}")

    print(f"\n--- 所有流程完成 ---")