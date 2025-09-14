
import torch
from torchvision.transforms import transforms
from torch import nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(3, 3), padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout1 = nn.Dropout(p=0.25)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(3, 3), padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout2 = nn.Dropout(p=0.25)

        self.conv3 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout3 = nn.Dropout(p=0.25)

        self.conv4 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout4 = nn.Dropout(p=0.25)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=588, out_features=3 * 16)
        self.fc2 = nn.Linear(in_features=3 * 16, out_features=4)

        self.classes = ['F', 'N', 'Q','V']

    def forward(self, x):
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

        x=self.flatten(x)

        
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))

        return x
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load("./model112.pth")
model = ConvNet()
model.load_state_dict(state_dict)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(112),
    transforms.Grayscale(),
    transforms.ToTensor(),
])
def CNN_processing(cwt_images):
    pred_list=[]
    for i, image in enumerate(cwt_images):
        # 創建 PILLOW 
        image = Image.fromarray((image * 255).astype(np.uint8))
        input_data = transform(image).unsqueeze(0).to(device)

        # 進行推理
        with torch.no_grad():
            output = model(input_data)
            pred = output.argmax(dim=1)

        # 列印結果
        a=np.array(['F', 'N', 'Q','V'])
        i=pred.item()
        pred_list.append(a[i])
    return pred_list