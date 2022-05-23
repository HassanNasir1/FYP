import torch.nn as nn
import torch.nn.functional as F
import torch


class CNN(nn.Module):
    def __init__(self, kernel_size=5, s=1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size, s)
        self.conv2 = nn.Conv2d(16, 32, kernel_size, s)
        self.conv3 = nn.Conv2d(32, 64, kernel_size, s)
        self.conv4 = nn.Conv2d(64, 128, kernel_size, s)
        self.conv5 = nn.Conv2d(128, 256, kernel_size, s)
        self.conv6 = nn.Conv2d(256, 512, kernel_size, s)
        self.conv7 = nn.Conv2d(512, 1024, kernel_size, s)
        self.fc1 = nn.Linear(208896, 556)

    # x represents our data
    def forward(self, x):
        # Pass data through conv1
        x = self.conv1(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv6(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv7(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # Flatten x with start_dim=1
        x = torch.flatten(x, 1)
        # Pass data through fc1
        x = self.fc1(x)
        x = F.relu(x)
        output = x
        return output


def load_model():
    path = "./cnn.pth.tar"
    cnn = CNN()

    torch.save(cnn.state_dict(), path)

    model = CNN()
    model.load_state_dict(torch.load(path))
    return model.eval()
