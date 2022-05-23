import torch.nn as nn
import torch.nn.functional as F
import torch


class RNN(nn.Module):
    def __init__(self, kernel_size=5, s=1):
        super(RNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size, s)
        self.conv2 = nn.Conv2d(16, 32, kernel_size, s)
        self.conv3 = nn.Conv2d(32, 64, kernel_size, s)
        self.conv4 = nn.Conv2d(64, 128, kernel_size, s)
        self.conv5 = nn.Conv2d(128, 256, kernel_size, s)
        self.conv6 = nn.Conv2d(256, 512, kernel_size, s)
        self.conv7 = nn.Conv2d(512, 1024, kernel_size, s)

        self.rnn = nn.LSTM(
            input_size=208896,
            hidden_size=128,
            num_layers=1,
            batch_first=True)
        # self.linear = nn.Linear(64, 10)
        # self.dropout1 = nn.Dropout2d(0.25)
        # self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(128, 556)

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
        # Pass data through dropout1
        # x = self.dropout1(x)
        # Flatten x with start_dim=1
        x = torch.flatten(x, 1)
        # Pass data through fc1
        x, _ = self.rnn(x)
        x = self.fc1(x)
        x = F.relu(x)
        # Apply softmax to x
        # output = F.log_softmax(x, dim=1)
        output = x
        return output


def load_model():
    path = "./rnn.pth.tar"
    rnn = RNN()

    torch.save(rnn.state_dict(), path)

    model = RNN()
    model.load_state_dict(torch.load(path))
    return model.eval()
