import torch.nn as nn
import torch.nn.functional as F


class RandomCnn(nn.Module):
    def __init__(self):
        super(RandomCnn, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(12, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(320, 50),
            nn.Linear(50, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 320)
        x = self.fully_connected(x)
        return F.log_softmax(x)
