import torch.nn as nn
import torch.nn.functional as F

from classifiers.tap import Tap

def print_shape(instance, input):
    shape = str(input.shape)
    print("Peek Instance: {} shape: {}".format(instance, shape))

class RandomCnn(nn.Module):
    def __init__(self):
        super(RandomCnn, self).__init__()

        self.features = nn.Sequential(
            # nn.Conv2d(1, 12, kernel_size=11, stride=3, padding=4),
            nn.Conv2d(1, 15, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(15, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            Tap(print_shape, label="after nn.MaxPool2d(kernel_size=2)"),
            nn.ReLU()
        )

        # input features are calculated by the number of output channels in last "feature" step
        #   multiplied by the size of the image on output
        self.fully_connected = nn.Sequential(
            nn.Linear(20, 50),
            nn.Linear(50, 10)
        )

    def forward(self, x):
        x = self.features(x)
        # 320 = 20 out channels * 4 height * 4 height
        x = x.view(-1, 20)
        x = self.fully_connected(x)
        return F.log_softmax(x)