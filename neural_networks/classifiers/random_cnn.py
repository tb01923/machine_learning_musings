import torch.nn as nn
import torch.nn.functional as F


def print_shape(instance, input):
    shape = str(input.shape)
    print("Peek Instance: {} shape: {}".format(instance, shape))


class RandomCnn(nn.Module):
    def __init__(self):
        super(RandomCnn, self).__init__()

        # self.features = nn.Sequential(
        #     # nn.Conv2d(1, 12, kernel_size=11, stride=3, padding=4),
        #     nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=0),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=10, out_channels=20, kernel_size=4, stride=1, padding=0),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=20, out_channels=25, kernel_size=2, stride=1, padding=0),
        #     nn.Dropout2d(),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.ReLU()
        # )

        self.features = nn.Sequential(
            # nn.Conv2d(1, 12, kernel_size=11, stride=3, padding=4),
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )

        self.linear_input = 100
        # input features are calculated by the number of output channels in last "feature" step
        #   multiplied by the size of the image on output
        self.fully_connected = nn.Sequential(
            nn.Linear(in_features=self.linear_input, out_features=50),
            nn.Linear(in_features=50, out_features=10)
        )

    def forward(self, x):
        x = self.features(x)
        # 320 = 20 out channels * 4 height * 4 height
        x = x.view(-1, self.linear_input)
        x = self.fully_connected(x)
        return F.log_softmax(x)