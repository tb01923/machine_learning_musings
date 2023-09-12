import torch.nn as nn

class TwoLayerFfNnClassifier(nn.Module):
    def __init__(self, in_dim, feature_dim, out_dim):
        super(TwoLayerFfNnClassifier, self).__init__()
        # two layer classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, feature_dim, bias=True),
            nn.ReLU(),
            nn.Linear(feature_dim, out_dim, bias=True)
        )

    def forward(self, x):
        return self.classifier(x)
