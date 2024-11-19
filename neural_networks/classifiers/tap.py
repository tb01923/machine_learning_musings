import torch.nn as nn

class Tap(nn.Module):
    def __init__(self, fn, label="", fire_once=True,):
        super(Tap, self).__init__()

        self.label = label
        self.fn = fn
        self.fire_once=fire_once
        self.not_fired = True

    def forward(self, input):
        if self.fire_once and self.not_fired:
            self.not_fired = False
            self.fn(self.label, input)
        elif not self.fire_once:
            self.fn(self.label, input)

        return input
