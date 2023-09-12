import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch import optim
# from PIL import Image
# from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


def train(classifier: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, epochs, loss_fn):
    classifier.train()
    total_training_loss = []
    for epoch in range(epochs):
        epoch_loss = 0.0

        num_batches = len(loader)
        # for each "tensor of images" in the training set minibatch
        for minibatch in loader:
            data, target = minibatch
            data = data.flatten(start_dim=1)

            # run the classifier
            out = classifier(data)
            # calculate the loss from the prediciton and the true "label"
            minibatch_loss = loss_fn(out, target)
            # calculate the partial derivatives of the weights, and
            #   have the optimizer update the weights
            minibatch_loss.backward()
            optimizer.step()
            # [COMMON MISTAKE] zero out the gradients otherwise they will increment
            optimizer.zero_grad()

            # keep track of the running loss for this minibatch
            epoch_loss += minibatch_loss.item()

        # append running loss from minibatch tot he
        total_training_loss.append(epoch_loss / num_batches)
        print("Epoch: {} train loss: {}".format(epoch + 1, epoch_loss / num_batches))
    return total_training_loss


def test(classifier: nn.Module, loader: DataLoader, loss_fn):
    # put model in inference (prediciton) mode (TODO, this prolly shouldn't cbe here)
    classifier.eval()
    accuracy = 0.0

    with torch.no_grad():
        computed_loss = 0.0
        # for each minibatch in the loader (TODO, does the tester have minibatches?)
        for data, target in loader:
            data = data.flatten(start_dim=1)

            # run the classifier
            out = classifier(data)
            # derive the prediction
            _, predictions = out.max(dim=1)

            # Get loss and accuracy
            computed_loss += loss_fn(out, target)
            # accuracy increases when the prediction
            #  is the same as the attached liable
            accuracy += torch.sum(predictions == target)

        # calculate the length
        length = len(loader) * 64

        print("Test loss: {}, test accuracy: {}".format(
            computed_loss.item() / length,
            accuracy * 100.0 / length))

        return accuracy * 100 / length


def plot_loss(title, loss):
    l = len(loss)
    plt.plot([i for i in range(1, l + 1)], loss)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(title)
    plt.show()


def classify(classifier):
    def _classify(data):
        data = data.flatten(start_dim=1)
        # run the classifier
        out = classifier(data)
        # derive the prediction
        _, preds = out.max(dim=1)
        return preds.item()

    return _classify
