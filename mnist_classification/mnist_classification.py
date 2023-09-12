import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import two_layer_classifier as todd_class
import common.trainer as todd_common

from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


def mnist_datasets() -> (Dataset, Dataset, Dataset):
    generator = torch.Generator().manual_seed(42)

    # create datasets
    train_dataset = MNIST(".", train=True, download=True, transform=ToTensor())

    # MINST doesn't separate testing from validation so use the
    #  random_split function to allocate 80% to test and 20% to validation
    temp_dataset = MNIST(".", train=False, download=True, transform=ToTensor())
    test_dataset, validation_dataset = random_split(temp_dataset, [.8, .2], generator)

    return train_dataset, test_dataset, validation_dataset


def mnist_dataloaders(train_dataset, test_dataset) -> (DataLoader, DataLoader):
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader


def test_and_plot_loss(classifier, learning_rate, training_loss, test_loader):
    title = "MNIST Training Loss: optimizer {}, lr {}".format("SGD", learning_rate)
    todd_common.plot_loss(title, training_loss)

    accuracy = todd_common.test(classifier, test_loader, nn.CrossEntropyLoss())
    print("total accuracy:", accuracy.item())


def main():
    (train_dataset, test_dataset, validation_dataset) = mnist_datasets()
    (train_loader, test_loader) = mnist_dataloaders(train_dataset, test_dataset)

    # 2. Train the classifier
    learning_rate = 1e-3
    mnist_classifier = todd_class.TwoLayerFfNnClassifier(784, 256, 10)

    # instantiate stochastic gradient descent optimizer
    optimizer: optim.Optimizer = optim.SGD(mnist_classifier.parameters(), lr=learning_rate)

    loss_fn = nn.CrossEntropyLoss()

    # train the classifier, with train   _dataset
    training_loss = todd_common.train(classifier=mnist_classifier,
                                      loader=train_loader,
                                      optimizer=optimizer,
                                      epochs=40,
                                      loss_fn=loss_fn)

    test_and_plot_loss(classifier=mnist_classifier,
                       learning_rate=learning_rate,
                       training_loss=training_loss,
                       test_loader=test_loader)

    classify = todd_common.classify(mnist_classifier)

    # save the trained model
    # torch.save(classifier.state_dict(), 'mnist.pt')

    labels_map = {
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        0: "zero"
    }

    length = len(validation_dataset)

    # todo: I couldn't figure out how to get this to plt in a grid of say 12
    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    for i in range(1, 2):
        sample_idx = torch.randint(length, size=(1,)).item()
        img, label = validation_dataset[sample_idx]

        predicted_label = classify(img)
        title = "guess: " + labels_map[predicted_label] + "; actual: " + labels_map[label]

        plt.title(title)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
        plt.show()


main()
