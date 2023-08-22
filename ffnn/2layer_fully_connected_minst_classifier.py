# -*- coding: utf-8 -*-
"""MINST Classifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-2VXSlXg6a1lq-5hNsdJycd1-hd53Nug

# MINST classifier based on the examples within:

*   Fundamentals of Deep Learning, 2nd Edition
*   Programming PyTorch for Deep Learning
"""

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch import optim
#from PIL import Image
#from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

"""# The core objects and functions

1. set up the base [classifier](https://learning.oreilly.com/library/view/fundamentals-of-deep/9781492082170/ch05.html#:-:text=class%20BaseClassifier(nn.Module)%3A)
"""

class BaseClassifier(nn.Module):
  def __init__(self, in_dim, feature_dim, out_dim):
    super(BaseClassifier, self).__init__()
    # two layer classifier
    self.classifier = nn.Sequential(
      nn.Linear(in_dim, feature_dim, bias=True),
      nn.ReLU(),
      nn.Linear(feature_dim, out_dim, bias=True)
    )

  def forward(self, x):
    return self.classifier(x)

"""2. define the [training loop](https://learning.oreilly.com/library/view/fundamentals-of-deep/9781492082170/ch05.html#:-:text=def%20train(classifier,epochs%2C%0A%C2%A0%C2%A0%C2%A0%C2%A0%C2%A0%C2%A0%C2%A0%C2%A0%C2%A0%C2%A0loss_fn%3Dloss_fn)%3A), inputs the base classifier, an optimization function, loss function and the number of epocks"""

def train(classifier, loader, optimizer, epochs, loss_fn):
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
    total_training_loss.append(epoch_loss/num_batches)
    print("Epoch: {} train loss: {}".format(epoch+1, epoch_loss/num_batches))
  return total_training_loss

"""3. define the test loop"""

def test(classifier, loader, loss_fn):
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
          _, preds = out.max(dim=1)

          # Get loss and accuracy
          computed_loss += loss_fn(out, target)
          # accuracy increases when the prediciton
          #  is the same as the attached lable
          accuracy += torch.sum(preds==target)

      # calculate the length
      length = len(loader)*64

      print("Test loss: {}, test accuracy: {}".format(
          computed_loss.item()/length,
          accuracy*100.0/length))

      return accuracy * 100 / length

"""4. create a function that classifies an image with the trained model. pass in the tensor for the image data, and output the classifier: 0..9"""

def classify(data):
  data = data.flatten(start_dim=1)
  # run the classifier
  out = classifier(data)
  # derive the prediction
  _, preds = out.max(dim=1)
  return preds.item()

""" 5. define a function that plots the loss from the training exercise, for no real reason other than to look at it"""

def plot_loss(title, loss):
  l = len(loss)
  plt.plot([i for i in range(1,l+1)], loss)
  plt.xlabel("Epoch")
  plt.ylabel("Training Loss")
  plt.title(title)
  plt.show()

"""# Train the classifier

1. load MNIST training and test as pytorch **DataSet**, and construct the pytorch **DataLoader**
"""

# for random numbers
generator = torch.Generator().manual_seed(42)

# create datasets
train_dataset = MNIST(".", train=True, download=True, transform=ToTensor())

# MINST doesn't separate testing from validation so use the
#  random_split function to allocate 80% to test and 20% to validation
temp_dataset  = MNIST(".", train=False, download=True, transform=ToTensor())
test_dataset, validation_dataset = random_split(temp_dataset, [.8, .2], generator)

train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader   = DataLoader(test_dataset, batch_size=64, shuffle=False)

"""2. Train classifier, and capture the loss after every epoch (this bit takes 5-10 minutes)"""

# define the classifier
learning_rate = 1e-3
in_dim, feature_dim, out_dim = 784, 256, 10
classifier = BaseClassifier(in_dim, feature_dim, out_dim)

# instantiate stochasit gradient descent optimizer
optimizer = optim.SGD(classifier.parameters(), lr=learning_rate)

#train the classifier, with train_dataset
training_loss = train(classifier=classifier, loader=train_loader, optimizer=optimizer, epochs=40, loss_fn=nn.CrossEntropyLoss())

# save the trained model
torch.save(classifier.state_dict(), 'mnist.pt')

"""3. plot the loss (just to visualize it)"""

title = "MNIST Training Loss: optimizer {}, lr {}".format("SGD", learning_rate)
plot_loss(title, training_loss)

"""4. test the classifier and show the accuracy"""

accuracy = test(classifier, test_loader, nn.CrossEntropyLoss())
print("total accuracy:", accuracy.item())

"""5. pick some images from the validation dataset and a) classify them, and then plot that image and the correct label"""

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

#todo: I couldn't figure out how to get this to plt in a grid of say 12
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
for i in range(1,2):
  sample_idx = torch.randint(length, size=(1,)).item()
  img, label = validation_dataset[sample_idx]

  predicted_label = classify(img)
  title = "guess: " + labels_map[predicted_label] + "; actual: " + labels_map[label]

  plt.title(title)
  plt.axis("off")
  plt.imshow(img.squeeze(), cmap="gray")
  plt.show()