from classifiers.two_layer_ffnn_classifier import TwoLayerFfNnClassifier
from classifiers.random_cnn import RandomCnn
from classifiers.alexnet_cnn import AlexNet
from mnist_classification.mnist_classification import mnist_main, mnist_vector_size

# model_architecture = TwoLayerFfNnClassifier(mnist_vector_size(), 256, 10)
model_architecture = RandomCnn()
# model_architecture = AlexNet()
mnist_classifier = mnist_main(model_architecture)