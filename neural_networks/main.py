from neural_networks.classifiers.random_cnn import RandomCnn
from neural_networks.mnist.mnist_classification.mnist_classification import mnist_main

# model_architecture = TwoLayerFfNnClassifier(mnist_vector_size(), 256, 10)
model_architecture = RandomCnn()
# model_architecture = AlexNet()
mnist_classifier = mnist_main(model_architecture, num_epoch=40, batch_size=64, num_validations=10)