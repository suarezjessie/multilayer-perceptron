# Multilayer Perceptron

This repository contains the class that implements a Multilayer Perceptron without the use of libraries such as Tensorflow, Keras, etc, to be specific, a 2-layer Multilayer Perceptron. A Multilayer Perceptron is the predecessor of the well-known Neural Network.

## Example Usage

The code snippet below shows the code example that demonstrates how to use the Multilayer Perceptron class.
```
import numpy as np

x = np.random.rand(7,5)
y = np.array([[0,0,1],[0,1,0],[1,0,0],[1,0,0],[0,1,0],[1,0,0],[0,1,0]])

nn = TwoLayerMultilayerPerceptron(inputs = 5, neurons_hidden_layer1 = 7, neurons_hidden_layer2 = 5, learning_rate = 0.5, outputs = 3, epochs = 10, error_metric = 'least_squares')

nn.train(x, y)
nn.predict(np.random.rand(1,5))
```

## Class Parameters

_inputs_
- The number of neurons in the input layer

_neurons_hidden_layer1_
- The number of neurons in the first hidden layer

_neurons_hidden_layer2_
- The number of neurons in the second hidden layer

_learning_rate_
- The learning rate parameter which can directly affect the speed of training 

_outputs_
- The number of neurons in the output layer, ideally for a multiclass classification task, the number of output layers should be the number of classes.

_epochs_
- The number of passes a neural network has to go through the entire dataset for training

_init_method_
- The initialization method. Currently, only the _random_ weight initialization is available

_activation_method_
- The activation function for the neurons after computing the net internal activity. The only available activation is _sigmoid_ 

_error_metric_
- The cost function that computes the error for the output layers. The two cost functions that are available are _least squares_ and _cross entropy_

## Function Definitions

_train(x_train, y_train)_
- trains the neural network. It only accepts _numpy_ arrays. The y_train parameter should be a one-hot encoding of the classes.

_predict(x_test)_
- predicts the class number of the features provided and prints it to the console.

_train_predict(x_train, y_train, x_test, y_test)_
- trains the multilayer perceptron normally but also predicts the test set for every epoch completed, it returns the list of test accuracies.
