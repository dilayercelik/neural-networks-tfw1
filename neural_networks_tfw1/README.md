# Neural Networks with Tensorflow version 1

# About
This is a Python package that allows you to build certain neural networks with the Tensorflow (version 1) framework: you can customise the number of hidden layers and hidden units, choose between the optimization algorithm (Gradient Descent, Momentum or Adam, default being Adam), the size of the minibatches...

This package is compatible with the HDF5 binary data format: the input datasets need to be in this format.

-> See the Usage section for an example usage.

-> See the package in [Pypi](https://pypi.org/project/neural-networks-tf1/)

# Files
setup.py - file necessary for pip installing

neural_networks_tf1 - folder with the files that make up the Python package

# Prerequisites
This package requires:
- python 3 or later  

- numpy package

- matplotlib package

- math Python package

- h5py package

- tensorflow (version 1)

We recommend installing Anaconda before installing the packages above.
Anaconda enables to work with an intuitive way of managing packages and setting up new virtual environments.

To install Anaconda Individual Edition: follow this [link]()


# Installation
## Install with pip:

```bash
pip install neural-networks-tf1
```

## To upgrade the package:

```bash
pip install neural-networks-tf1 --upgrade
```

# Usage

```python
import neural_networks_tf1 as nn_tf1

# instance of Model class
layers_list = [1048, 524, 256, 64, 6]

nn_model_example = Model(layers_list) #with default values: learning_rate=0.001, n_epochs=10, minibatch_size=32):


# build and train the model, then plots the cost of the model
# (specify the arguments of the model function)
quantities, costs, parameters = nn_model_example.model(path_train_dataset, path_test_dataset,
                                     X_train_column, Y_train_column, X_test_column, Y_test_column,
                                     classes_list, optimizer_algo='adam', print_cost=True)

nn_model_example.plot_costs(costs)

# Calculate the correct predictions
correct_predictions = nn_model_example.correct_predictions(quantities)

# Calculate accuracy on the train set
train_accuracy = nn_model_example.train_accuracy(correct_predictions, quantities)

# Calculates accuracy on the test set
test_accuracy = nn_model_example.test_accuracy(correct_predictions, quantities)
```


# Author
Dilay Fidan Ercelik

LinkedIn: [Dilay Fidan Ercelik](https://www.linkedin.com/in/dilay-fidan-ercelik-682675194/)

# Acknowledgments
I would like to express my gratitude for the lessons provided at Udacity - Machine Learning Engineer Nanodegree Course
and the Deep Learning specialization provided by deeplearning.ai and Coursera, without which I wouldn't have been able to build this package.

# License
MIT License - see [license.txt]()
