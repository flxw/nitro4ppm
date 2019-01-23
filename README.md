# Introduction
This framework can assist you in training a Keras model for sequence prediction.
The framework provides a training infrastructure that logs training information,
checkpoints models, and offers early stopping.
Inherit the `AbstractBuilder` class to add a model.
The `model_runner` framework will detect it automatically and list it when calling `model_runner -h`.

# Installation
Install the required packages via `pip install -r requirements.txt`or via any other package manager you use for Python.

# Usage instructions


# Example

`model_runner.py evermann grouped --gpu=5 ../logs/bpic2011`
