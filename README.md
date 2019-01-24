# Introduction
This framework can assist you in training a Keras model for sequence prediction.
It provides a training infrastructure that logs training information, checkpoints models, and offers early stopping.
The infrastructure ties together implementations of two abstract classes.
The `model_runner` framework will detect each implementation automatically and list it when calling `model_runner -h`.

# Installation
Install the required packages via `pip install -r requirements.txt`or via any other package manager you use for Python.

# Adding a custom network
Inherit from `AbstractBuilder` and place the file into `builders/`.
This class file should contains at least two method:

```python
prepare_datasets(path_to_original_data, target_variable)
```
The method reads in sequential training and validation data from a source of your choosing.
It returns a four-tupel with X and Y values for both training and validation sets.
At this point, the return data is a simple array of samples, not divided into batches.
The exact contents of the return values are described in the abstract class' docstring of the method.

```python
construct_model(n_train_cols, n_target_cols, learn_windows=False)
```
A method which constructs the Keras model. Pretty straightforward.

Additionally, the variable `n_epochs` can be defined so that the training scripts knows a maximum number of epochs

# Adding a custom batching strategy
Inherit from `AbstractBatcher` and place the file into `batchers/`.
This class file only needs to implement a single method:

```python
format_datasets(model_formatted_data_fn, datapath, target_variable)
```
The method takes a function pointer, path to the raw data, and the column name of the target variable.
All parameters and return values are documented in the docstring of the abstract class.
The point of the function is to format the input data into batches which can be fed into the network, i.e. that honor the requirement of samples containing the same number of timesteps.

# Example
The frontend could be invoked with a command like the following
```bash
model_runner.py evermann grouped --gpu=5 ../logs/bpic2011
```

This runs the `EvermannBuilder` implementation with the `GroupedBatcher` batching strategy.
The grouped batcher creates batches of samples with the same number of timesteps.
