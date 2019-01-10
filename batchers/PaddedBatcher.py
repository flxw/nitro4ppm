from .AbstractBatcher import AbstractBatcher
from numpy import percentile, array_split
from keras.preprocessing.sequence import pad_sequences
from math import ceil

class PaddedBatcher(AbstractBatcher):
  def format_datasets(model_formatted_data_fn, datapath, target_variable):
    train_X, train_Y, test_X, test_Y = model_formatted_data_fn(datapath, target_variable)

    # make cutoff step a function of the trace length in each percentile
    mlen = ceil(percentile([len(t) for t in train_Y], 80))

    # remove traces from training which are longer than mlen
    train_Y = list(filter(lambda t: len(t) <= mlen, train_Y))

    for layer_name in test_X.keys():
      train_X[layer_name] = list(filter(lambda t: len(t) <= mlen, train_X[layer_name]))

    # now pad all sequences to same length
    # and reshape into batch format
    batch_size = ceil(0.01*len(train_Y))
    split_size = int(len(train_Y) / batch_size)
    n_y_cols = train_Y[0].shape[1]

    train_targets = array_split(pad_sequences(train_Y, padding='post', value=-1337), split_size)
    train_inputs  = {}
    print(len(train_targets), train_targets[0].shape)

    for layer_name in test_X.keys():
      n_x_cols = train_X[layer_name][0].shape[1]
      train_inputs[layer_name] = array_split(pad_sequences(train_X[layer_name], padding='post',value=-1337), split_size)

    # finish the testing set
    n_y_cols = test_Y[0].shape[1]
    test_targets  = [ t.reshape((1, -1, n_y_cols)) for t in test_Y ]

    test_inputs  = {}
    for layer_name in test_X.keys():
      n_x_cols = test_X[layer_name][0].shape[1]
      test_inputs[layer_name]  = [ t.reshape((1, -1, n_x_cols)) for t in test_X[layer_name] ]

    print("Padded batch size for 80% of all shorter traces:", split_size)        
    return train_inputs, train_targets, test_inputs, test_targets
