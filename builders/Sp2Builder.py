import numpy  as np
import pandas as pd

from keras.models import Model
from keras.layers import Dense, Embedding, Input, Masking, concatenate, ReLU, Activation, LSTM, Dropout
from keras.utils  import np_utils
from keras.initializers import glorot_normal

from utils import *
from .AbstractBuilder import AbstractBuilder

class Sp2Builder(AbstractBuilder):
  n_epochs = 150

  def prepare_datasets(path_to_original_data, target_variable):
    train_traces_categorical = load_trace_dataset(path_to_original_data, 'categorical', 'train')
    train_traces_ordinal = load_trace_dataset(path_to_original_data, 'ordinal', 'train')
    train_targets = load_trace_dataset(path_to_original_data, 'target', 'train')
    train_traces_sp2 = load_trace_dataset(path_to_original_data, 'sp2', 'train')

    test_traces_categorical = load_trace_dataset(path_to_original_data, 'categorical', 'test')
    test_traces_ordinal = load_trace_dataset(path_to_original_data, 'ordinal', 'test')
    test_targets = load_trace_dataset(path_to_original_data, 'target', 'test')
    test_traces_sp2 = load_trace_dataset(path_to_original_data, 'sp2', 'test')

    feature_dict = load_trace_dataset(path_to_original_data, 'mapping', 'dict')

    # Use one-hot encoding for categorical values in training and test set
    for col in train_traces_categorical[0].columns:
      nc = len(feature_dict[col]['to_int'].values())
      for i in range(0, len(train_traces_categorical)):
        tmp = train_traces_categorical[i][col].map(feature_dict[col]['to_int'])
        tmp = np_utils.to_categorical(tmp, num_classes=nc)
        tmp = pd.DataFrame(tmp).add_prefix(col)

        train_traces_categorical[i].drop(columns=[col], inplace=True)
        train_traces_categorical[i] = pd.concat([train_traces_categorical[i], tmp], axis=1)

      for i in range(0, len(test_traces_categorical)):
        tmp = test_traces_categorical[i][col].map(feature_dict[col]['to_int'])
        tmp = np_utils.to_categorical(tmp, num_classes=nc)
        tmp = pd.DataFrame(tmp).add_prefix(col)

        test_traces_categorical[i].drop(columns=[col], inplace=True)
        test_traces_categorical[i] = pd.concat([test_traces_categorical[i], tmp], axis=1)

    # categorical and ordinal inputs are fed in on one single layer
    train_traces_seq = [ pd.concat([a,b], axis=1) for a,b in zip(train_traces_ordinal, train_traces_categorical) ]
    test_traces_seq  = [ pd.concat([a,b], axis=1) for a,b in zip(test_traces_ordinal,  test_traces_categorical)  ]

    train_input_batches_seq  = np.array([ t.values for t in train_traces_seq ])
    train_input_batches_sp2  = np.array([ t.values for t in train_traces_sp2 ])
    train_target_batches     = np.array([ t.values for t in train_targets])

    test_input_batches_seq  = np.array([ t.values for t in test_traces_seq ])
    test_input_batches_sp2  = np.array([ t.values for t in test_traces_sp2 ])
    test_target_batches     = np.array([ t.values for t in test_targets ])

    train_input = { 'seq_input': train_input_batches_seq, 'sec_input': train_input_batches_sp2 }
    test_input  = { 'seq_input': test_input_batches_seq, 'sec_input': test_input_batches_sp2 }

    return train_input, train_target_batches, test_input, test_target_batches


  def construct_model(n_train_cols, n_target_cols, learn_windows=False):
    """
    :param n_train_cols: A dictionary of the number of input columns, keyed with the respective layer name
    :type n_train_cols: Tuple
    :param n_target_cols: number of target columns in the output layer
    :type n_target_cols: int
    :returns: Keras model of the constructed model
    :rtype: object
    """
    batch_size = None # None translates to unknown batch size
    window_size = None
    dropout = 0.3
    seq_unit_count = n_train_cols[0] + n_target_cols
    sp2_unit_count = n_train_cols[1] + n_target_cols

    # array format: [samples, time steps, features]
    il = Input(batch_shape=(batch_size, window_size, n_train_cols[0]), name="seq_input")
    main_output = Masking(mask_value=-1337)(il)

    # sizes should be multiple of 32 since it trains faster due to np.float32
    main_output = LSTM(seq_unit_count,
                       batch_input_shape=(batch_size, window_size, n_train_cols[0]),
                       stateful=False,
                       return_sequences=True,
                       unroll=False,
                       kernel_initializer=glorot_normal(),
                       dropout=dropout)(main_output)
    main_output = LSTM(seq_unit_count,
                       stateful=False,
                       return_sequences=not learn_windows,
                       unroll=False,
                       kernel_initializer=glorot_normal(),
                       dropout=dropout)(main_output)

    # SP2 input here
    sp2_batch_shape = (batch_size,window_size, n_train_cols[1]) if not learn_windows else (batch_size, n_train_cols[1])
    il2 = Input(batch_shape=sp2_batch_shape, name="sec_input")
    sp2 = Masking(mask_value=-1337)(il2)
    sp2 = Dense(sp2_unit_count, activation='relu')(sp2)

    main_output = concatenate([main_output, sp2], axis=-1)
    main_output = Dropout(dropout)(main_output)
    main_output = Dense(n_target_cols, activation='relu')(main_output)
    main_output = Dropout(dropout)(main_output)
    main_output = Dense(n_target_cols, activation='softmax')(main_output)

    full_model = Model(inputs=[il, il2], outputs=[main_output])

    full_model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop',
                       metrics=['categorical_accuracy'])

    return full_model
