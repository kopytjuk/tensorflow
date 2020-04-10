from tensorflow.python.keras.preprocessing.timeseries import timeseries_dataset_from_array
import numpy as np

X = np.arange(10).reshape((-1, 1))
y = np.arange(10).reshape((-1, 1))

ts_ds = timeseries_dataset_from_array(X, y, sequence_length=4, batch_size=2)