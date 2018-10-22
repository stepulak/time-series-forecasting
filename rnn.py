import numpy
import pandas
import math
import time
import sys
import datetime
import matplotlib.pyplot as ma
import keras.models as km
import keras.layers as kl
import sklearn.preprocessing as sp

numpy.random.seed(42)

"""
Load time series from CSV file, parse date times and
select column with values.
"""
def ts_load(filename, value_name, date_name, date_parser):
    csv = pandas.read_csv(filename)
    csv.index = date_parser(csv[date_name])
    for x in csv.columns.values.tolist():
	    if x != value_name:
		    del csv[x]
    return csv

"""
LSTM cells are sensitive to large-scaled values,
normalize them to get better resuts.
"""
def ts_normalize(ts):
	scaler = sp.MinMaxScaler(feature_range=(0,1))
	return scaler.fit_transform(ts.values), scaler

"""
Inverse operation to ts_normalize.
"""
def ts_undo_normalization(ts, scaler):
	return scaler.inverse_transform(ts)

"""
Split time series into two series - train and test.
"""
def ts_split_train_test(ts, ts_split_train_test=0.8):
    ts_len = len(ts)
    train_end = (int)(ts_len*ts_split_train_test)
    train, test = ts[:train_end], ts[train_end+1:]
    return train, test

"""
Create LSTM RNN.
"""
def network_create(num_lstm, loss="mse", optimizer="sgd"):
	# Layer based network
	network = km.Sequential()
	# Hidden layer is made from LSTM nodes
	network.add(kl.LSTM(num_lstm, activation="sigmoid", input_shape=(1,1)))
	# Output layer with one output
	network.add(kl.Dense(1))
	network.compile(loss=loss, optimizer=optimizer)
	return network

"""
Train LSTM RNN.
"""
def network_fit(network, train_data, target_data, num_training_iterations):
	return network.fit(train_data, target_data, epochs=num_training_iterations, batch_size=1, verbose=0)

"""
Reshape time series dataset for LSTM RNN 
into [batch size; timesteps; input dimensionality] format.
"""
def dataset_reshape_for_network(dataset):
	return dataset.reshape((dataset.shape[0], 1, dataset.shape[1]))

"""
Create dataset for LSTM RNN training.
Basically this creates two lists, first with training values
and second with lagged target values.
"""
def dataset_create(ts, num_lags=1):
	x = []
	y = []
	for i in range(len(ts)-num_lags-1):
		x.append(ts[i:(i+num_lags), 0])
		y.append(ts[i+num_lags, 0])
	return numpy.array(x), numpy.array(y)

"""
Predict new values with LSTM RNN.
"""
def network_predict_new_values(network, data):
	return network.predict(data)

"""
Load time series from CSV file, 
create LSTM RNN with custom number of cells, train it on data
and try to predict new values.
"""
def rnn(ts_name, num_lstm, iterations,
    train_test_ratio, value_column_name, timestamp_column_name):
    	
    ts = ts_load(ts_name,
		value_column_name,
		timestamp_column_name,
		lambda x : pandas.to_datetime(x))
    predicted_values = []

    # Sigmoids are sensitive to large scaled values, normalize them to <0,1>
    ts, scaler = ts_normalize(ts)
    ts_train, ts_test = ts_split_train_test(ts, train_test_ratio)

    # Create dataset from TS
    train_dataset_x, train_dataset_y = dataset_create(ts_train)
    test_dataset_x, test_dataset_y = dataset_create(ts_test)

    # The input data for our network needs to be
    # provided in [batch size; timesteps; input dimensionality] format
    train_dataset_x = dataset_reshape_for_network(train_dataset_x)
    test_dataset_x = dataset_reshape_for_network(test_dataset_x)

    # Create and fit LSTM network
    start_time = time.time()
    network = network_create(num_lstm)
    network_fit(network, train_dataset_x, train_dataset_y, iterations)
    print("TIME ELAPSED ", time.time() - start_time)
    
    predicted_unscaled = network_predict_new_values(network, test_dataset_x)
    predicted_scaled_back = ts_undo_normalization(predicted_unscaled, scaler)
    test_scaled_back = ts_undo_normalization(ts_test, scaler)
    
    # Present results
    test_result = []
    predicted_result = []
    score = 0
    iterations = 0
    
    print("Real value;predicted value")
    for x in zip(test_scaled_back, predicted_scaled_back):
        test_value = x[0][0]
        predicted_value = x[1][0]
        print("%f,%f" % (test_value, predicted_value))
        test_result.append(test_value)
        predicted_result.append(predicted_value)
        score += pow(test_value - predicted_value, 2)
        iterations += 1
    
    print("MSE ", score / iterations)
    
    ma.plot(test_result, color="blue")
    ma.plot(predicted_result, color="red")
    ma.show()

def main():
	if len(sys.argv) != 7:
		print("Usage:")
		print("python3.6 %s ts_path num_lstm train_iterations train_test_ratio value_column_name timestamp_column_name" % (sys.argv[0]))
		exit()
	
	rnn(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5], sys.argv[6])

if __name__ == "__main__":
    main()
