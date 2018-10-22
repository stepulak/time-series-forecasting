import numpy
import pandas
import datetime
import sys
import time
import matplotlib.pyplot as ma
import statsmodels.tsa.seasonal as st
import statsmodels.tsa.arima_model as arima
import statsmodels.tsa.stattools as tools

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
Deep copy of time series.
"""
def ts_copy(ts):
	return ts.copy(deep=True)

"""
Check whether given time series is stationary.
"""
def ts_check_stationarity(ts, critic_value=0.5):
    try:
        # Dickey-Fuller algorithm
        result = tools.adfuller(ts)
        return result[0] < 0.0 and result[1] < critic_value
    except:
        # Program may raise an exception when there are NA values in TS 
        return False

"""
Fit ARIMA model on given time series.
"""
def ts_fit_arima(ts, order):
	return arima.ARIMA(ts, order=order).fit(disp=0)

"""
Find best ARIMA model for given time series using Akaike information criterion.
"""
def ts_find_best_arima_model(ts, arima_orders):
    best_score = sys.maxsize
    best_order = None
    
    for order in arima_orders:
        model_fit = ts_fit_arima(ts, order)
        score = model_fit.aic
        if score <= best_score:
            best_score = score
            best_order = order

    return best_order

"""
Forecast new values using ARIMA model.
"""
def ts_forecast_arima(arima_model, samples=1):
    return arima_model.forecast(steps=samples)

"""
Estimate integrate (I) parameter by try-fail-success algorithm.
"""
def estimate_integrate_param(ts):
	integrate_param = 0
	ts2 = ts_copy(ts)
	
	while not ts_check_stationarity(ts2) and integrate_param < 2:
		integrate_param += 1
		ts2 = (ts2 - ts2.shift()).interpolate(limit_direction="both")
	
	return integrate_param

"""
Plot graphs for ACF and PACF functions.
"""
def ts_plot_acf_pacf(ts, nlags=40):
	
	def plot_bar(ts, horizontal_line=None):
		ma.bar(range(0, len(ts)), ts, width=0.5)
		ma.axhline(0)
		if horizontal_line != None:
			ma.axhline(horizontal_line, linestyle="-")
			ma.axhline(-horizontal_line, linestyle="-")

	acf = tools.acf(ts, nlags=nlags)
	plot_bar(acf, 1.96 / numpy.sqrt(len(ts)))
	ma.show()
	pacf = tools.pacf(ts, nlags=nlags)
	plot_bar(pacf, 1.96 / numpy.sqrt(len(ts)))
	ma.show()

"""
Split time series into two series - train and test.
"""
def ts_split_train_test(ts, ts_split_train_test=0.8):
    ts_len = len(ts)
    train_end = (int)(ts_len*ts_split_train_test)
    train, test = ts[:train_end], ts[train_end+1:]
    return train, test

"""
Apply ARIMA on given time series with given order.
@M = number of past train values
@N = number of values to predict in one iteration
"""
def run_arima(ts, order, M, N, train_test_ratio):
    # Ignore timestamps
    ts = [x[0] for x in ts.values]

    # Split time series sequence
    train, test = ts_split_train_test(ts, train_test_ratio)
    predictions = []
    confidence = []
    train_end = len(train)+1
    
    # Performance measure
    start_time = time.time()

    # Forecast
    for i in range(train_end, len(ts), N):
        print("Forecasting ", i)
        try:
            start = i-M if i-M >= 0 else 0
            arima_model = ts_fit_arima(ts[start:i], order)
            forecast = ts_forecast_arima(arima_model, N)
            for j in range(0, N):
                predictions.append(forecast[0][j])
                confidence.append(forecast[2][j])
        except:
            print("Error during forecast ", i)
            # Push back last successful predictions
            for j in range(0, N):
                predictions.append(predictions[-1])
                confidence.append(confidence[-1])

    print("TIME ELAPSED ", time.time() - start_time)

    score = 0
    iterations = 0
    result = zip(test, predictions, confidence)
		
    print("Real value,predicted value,conf. interval lower,conf. interval upper")
    for x in result:
        print(x[0], x[1], x[2][0], x[2][1])
        score += pow(x[0]-x[1], 2)
        iterations += 1
    
    print("MSE ", score / iterations)

    ma.plot(ts[train_end+1:], color="blue")
    ma.plot(predictions, color="red")
    ma.show()

def main():
	if len(sys.argv) == 1:
		program_name = sys.argv[0]
		print("Usage:\n")
		print("For ACF, PACF plot:\npython3.6 %s acf_pacf <ts_path> " % (program_name) +
			"<value_column_name> <timestamp_column_name>\n")
		print("For best order estimation:\npython3.6 %s best_order " % (program_name) +
			"<ts_path> <value_column_name> <timestamp_column_name>\n")
		print("For predictions:\npython3.6 %s predictions <ts_path> " % (program_name) +
			"<value_column_name> <timestamp_column_name> <train_test_ratio> " +
			"<arima_order(P D Q)> <number_of_train_samples(or -1 for all)> " + 
			"<number_of_values_to_predict>\n")
		exit()

	method_type = sys.argv[1]
	ts_path = sys.argv[2]
	value_column = sys.argv[3]
	timestamp_column = sys.argv[4]
	ts = ts_load(ts_path, value_column, timestamp_column, lambda x : pandas.to_datetime(x))
		
	def acf_pacf():
		integrate_param = estimate_integrate_param(ts)
		print("POSSIBLE INTEGRATE PARAMETER ", integrate_param)
		ts_plot_acf_pacf(ts)

	def best_order():
		print("INSERT P D Q PARAMETERS OR LEAVE EMPTY LINE FOR BREAK")
		possible_models = []
		for line in sys.stdin:
			params = line.split()
			if len(params) == 0:
				break
			possible_models.append((int(params[0]), int(params[1]), int(params[2])))
		
		order = ts_find_best_arima_model(ts, possible_models)
		print("BEST ORDER ", order)

	def predictions():
		p, d, q = int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]) # ARIMA order
		m, n = int(sys.argv[9]), int(sys.argv[10]) # past and prediction values
		m = len(ts) if m < 0 else m
		run_arima(ts, (p, d, q), m, n, float(sys.argv[5]))

	if method_type == "acf_pacf":
		acf_pacf()
	elif method_type == "best_order":
		best_order()
	elif method_type == "predictions":
		predictions()

if __name__ == "__main__":
    main()
