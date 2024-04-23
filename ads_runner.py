# from ischedule import schedule, run_loop
from datetime import timedelta
import prometheus_api
import calendar
import time
import config
import json
import csv
from datetime import timedelta
# from dateutil.parser import *
from datetime import date, datetime, time
from email_handler import email_anomaly
import pandas as pd
from codes.test import *

import grafana_api
import plotly.graph_objects as go

from Sarima_Code.Sarima_Train import grid_search, model_creation

# from Sarima_Code.Sarima_Train import *

from apscheduler.schedulers.background import BackgroundScheduler

csv_anomalies_file = "detected_anomalies.csv"
csv_anomalies_file_statistic = "detected_anomalies_stat.csv"

RETRAIN_JOB_ID = 'retrain_job'
TEST_JOB_ID = 'test_job'
GENERAL_LOG_FILE = 'general_log.txt'
ERROR_LOG_FILE = 'error_log.txt'

retrain_scheduler = BackgroundScheduler()
test_scheduler = BackgroundScheduler()

default_model_vae, default_lstm_nn_model = create_models()

import time

model_reading_lock = False
model_vae_list = [None]
lstm_nn_model_list = [None]
current_threshold = -1

test_cpu_usage_query = "container_cpu_usage_seconds_total%7Bnamespace%3D%22default%22%2C%20pod%3D%22ads-kube-state-metrics-fbb87fd8f-xgx2h%22%2Ccontainer%3D%22POD%22%7D"

# default setting!
metric_query = test_cpu_usage_query
re_train_interval = 1
num_train_sample = 500
sampling_resolution = 15
model_name = "VAE"

test_interval = 1
grafana_base_url = "http://localhost:53639/api/"
grafana_auth = 'eyJrIjoiak5qWjZmNVFROVdoT0tiaEtxOEplZFVyVGlMbEg1MUIiLCJuIjoiZGVtbyIsImlkIjoxfQ=='
panel_id = None
dashboard_id = None

# the time stamp that test data has been collected!
collected_data_timestamp = None

from pathlib import Path  # convenient way to deal w/ paths
import plotly.graph_objects as go  # creates plots
import numpy as np  # standard for data processing
import pandas as pd  # standard for data processing
import json  # we have anomalies' timestamps in json format
import matplotlib.pyplot as plt
import pandas as pd

import random
fake_anomaly_generator = True
num_fake_anomalies = 1

def save_csv_data(_query, _data_num, _step):
    '''
    Store collected data which is json after converting it to a format which can be accepted by our models.
    At the end it should call one of the training function again with respect to the selected model.
    :param _query: query which need to send to prometheus. It can come from api-call.
    :param _data_num: number of data points that need to be collect.
    :param _step: query resolution (i.e: 15s)
    :return: True if it was able to store data, False otherwise.
    '''
    gmt = time.gmtime()
    current_ts = calendar.timegm(gmt)
    print("timestamp:-", current_ts)

    start_ts = current_ts - (_data_num * _step)

    query_time_range = "&start={}&end={}&step={}".format(start_ts, current_ts, _step)
    final_query = _query + query_time_range
    result = prometheus_api.general_range_promql(final_query)
    try:
        ls_result = [[str(datetime.fromtimestamp(i[0])), i[1]] for i in
                     json.loads(result.text)['data']['result'][0]['values']]
        output_file_name = 'retraining_file.csv'
        with open(output_file_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'value'])
            writer.writerows(ls_result)

        # impute missing values
        _resolution = str(_step) + 'S'
        df = pd.read_csv('retraining_file.csv')
        df['timestamp'] = pd.to_datetime(df.timestamp)
        # DatetimeIndex = pd.date_range(start=df['timestamp'][0], end=df['timestamp'].iloc[-1], freq='5s')
        # Missing_DatetimeIndex = DatetimeIndex[~DatetimeIndex.isin(df["timestamp"])]
        # print(Missing_DatetimeIndex)
        # print(df[df['timestamp'] >= '2022-03-15 13:39:51'])
        df = df.resample(_resolution, on='timestamp', base=6).mean()
        # print(df[df.index >= '2022-03-15 13:39:51'])
        df['value'] = df['value'].fillna(method='ffill')
        df.to_csv('retraining_file.csv', index=True, index_label=None)
        return True
    except ValueError as e:
        print(e)
        return False


# def save_csv_data(_query, _start_collection, _step):
#     '''
#     Store collected data which is json after converting it to a format which can be accepted by our models.
#     At the end it should call one of the training function again with respect to the selected model.
#     :param _query: query which need to send to prometheus. It can come from api-call.
#     :param _start_collection: start time of query(i.e: 1500 second).
#     :param _step: query resolution (i.e: 15s)
#     :return: True if it was able to store data, False otherwise.
#     '''
#
#     gmt = time.gmtime()
#     current_ts = calendar.timegm(gmt)
#     print("timestamp:-", current_ts)
#     start_ts = current_ts - _start_collection
#
#     query_time_range = "&start={}&end={}&step={}".format(start_ts, current_ts, _step)
#     final_query = _query + query_time_range
#     result = prometheus_api.general_range_promql(final_query)
#     try:
#         ls_result = [[str(datetime.fromtimestamp(i[0])), i[1]] for i in
#                      json.loads(result.text)['data']['result'][0]['values']]
#         output_file_name = 'retraining_file.csv'
#         with open(output_file_name, 'w', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(['timestamp', 'value'])
#             writer.writerows(ls_result)
#
#         return True
#     except ValueError as e:
#         print(e)
#         return False
def forcasting_Visualisation_on_validation(valid) :
   # Prepare layout w/ titles
   layout = dict(xaxis=dict(title='Timestamp-FORCASTING ON VALIDATION SET'), yaxis=dict(title='CPU Utilization'))

#   # Create the figure for plotting the data
   fig = go.Figure(layout=layout)
   fig.add_trace(go.Scatter(x=valid['timestamp'], y=valid['value'],
                           mode='markers', name='Ground Truth',
                           marker=dict(color='blue')))
   fig.add_trace(go.Scatter(x=valid['timestamp'],y=valid['anomaly_predicted'],
                           mode='markers', name='Predicted Value',
                           marker=dict(color='yellow')))
   #for r in valid.loc[valid.anomaly_label.astype(bool),].iterrows():
        #fig.add_vline(x=r[1]["timestamp"],  line_width=1, line_dash="solid", line_color="red")
   fig.write_image("FORCASTING ON VALIDATION SET.png")
   #savefig('./test_reconstructed_{}.pdf'.format(start_test_time))
   fig.show()


def forcasting_Train_Data(train, model):
    start = len(train)
    end = len(train) - 1
    train['predict'] = model.predict(start=0, end=len(train) - 1, typ='levels').rename('sARIMA Predictions')
    # plt.plot(train['predict'],label='predict')
    # #valid['value'].plot(legend=True)
    # plt.plot(train['value'],label='value')
    # plt.legend(bbox_to_anchor=(0.75, 1.15), ncol=2)
    # plt.show()
    train_forcast_errors = np.mean(np.abs(train['predict'] - train['value']) / np.abs(train['value']) * 100)
    return train_forcast_errors


def forcasting_Test_Data(test, model):
    start = len(test)
    end = len(test) - 1
    test['predict'] = model.predict(start=0, end=len(test) - 1, typ='levels').rename('sARIMA Predictions')
    # plt.plot(train['predict'],label='predict')
    # #valid['value'].plot(legend=True)
    # plt.plot(train['value'],label='value')
    # plt.legend(bbox_to_anchor=(0.75, 1.15), ncol=2)
    # plt.show()
    train_forcast_errors = np.mean(np.abs(test['predict'] - test['value']) / np.abs(test['value']) * 100)
    return forcasting_Test_Data


##zscore

##threshold


def calculate_mean_std(df, threshold):
    upper = df.value.mean() + threshold * df.value.std()
    lower = df.value.mean() - threshold * df.value.std()
    return upper, lower

def detect_anomalies_threshold(threshold,train,test):
  print("inside detect anomalies threshold")
  # Calculate errors for the gicen data
  Upper,Lower= calculate_mean_std(train,threshold)
  test['anomaly_predicted']=1#considering all values are anomaly
  new_df= test['value'][(test.value<Upper) & (test.value>Lower)]
  test['anomaly_predicted'][new_df.index]=0 #this are all non_anomaly point
  index=test[test['anomaly_predicted']==1].index
  # print(df)
  # print(len(index))
  return index,test



def calculate_Z_SCORE(df):
  mean =df.value.mean()
  std =df.value.std()
  z_score= (df['value']-mean)/std
  return z_score

def detect_anomalies_zscore(threshold,train,test):
  # Calculate errors for the gicen data
  z_score= calculate_Z_SCORE(train)
  test["z_score"]=z_score
  test['anomaly_predicted']=1
  new_df= test['z_score'][(test.z_score>-threshold) & (test.z_score<threshold)]
  test['anomaly_predicted'][new_df.index]=0 #this are all non_anomaly point
  index=test[test['anomaly_predicted']==1].index
  return index,test


def run_statistical_model():
    '''
    It contains both train and test functionality of the SARIMA model.
    '''
    print('statistical model re train triggered!')
    # save new csv file
    global collected_data_timestamp
    global re_train_interval
    global sampling_resolution
    global num_train_sample

    gmt = time.gmtime()
    current_ts = calendar.timegm(gmt)
    print("timestamp:-", current_ts)
    # start_ts = current_ts - _start_collection
    # query_time_range = "&start={}&end={}&step={}".format(collected_data_timestamp, current_ts, sampling_resolution)

    total_num_samples = num_train_sample + re_train_interval * sampling_resolution

    # update current timestamp!
    if collected_data_timestamp is None:
       collected_data_timestamp = current_ts
       test_dt_start = collected_data_timestamp
       print("its in if")
    else:
        test_dt_start = collected_data_timestamp
        print(test_dt_start)
        collected_data_timestamp = current_ts

    # save data into the .csv file.
    save_csv_data(metric_query, total_num_samples, sampling_resolution)

    # load saved data and find test index
    temp_df = pd.read_csv('retraining_file.csv', index_col=0)
    temp_df = temp_df.iloc[1:, :]
    temp_df.index = pd.to_datetime(temp_df.index)
    
    start_test_index = temp_df.index.get_loc(datetime.fromtimestamp(test_dt_start), method='backfill')
    print(start_test_index)

    print(len(temp_df.iloc[start_test_index:, :]))

    # Train
    
    Train = temp_df[:start_test_index]
    #adding here
    Train = Train.reset_index()
    #Train = pd.DataFrame(Train)
    #print(len(temp_df.iloc[start_test_index:, :]))
    #best_pdq = []
    #best_pdq, best_seasonal_pdq, aic = grid_search(Train)
    #_model = model_creation(Train, best_pdq, best_seasonal_pdq)
    _model = model_creation(Train)
    # forcasting_errors=forcasting_Train_Data(Train,Model)
    # Threshold=3
    # anomalies_in_Forcasted_Data_threshold,dataframe = detect_anomalies_threshold(Threshold, Train)
    # anomalies_in_Forcasted_Data_threshold_Zscore,dataframe = detect_anomalies_zscore(Threshold, Train)

    # Test
    #start_test_index = temp_df.index.get_loc(datetime.fromtimestamp(test_dt_start), method='backfill')
    #print(start_test_index)
    #Test = start_test_index
    Test = temp_df[start_test_index:]
    Test = Test.reset_index()
    
    Threshold = 3
    #forcasting_errors = forcasting_Test_Data(Test, _model)
    #anomalies_in_Forcasted_Data_threshold, th_dataframe = detect_anomalies_threshold(Threshold, Test)
    #anomalies_in_Forcasted_Data_threshold_Zscore, zs_dataframe = detect_anomalies_zscore(Threshold, Test)
    print("printing train and test")
    print(Train.head())
    print(Test.head())
    
    print("first training index",Train['timestamp'][0])
    print("first test index",Test['timestamp'][0])
   # print("last test index",Test['timestamp'].iat[-1])
    anomalies_in_Forcasted_Data_threshold,th_dataframe = detect_anomalies_threshold(Threshold,Train,Test)
    #anomalies_in_Forcasted_Data_threshold_Zscore, zs_dataframe = detect_anomalies_zscore(Threshold,Train,Test)
    print("anomalies")
    print(anomalies_in_Forcasted_Data_threshold)
    forcasting_Visualisation_on_validation(Test)
    # save anomalies
    with open(csv_anomalies_file_statistic, 'a') as csv_file:
        writer = csv.writer(csv_file)
        for item in list(anomalies_in_Forcasted_Data_threshold):
            writer.writerow([th_dataframe.iloc[item]['timestamp']])

    for item in list(anomalies_in_Forcasted_Data_threshold):
        # Add reconstruction error and the threshold to anomaly description.
        desc = ""

        grafana_api.adding_anomalies_annotation(grafana_base_url, grafana_auth, dashboard_id, panel_id,
                                                int(th_dataframe.iloc[item]['timestamp']), int(th_dataframe.iloc[item]['timestamp']) + 1, desc)

    if len(anomalies_in_Forcasted_Data_threshold) > 0:
        test_range = "{} {}".format(th_dataframe.iloc[0]['timestamp'], th_dataframe.iloc[-1]['timestamp'])
        email_anomaly(len(anomalies_in_Forcasted_Data_threshold), test_range)

    # we need to check and retrieve the data index which is near to the @collected_data_timestamp which is the timestamp
    # of previous re-training/testing.
    # add the test here. Calculate the indices which need to be test!





def VAE_LSTM_load_data():
    '''
    This function load one .cvs (a sequence). It will be called in process_and_save_specified_dataset() func
    :return:
    '''
    data_file = 'retraining_file.csv'
    anomalies = []
    t_unit = config.query_step

    t = []
    readings = []
    idx_anomaly = []
    i = 0
    with open(data_file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        print("\n--> Anomalies occur at:")
        for row in readCSV:
            if i > 0:
                t.append(i)
                readings.append(float(row[1]))
                for j in range(len(anomalies)):
                    if row[0] == anomalies[j]:
                        idx_anomaly.append(i)
                        print("  timestamp #{}: {}".format(j, row[0]))
            i = i + 1
    t = np.asarray(t)
    readings = np.asarray(readings)
    print(readings)
    print(t)
    print("\nOriginal csv file contains {} timestamps.".format(t.shape))
    print("Processed time series contain {} readings.".format(readings.shape))
    print("Anomaly indices are {}".format(idx_anomaly))

    return t, t_unit, readings, []


# model specific
def process_and_save_specified_dataset():
    '''
    VAE_LSTM preprocessing. Save processed data with npz format.
    :return:
    '''

    t, t_unit, readings, idx_anomaly = VAE_LSTM_load_data()

    # assign everything to the training part
    idx_split = [0, len(t) + 1]

    # split into training and test sets
    training = readings[idx_split[0]:idx_split[1]]
    t_train = t[idx_split[0]:idx_split[1]]

    # normalise by training mean and std
    train_m = np.mean(training)
    train_std = np.std(training)
    print("\nTraining set mean is {}".format(train_m))
    print("Training set std is {}".format(train_std))
    readings_normalised = (readings - train_m) / train_std

    training = readings_normalised[idx_split[0]:idx_split[1]]
    if idx_split[0] == 0:
        test = readings_normalised[idx_split[1]:]
        t_test = t[idx_split[1]:] - idx_split[1]
        idx_anomaly_test = np.asarray(idx_anomaly) - idx_split[1]
    else:
        test = [readings_normalised[:idx_split[0]], readings_normalised[idx_split[1]:]]
        t_test = [t[:idx_split[0]], t[idx_split[1]:] - idx_split[1]]
        idx_anomaly_split = np.squeeze(np.argwhere(np.asarray(idx_anomaly) > idx_split[0]))
        idx_anomaly_test = [np.asarray(idx_anomaly[:idx_anomaly_split[0]]),
                            np.asarray(idx_anomaly[idx_anomaly_split[0]:]) - idx_split[1]]
    print("Anomaly indices in the test set are {}".format(idx_anomaly_test))

    save_dir = './datasets/processed_data/'
    np.savez(save_dir + 'retraining_file' + '.npz', t=t, t_unit=t_unit, readings=readings, idx_anomaly=idx_anomaly,
             idx_split=idx_split, training=training, test=test, train_m=train_m, train_std=train_std,
             t_train=t_train, t_test=t_test, idx_anomaly_test=idx_anomaly_test)
    print("\nProcessed time series are saved at {}".format(save_dir + 'retraining_file' + '.npz'))


def run_re_train_schedulers(_query, _re_train_interval, _num_train_sample, _sampling_resolution):
    global metric_query
    global re_train_interval
    global num_train_sample
    global sampling_resolution
    global model_name
    metric_query = _query
    re_train_interval = int(_re_train_interval)
    model_name = "VAE"
    num_train_sample = int(_num_train_sample)
    sampling_resolution = int(_sampling_resolution)
    start_re_train_scheduler()


def run_statistical_model_schedulers(_query, _re_train_interval, _num_train_sample, _sampling_resolution,
                                     _grafana_base_url,
                                     _grafana_auth, _panel_id, _dashboard_id):
    '''
    It will run the single scheduler to do the train/test with statistical model.
    It also set all the required variable for the train/test.
    :param _query: promql query.
    :param _re_train_interval:  interval for running the model.
    :param _num_train_sample: number of samples to be considered each time,
    :param _sampling_resolution: resolution which will be used in promql query.
    :param _grafana_base_url: grafana url to send anomalies to it.
    :param _grafana_auth:
    :param _panel_id: grafana panel_id to visualize the anomalies
    :param _dashboard_id: dashboard panel_id to visualize the anomalies
    :return: None
    '''
    # Train variables
    global metric_query
    global re_train_interval
    global num_train_sample
    global sampling_resolution
    metric_query = _query
    re_train_interval = int(_re_train_interval)
    num_train_sample = int(_num_train_sample)
    sampling_resolution = int(_sampling_resolution)
    # Test variables
    global grafana_base_url
    global grafana_auth
    global panel_id
    global dashboard_id
    global grafana_auth
    grafana_base_url = _grafana_base_url
    grafana_auth = _grafana_auth
    panel_id = int(_panel_id)
    dashboard_id = int(_dashboard_id)
    # start statistical models scheduler
    start_statistical_scheduler()


def run_test_schedulers(_test_interval, _start_delay, _grafana_base_url, _grafana_auth, _panel_id, _dashboard_id):
    global test_interval
    global grafana_base_url
    global grafana_auth
    global panel_id
    global dashboard_id
    global grafana_auth
    test_interval = int(_test_interval)
    grafana_base_url = _grafana_base_url
    grafana_auth = _grafana_auth
    panel_id = int(_panel_id)
    dashboard_id = int(_dashboard_id)
    if model_vae_list[0] is None or lstm_nn_model_list[0] is None:
        # model should train first!
        return False
    else:
        # start testing!
        start_test_scheduler(int(_start_delay))
        return True


def start_statistical_scheduler():
    if len(retrain_scheduler.get_jobs()) == 0:
        print("re train interval is {}".format(re_train_interval))
        retrain_scheduler.add_job(run_statistical_model, 'interval', minutes=re_train_interval, id=RETRAIN_JOB_ID)
        retrain_scheduler.start()
        print('Start training schedule!')
        return True
    else:
        print('Job already exist!')
        return False


def start_re_train_scheduler():
    if len(retrain_scheduler.get_jobs()) == 0:
        print("re train interval is {}".format(re_train_interval))
        if model_name == "VAE":
            retrain_scheduler.add_job(re_train_VAE, 'interval', minutes=re_train_interval, id=RETRAIN_JOB_ID)
        else:
            retrain_scheduler.add_job(run_statistical_model, 'interval', minutes=re_train_interval, id=RETRAIN_JOB_ID)
        try:
            retrain_scheduler.start()
        except Exception as e:
            print('The scheduler already started!')

        print('Start training schedule!')
        return True
    else:
        print('Job already exist!')
        return False


def remove_re_train_scheduler():
    retrain_scheduler.remove_job(RETRAIN_JOB_ID)


def start_test_scheduler(delay=2):
    '''
    @delay: start delay by hours
    :return:
    '''
    # TODO: minutes must change to hours!
    time_change = timedelta(minutes=delay)
    start_date = datetime.now() + time_change

    global test_interval

    if len(test_scheduler.get_jobs()) == 0:
        if model_name == "VAE":
            test_scheduler.add_job(test_VAE, 'interval', minutes=test_interval, id=TEST_JOB_ID,
                                   start_date=start_date)
        else:
            # TODO change test_VAE to statistical model test function (@Rezwan)
            test_scheduler.add_job(test_VAE, 'interval', minutes=test_interval, id=TEST_JOB_ID,
                                   start_date=start_date)
        try:
            test_scheduler.start()
        except Exception as e:
            print('The scheduler already started!')

        return True
    else:
        print('Testing job already exist!')
        return False


def remove_test_scheduler():
    test_scheduler.remove_job(TEST_JOB_ID)


def re_train_VAE():
    '''
    It should start the procedure which start by collecting and saving data and lead to the training.
    At the end it should update the variable which test function use to select the stored model
    :return:
    '''
    # try:
    with open(GENERAL_LOG_FILE, 'a', newline='') as f:
        f.write("re-train-VAE triggered!")
        f.write('------------------------')

    # fill for the first time collected_data_timestamp!
    global collected_data_timestamp
    if collected_data_timestamp is None:
        gmt = time.gmtime()
        # TODO change 1000 to realistic number ( maybe remove and just add if the data is not sufficient)
        collected_data_timestamp = calendar.timegm(gmt) - 1000

    print('re training triggered!')
    save_csv_data(metric_query, num_train_sample, sampling_resolution)
    process_and_save_specified_dataset()
    # run the training command
    os.system('python3 codes/train.py --config codes/NAB_config.json')

    # Update parameters for testing

    _config = process_config('codes/NAB_config.json')
    _data = load_train_data(_config)

    # change the current model for testing after new training.
    model_vae, lstm_nn_model = load_model(_config, default_model_vae)

    # calculate the threshold!
    val_vae_recons_error, val_lstm_recons_error = calculte_val_reconstruction_error(_data, model_vae, lstm_nn_model,
                                                                                    _config)
    calculated_threshold = get_percentiles_threshold(val_lstm_recons_error, 90)
    print("calculated threshold is {}".format(calculated_threshold))

    global current_threshold
    current_threshold = calculated_threshold

    # add delay if the lists are being used
    global model_reading_lock
    if model_reading_lock is True:
        time.sleep(0.01)

    model_reading_lock = True
    model_vae_list[0] = model_vae
    lstm_nn_model_list[0] = lstm_nn_model
    model_reading_lock = False
    # except Exception as e:  # most generic exception you can catch
    #     with open(ERROR_LOG_FILE, 'a', newline='') as f:
    #         e = str(e) + '\n'
    #         f.write(e)
    #         f.write('------------------------')


def read_test_data(_query, _step):
    '''
    It should use promql query to collect test data in order to call test function.
    @_start_collection is the previous timestamp which data for test has been collected.
    :return:
    '''
    global collected_data_timestamp

    gmt = time.gmtime()
    current_ts = calendar.timegm(gmt)
    print("timestamp:-", current_ts)
    # start_ts = current_ts - _start_collection

    query_time_range = "&start={}&end={}&step={}".format(collected_data_timestamp, current_ts, _step)

    # update current timestamp!
    collected_data_timestamp = current_ts

    final_query = _query + query_time_range
    result = prometheus_api.general_range_promql(final_query)
    ls_result = [[str(datetime.fromtimestamp(i[0])), i[1]] for i in
                 json.loads(result.text)['data']['result'][0]['values']]
    return ls_result


def test_VAE():
    '''
    It should be trigger with scheduler with the pre-defined interval. It need to have a separate scheduler from the
    re-training scheduler. It should decide which test function need to be trigger base on user choice(Deep learning or
     statistical model).
    :return:
    '''

    # try:

    with open(GENERAL_LOG_FILE, 'a', newline='') as f:
        f.write("test-VAE triggered!")
        f.write('------------------------')

    print('testing triggered!')
    # TODO add delay until the first training has been finished

    _config = process_config('codes/NAB_config.json')

    # TODO uncomment following statement
    test_data = read_test_data(metric_query, sampling_resolution)

    # TODO remove the following lines ( it's for testing the test procedure!)
    # save_dir = 'datasets/processed_data/'
    # dataset = _config['dataset']
    # filename = '{}.npz'.format(dataset)
    # result = dict(np.load(save_dir + filename, allow_pickle=True))
    # test_data = result['test']

    print("testing ... ")
    print(len(test_data))

    test_value = [v for k, v in test_data]
    test_timestamp = [k for k, v in test_data]

    if fake_anomaly_generator:
        for i in range(num_fake_anomalies):
            fake_anomaly_index = random.randint(0,len(test_value))
            test_value[fake_anomaly_index] = str(float(test_value[fake_anomaly_index]) * 30)
            with open('generated_fake_anomalies.csv', 'a') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([test_timestamp[fake_anomaly_index], test_value[fake_anomaly_index]])

                _start_datatime = datetime.timestamp(datetime.strptime(test_timestamp[fake_anomaly_index], "%Y-%m-%d %H:%M:%S"))
                # _start_datatime = datetime.timestamp(datetime.fromisoformat(test_timestamp[fake_anomaly_index]))
                _start_datatime = _start_datatime * 1000
                grafana_api.adding_anomalies_annotation(grafana_base_url, grafana_auth, dashboard_id, panel_id,
                                                        int(_start_datatime), int(_start_datatime) + 1, 'These are randomly generated anomalies fake-anomalies','Fake_anomaly')

    # convert test data to rolling windows and sequences
    test_windows, test_seq, test_sample_m, test_sample_std = slice_rolling_windows_and_sequences(_config,
                                                                                                 test_value)

    test_windows = np.expand_dims(test_windows, -1)
    test_seq = np.expand_dims(test_seq, -1)

    n_test_lstm = test_seq.shape[0]

    test_lstm_recons_error, test_lstm_embedding_error = np.zeros(n_test_lstm), np.zeros(n_test_lstm)

    # check the model lock
    global model_reading_lock
    if model_reading_lock is True:
        time.sleep(0.03)

    _config = process_config('codes/NAB_config.json')
    _data = load_train_data(_config)
    # change the current model for testing after new training.
    _model_vae, _lstm_nn_model = load_model(_config, default_model_vae)

    recons_win_lstm = []

    model_reading_lock = True
    for i in range(n_test_lstm):
        test_lstm_recons_error[i], test_lstm_embedding_error[i], _recons_win_lstm = evaluate_lstm_anomaly_metric_for_a_seq(
            test_seq[i],
            _model_vae,
            _lstm_nn_model,
            loaded_config)
        recons_win_lstm.append(_recons_win_lstm)
    model_reading_lock = False

    global current_threshold
    anomaly_indices = return_anomaly_idx_by_threshold(test_lstm_recons_error, current_threshold)
    print(anomaly_indices)
    print('number of tested windows is {}'.format(test_lstm_recons_error))

    print("current threshold is {}".format(current_threshold))
    print('Test start from these time-stamps {} and end at {} .'.format(test_timestamp[0],test_timestamp[-1]))
    print('The maximum of recunstruction error is: {}'.format(max(test_lstm_recons_error)))
    # save anomalies
    with open(csv_anomalies_file, 'a') as csv_file:
        writer = csv.writer(csv_file)
        for item in anomaly_indices:
            writer.writerow([test_timestamp[item]])

    # if len(anomaly_indices) > 0:
    # TODO trigger alart system

    # add one annotation for each anomaly in time test period!

    # test_lstm_recons_error[anomaly_indices]

    plot_reconstructed_signal(np.squeeze(test_seq[0][1:]), recons_win_lstm[0],test_timestamp[0])


    if len(anomaly_indices) > 0:
        test_range = "{} {}".format(test_timestamp[0], test_timestamp[-1])
        email_anomaly(len(anomaly_indices), test_range)

    for item in anomaly_indices:
        # Add reconstruction error and the threshold to anomaly description.
        desc = "reconstruction error was:{} and the threshold was:{}".format(test_lstm_recons_error[item],
                                                                             current_threshold)

        start_datatime = datetime.timestamp(datetime.strptime(test_timestamp[item], "%Y-%m-%d %H:%M:%S"))
        # start_datatime = datetime.timestamp(datetime.fromisoformat(test_timestamp[item]))
        start_datatime = start_datatime * 1000
        grafana_api.adding_anomalies_annotation(grafana_base_url, grafana_auth, dashboard_id, panel_id,
                                                int(start_datatime), int(start_datatime) + 1, desc)

    return anomaly_indices

    # except Exception as e:  # most generic exception you can catch
    #     with open(ERROR_LOG_FILE, 'a', newline='') as f:
    #         e = str(e) + '\n'
    #         f.write(e)
    #         f.write('------------------------')
def plot_reconstructed_signal(test_set_vae, output_test, start_test_time):
    input_images = np.squeeze(test_set_vae)
    decoded_images = np.squeeze(output_test)
    # n_images = 20
    # plot the reconstructed image for a shape
    fig, axs = plt.subplots(len(test_set_vae), 1, figsize=(18, 10), edgecolor='k')
    fig.subplots_adjust(hspace=.4, wspace=.4)
    axs = axs.ravel()
    for i in range(len(test_set_vae)):
      axs[i].plot(input_images[i])
      axs[i].plot(decoded_images[i])

      axs[i].grid(True)

      l_win = 24
      axs[i].set_xlim(0, l_win)
      axs[i].set_ylim(-5, 5)
      if i == 19:
        axs[i].legend(('original', 'reconstructed'))
    plt.suptitle('Reconstructed CPU usage {}'.format(start_test_time))
    savefig('./test_reconstructed_{}.pdf'.format(start_test_time))
    fig.clf()
    plt.close()

if __name__ == '__main__':
    #read_test_data()

    # temp_df = pd.read_csv('retraining_file.csv')

    # index = temp_df.iloc[temp_df.index.get_loc(datetime.fromisoformat("2022-03-15 16:19:09"), method='nearest')]
    # print(index)
    import pandas as pd
    # from datetime import datetime
    from datetime import datetime

    #print(datetime.fromisoformat("2022-03-15 16:19:09"))
    print("hello")

#     print('scheduler step')
# process_and_save_specified_dataset()
# train_statistical_model()

# r = grafana_api.adding_anomalies_annotation(grafana_base_url, grafana_auth, dashboard_id, panel_id,
#                                             time.time(), time.time() + 1)

# print(r)
# save_csv_data(test_cpu_usage_query, 1, 15)
# start_re_train_scheduler(1)
