import pandas as pd
import os
from sklearn import preprocessing 
from collections import deque
import numpy as np
import random
import time
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization,Flatten
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
'''
We are using Four Major Cryptocurrencys
1.BitCoin 2.LiteCoin 3.Ethereum 4.BitCoin_Cash
''' 

SEQ_LEN = 60 #This is for 60 Mins
FUTURE_PRED_PERIOD = 3 #Mins
RATIO_TO_PRED = "ETH-USD"
EPOCHS = 6
BATCH_SIZE = 64
NAME_MODEL = f"{RATIO_TO_PRED}-{SEQ_LEN}-SEQ-{FUTURE_PRED_PERIOD}-PRED-{int(time.time())}"

DIR_DATA = "crypto_data/"

def classification_for_stock(current_price,future_price):
	if float(current_price) < float(future_price):
		return 1
	else:
		return 0

def preprocessing_data(df):
	df = df.drop('future',axis=1)

	for cols in df.columns:
		if cols != "target":
			df[cols] = df[cols].pct_change()
			df.dropna(inplace=True)
			df[cols] = preprocessing.scale(df[cols].values)
	df.dropna(inplace=True)
	sequential_data = []
	prev_days = deque(maxlen=SEQ_LEN)
	
	for value in df.values:
		prev_days.append([x for x in value[:-1]])
		if len(prev_days) == SEQ_LEN:
			sequential_data.append([np.array(prev_days),value[-1]])
	random.shuffle(sequential_data)
	#print(sequential_data)

	#Balancing Recurrent Neural Network sequence
	#Seperating O and 1 from Prediction Column for Feeding
	buys = []
	sells = []
	#We Can't Split time series data into a Ratio so We Loop
	#and find the Minimum(Total Value of Both)
	for seque, target in sequential_data:
		if target == 0:
			sells.append([seque, target])
		elif target == 1:
			buys.append([seque, target])
	lower_threshold = min(len(buys), len(sells))
	#Setting Up our Data over the Split considering both Total and Ratio
	buys = buys[:lower_threshold]
	sells = sells[:lower_threshold]
	#Finally we are throwing our Filtered data into out sequence
	sequential_data = buys+sells
	random.shuffle(sequential_data)

	X = []
	y = []
	for data_x, target in sequential_data:
		X.append(data_x)
		y.append(target)
	return np.array(X), y


#Importing Out Data(Saved by WEB SCRAPING)
files_name = ['BTC-USD','LTC-USD','ETH-USA','BCH-USD']
files_path = ['BTC-USD.csv','LTC-USD.csv','ETH-USD.csv','BCH-USD.csv']

df_main = pd.DataFrame()

for nums,file in enumerate(files_path):
	all_datasets = os.path.join(DIR_DATA,file)
	data1 = pd.read_csv(all_datasets,names=['time','low','high','open','close','volume'])
	data1.rename(columns={"close":f'{files_name[nums]}_close','volume':f'{files_name[nums]}_volume'},inplace=True)
	data1.set_index("time",inplace=True)
	data1 = data1[[f'{files_name[nums]}_close',f'{files_name[nums]}_volume']]
	#print(data1.head())
	if bool(df_main.empty) != False:
		df_main = data1
	else:
		df_main = df_main.join(data1)

#This is a Test Function to Make sure every thing is right
def checker_function():
	df_main['future'] = df_main[f'{RATIO_TO_PRED}_close'].shift(-FUTURE_PRED_PERIOD)
	df_main['target'] = list(map(classification_for_stock ,df_main[f'{RATIO_TO_PRED}_close'], df_main['future']))
	#print(df_main[[f'{RATIO_TO_PRED}_close','future', 'target']].head(7))
checker_function()

preprocessing_data(df_main)
time_sque = sorted(df_main.index.values)
last_5perc_data_threshold = time_sque[-int(0.05*len(time_sque))]

#We are creating Validation set for last 5% of our Data
#The rest is Main data for Neural Network

validation_df_main = df_main[(df_main.index >= last_5perc_data_threshold)]
df_main = df_main[(df_main.index < last_5perc_data_threshold)]

#Setting up Our Test and Validation data for RNN
train_data_x, train_data_y = preprocessing_data(df_main)
validation_x, validatino_y = preprocessing_data(validation_df_main)

# print(f'Training Data :{len(train_data_x)}, Validation Data: {len(validation_x)}')
# print(f"Don't Buys: {train_data_y.count(0)}, Buys: {train_data_y.count(1)}")
# print(f"Validation Don't Buys: {validatino_y.count(0)}, Validation Buys: {validatino_y.count(1)}")

#We are Going to Build the Model
rnn_model = Sequential()
rnn_model.add(LSTM(128, input_shape=(train_data_x.shape[1:]), return_sequences=True))
rnn_model.add(Dropout(0.2))
rnn_model.add(BatchNormalization())

rnn_model.add(LSTM(128,  return_sequences=True))
rnn_model.add(Dropout(0.1))
rnn_model.add(BatchNormalization())

rnn_model.add(LSTM(128,  return_sequences=True))
rnn_model.add(Dropout(0.1))
rnn_model.add(BatchNormalization())

rnn_model.add(LSTM(128, return_sequences=True))
rnn_model.add(Dropout(0.2))
rnn_model.add(BatchNormalization())
rnn_model.add(Flatten())

rnn_model.add(Dense(32,activation="relu"))
rnn_model.add(Dropout(0.2))

rnn_model.add(Dense(2,activation="softmax"))
optimizer = tf.keras.optimizers.Adam(lr=0.001,decay=1e-6)

rnn_model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
#Fitting a Model to our Data
mode_fit = rnn_model.fit(
	train_data_x,train_data_y,
	batch_size=BATCH_SIZE,epochs=EPOCHS,
	validation_data=(validation_x,validatino_y),
	callbacks=None,
	)
rnn_model.save(f'{NAME_MODEL}.h5')



