#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 15:17:04 2018

@author: pdatta
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()
df = pd.read_excel('/home/pdatta/Downloads/test_dexma_Data/hourly_cosumption_test.xls', sheet_name=None)
#set the week number, month and year 
df['week'] = df['Date'].dt.week
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
# reformat the week number (because of continutation of 2015 week)
for i in range(len(df.iloc[0:100])):
    if ( df['month'][i] == 1 and df['week'][i] == 53 ):
        df['week'][i] = 0
# week 0 is not taken into consideration as it being the last week of 2015, whose data is incomplete 
df1 = df.groupby(['year','week'])['Main Load [kWh]'].sum()
#making plotfor each year
df2 = df1.fillna(0).swaplevel()
# plot data
fig, ax = plt.subplots(figsize=(15,7))
# use unstack()
df2.unstack().plot(ax=ax)
ax.set_ylabel('Load in KWH per week')
ax.set_xlabel('Week number')
# maximum and minimu value
max_dta = df[df['Main Load [kWh]']==df['Main Load [kWh]'].max()]
min_dta = df[df['Main Load [kWh]']==df['Main Load [kWh]'].min()]


# we will use an LSTM to predict the time series forecasting( its simple and effective)
dataset_train = df.iloc[:17544,2:3].values

#feature scaling
from sklearn.preprocessing import MinMaxScaler # we are doing normalization not standarization
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(dataset_train)
# creating a data structure with 72 timesteps and 1 output using moving window
x_train=[]
y_train=[]
for i in range(72,len(dataset_train)): # because the previous 72 are stored as initial and the rest are test dataset
    x_train.append(training_set_scaled[i-72:i,0])
    y_train.append(training_set_scaled[i,0])
x_train, y_train = np.array(x_train),np.array(y_train)
#reshaping the training data
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
#building RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout # to prevent overfitting

#initialize the RNN
regressor = Sequential()
# stacked lSTM layers
#adding first LSTM layer and Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True, 
                   input_shape = (x_train.shape[1],1))) #units correspond to number of neurons inside LSTM

regressor.add(Dropout(0.2))
#adding second LSTM layer and Dropout regularization
regressor.add(LSTM(units = 150, return_sequences = True))
regressor.add(Dropout(0.2))
#adding third LSTM layer and some dropout reguralization
regressor.add(LSTM(units = 150, return_sequences = True))
regressor.add(Dropout(0.2))
#adding a fourth LSTM layer and some dropout regularization
regressor.add(LSTM(units =150))
regressor.add(Dropout(0.2))

#adding the output
regressor.add(Dense(units=1))

#compiling the RNN
regressor.compile(optimizer ='adam', loss='mean_squared_error' )
#fitting the regression
regressor.fit(x_train,y_train, epochs=100, batch_size = 48)

#making the predictions and visualising the results for 2018 Load of electricity 
dataset_test = df.iloc[17544:,2:3].values
real_load= dataset_test
dataset_total = np.concatenate((dataset_train, dataset_test), axis = 0)
inputs = dataset_total[len(dataset_total)-len(dataset_test) - 72:len(dataset_total)] # subtracting the 72 days of zeroth week
inputs = inputs.reshape(-1,1)

inputs = sc.transform(inputs) #scaling the inputs
 
x_test=[]
for i in range(72,len(inputs)): # because the previous 72 are stored as initial ( only 20 financial days for test)
    x_test.append(inputs[i-72:i,0])
x_test = np.array(x_test)

#reshaping
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1)) # 3D format for input test
predicted_load = regressor.predict(x_test)
predicted_load= sc.inverse_transform(predicted_load)
#put it in the datframe to group by weeks
df_predict= pd.DataFrame({'Load':predicted_load[:,0]})
temp =df.iloc[17544:,3:4].values
t1=pd.DataFrame({'week number':temp[:,0]})
df_predict['week number']=t1.values
df_real=pd.DataFrame({'Load':real_load[:,0]})
df_real['week']=pd.DataFrame({'week number':temp[:,0]}).values
#visualization
plt.plot(real_load, color = 'red', label = 'Real electric load')
plt.plot(predicted_load, color = 'blue', label = 'Predicted electric load')
plt.title('Load prediction ')
plt.xlabel('Time in units')
plt.ylabel('Total electric load prediction')
plt.legend()
plt.show()
#plot visualization in terms of weeks
plt.plot(df_real.groupby(['week'])['Load'].sum(),color='red', label='Real Electric load')
plt.plot(df_predict.groupby(['week number'])['Load'].sum(),color='blue', label='predicted Electric load')
ax.set_ylabel('Load in KWH per week')
ax.set_xlabel('Week number')
