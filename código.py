import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
from fbprophet import Prophet

df = pd.read_csv("../input/corn2015-2017/corn2013-2017.txt",sep=',',header=None, names=['date','price'])
df.head()

df.info()
df['date'] = pd.to_datetime(df['date'])
df.head()

import plotly.express as px

fig = px.line(df, x="date", y="price", title='Price Time Series')
fig.show();
df = df.rename(columns={'date':'ds', 'price':'y'})

model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

forecast.head()
fig1 = model.plot(forecast)
fig2 = model.plot_components(forecast)


#Statistical Time Series Model
!pip install pmdarima
from statsmodels.tsa.arima_model import ARMA,ARMAResults,ARIMA,ARIMAResults
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf 
from pmdarima import auto_arima 
from statsmodels.tsa.statespace.sarimax import SARIMAX

df = pd.read_csv("../input/corn2015-2017/corn2013-2017.txt",sep=',',header=None, names=['date','price'])
df.head()

df['date'] = pd.to_datetime(df['date'])

df.set_index("date",inplace=True)

df.index.freq='W'

df['price'].plot(figsize=(12,5));

adf_test(df['price'])
auto_arima(df['price'],seasonal=True).summary()
df.shape

train = df.iloc[:62]
test = df.iloc[62:]

model = SARIMAX(train['price'],order=(0,1,1))
results = model.fit()
results.summary()

start=len(train)
end=len(train)+len(test)-1
predictions = results.predict(start=start, end=end, dynamic=False, typ='levels').rename('SARIMAX(0,1,1) Predictions')

model = SARIMAX(df['price'],order=(0,1,1))
results = model.fit()
fcast = results.predict(len(df),len(df)+11,typ='levels').rename('SARIMA(0,1,1) Forecast')


title = 'Weekly Corn Price Prediction'
ylabel='Price'
xlabel='Date'

ax = df['price'].plot(legend=True,figsize=(12,6),title=title)
fcast.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel);

#LSTM
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("../input/corn2015-2017/corn2013-2017.txt",sep=',',header=None, names=['date','price'])
df.head()

df['date'] = pd.to_datetime(df['date'])

df = df.iloc[:,1].values
plt.plot(df)
plt.xlabel("date")
plt.ylabel("price")
plt.title("weekly corn price")
plt.show()

df = df.reshape(-1,1)
df = df.astype("float32")
df.shape

# scaling 
scaler = MinMaxScaler(feature_range=(0, 1))
df = scaler.fit_transform(df)

train_size = int(len(df) * 0.75)
test_size = len(df) - train_size
train = df[0:train_size,:]
test = df[train_size:len(df),:]
print("train size: {}, test size: {} ".format(len(train), len(test)))


time_stamp = 10

dataX = []
dataY = []

for i in range(len(train)-time_stamp-1):
    a = train[i:(i+time_stamp), 0]
    dataX.append(a)
    dataY.append(train[i + time_stamp, 0])
    
trainX = np.array(dataX)
trainY = np.array(dataY)  

dataX = []
dataY = []
for i in range(len(test)-time_stamp-1):
    a = test[i:(i+time_stamp), 0]
    dataX.append(a)
    dataY.append(test[i + time_stamp, 0])
testX = np.array(dataX)
testY = np.array(dataY)  

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential()

model.add(LSTM(10, input_shape=(1, time_stamp))) # 10 lstm neuron
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(trainX, trainY, epochs=50, batch_size=1)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shifting train
trainPredictPlot = np.empty_like(df)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[time_stamp:len(trainPredict)+time_stamp, :] = trainPredict

# shifting test
testPredictPlot = np.empty_like(df)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(time_stamp*2)+1:len(df)-1, :] = testPredict

plt.plot(scaler.inverse_transform(df))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
