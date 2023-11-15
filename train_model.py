# Loading Necessary Libraries
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler #only used sklearn for preprocessing
import numpy as np
from keras.callbacks import EarlyStopping
### loading the libraries to build the RNN network
from keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import mean_squared_error
from datetime import datetime
import sys

def train (inputfile,outputfile):
    #reading the data using pandas library
    df = pd.read_csv(inputfile,  parse_dates=True)
    df.columns=['# Date','Receipt_Count']


    df['Date']=pd.to_datetime(df['# Date'], format='%Y-%m-%d')
    print(pd.isnull(df['Receipt_Count']).sum()) #making sure we do not have null values
    df1=df.drop(['# Date'], axis=1)#removing the '# Date' column

    df1 = df1.reindex(columns=['Date', 'Receipt_Count'])#reordering the columns


    #Function to convert the date
    def date_to_day_of_year(date_string, date_format='%m/%d'):

        day_of_year = date_string.timetuple().tm_yday
        
        return day_of_year

    df1['dayOfYear'] = df1['Date'].apply(date_to_day_of_year, date_format='%m/%d/%Y')


    ### splitting the dataset for training and testing

    x_train= df1['dayOfYear'][:340]#pd.to_datetime()
    y_train=df1['Receipt_Count'][:340]
    x_test= df1['dayOfYear'][340:] #pd.to_datetime(
    y_test=df1['Receipt_Count'][340:]


    ### normaizing the values of the dataset for tanh activation function
    scaler = MinMaxScaler(feature_range=(0, 1)) #tanh takes data between -1 and 1
    y_train = scaler.fit_transform(np.reshape(y_train,((-1, 1))))
    x_train=np.expand_dims(x_train,axis=-1)
    x_train=np.expand_dims(x_train,axis=-1)


    es = EarlyStopping(monitor='loss', mode='auto', verbose=1, patience=8)
    model = Sequential()
    
    # dropout regularization
    model.add(keras.layers.SimpleRNN(units = 64, 
                            activation = "tanh",
                            return_sequences = True,
                            input_shape = (1,1)))
    model.add(keras.layers.Dropout(0.2))
    
    model.add(keras.layers.SimpleRNN(units = 64, 
                            activation = "tanh",
                            return_sequences = True))
    
    model.add(keras.layers.SimpleRNN(units = 64,
                            activation = "tanh",
                            return_sequences = True))
    
    model.add( keras.layers.SimpleRNN(units = 64))
    
    # adding the output layer
    model.add(keras.layers.Dense(units = 32,activation='sigmoid'))
    model.add(keras.layers.Dense(units = 16,activation='sigmoid'))
    model.add(keras.layers.Dense(units = 1,activation='sigmoid'))
    
    # compiling RNN
    model.compile(optimizer = SGD(learning_rate=0.001,
                                    decay=1e-6, 
                                    momentum=0.9, 
                                    nesterov=True), 
                    loss = "mean_squared_error")
    
    # fitting the model
    model.fit(x_train, y_train, epochs = 50, batch_size = 1,callbacks=[es])
    model.summary()
    model.save(outputfile+'-RNN.h5')


    # Second model
    model2 = keras.models.Sequential()
    model2.add(keras.layers.LSTM(units=64,
                                return_sequences=True,
                                input_shape=(1,1)))
    
    model2.add(keras.layers.LSTM(units=64))
    model2.add(keras.layers.Dense(32))
    model2.add(keras.layers.Dropout(0.25))
    model2.add(keras.layers.Dense(1))
    model2.compile(optimizer='adam',
                loss='mean_squared_error')
    history = model2.fit(x_train,
                        y_train,
                        epochs=50, batch_size = 1,callbacks=[es])
    model2.save(outputfile+'-lstm.h5')

if __name__ == "__main__":
    file_path = sys.argv[1]
    out_path = sys.argv[2]
    train(file_path, out_path)