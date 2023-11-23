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
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

def train (inputfile,outputfile):
    # Load the dataset

    reciept_data_path = "data_daily.csv"
    reciept_dataframe = pd.read_csv(reciept_data_path, usecols=[1], engine='python')

    # Plot the original dataset
    plt.figure(figsize=(8,4))
    plt.plot(reciept_dataframe, label='Original data')
    plt.legend()
    plt.show()


    ### preprocessing the dataset
    # Convert to Numpy Array and Normalize
    reciept_array = reciept_dataframe.values.astype('float32')
    scaler=MinMaxScaler(feature_range=(0, 1))
    normalized_reciept_data= scaler.fit_transform(reciept_array)
    # divide into training and testing sets
    training_rate= int(len(normalized_reciept_data)*0.80)
    train_partition, test_partition = normalized_reciept_data[0:training_rate,:], normalized_reciept_data[training_rate:len(normalized_reciept_data),:]


    # Preparing the dataset for LSTM: We then prepare the dataset for the LSTM model by reshaping it.
    """
    we create a function that converts the time series data into a format that's just right for training the LSTM model.
    For every data point in the dataset, this function does a neat trick. It grabs the count of reciepts at a specific time (let's call it "t") 
    and the count of reciept at the very next time ("t + 1"). Then, it adds these two counts into separate lists. This clever move results in a dataset 
    filled with sequences (reciept count at time "t") and their matching labels (reciept count at time "t + 1").
    The training and testing data are transformed using the function defined above. The data is then reshaped to the format expected by the LSTM layer, which is [number of samples, time steps, number of features].
    """

    def organize_data(sequence,history_length=1):
        input_data, target_data = [], []
        for i in range (len(sequence)-history_length-1):
            fragment=sequence[i:(i+history_length),0]
            input_data.append(fragment)
            target_data.append(sequence[i+history_length,0])
        return np.array(input_data),np.array(target_data)
    history_length=1
    x_train,y_train=organize_data(train_partition,history_length)
    x_test,y_test=organize_data(test_partition,history_length)

    x_train=np.reshape(x_train,(x_train.shape[0],1,x_train.shape[1]))
    x_test=np.reshape(x_test,(x_test.shape[0],1,x_test.shape[1]))
    es = EarlyStopping(monitor='loss', mode='auto', verbose=1, patience=8)
    model=Sequential()
    # dropout regularization
    model.add(keras.layers.SimpleRNN(units = 64, 
                activation = "tanh",
                return_sequences = True,
                input_shape = (1,history_length)))
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
    model.fit(x_train, y_train, epochs = 100, batch_size = 1,callbacks=[es])
    model.summary()
    model.save(outputfile+'-RNN.h5')


    # Second model
    model2 = keras.models.Sequential()
    model2.add(keras.layers.LSTM(units=64,
                    return_sequences=True,
                    input_shape=(1,history_length)))

    model2.add(keras.layers.LSTM(units=64))
    model2.add(keras.layers.Dense(32))
    model2.add(keras.layers.Dropout(0.25))
    model2.add(keras.layers.Dense(1))
    model2.compile(optimizer='adam',
    loss='mean_squared_error')
    history = model2.fit(x_train,
            y_train,
            epochs=100, batch_size = 1,callbacks=[es])
    model2.save(outputfile+'-lstm.h5')

if __name__ == "__main__":
    file_path = sys.argv[1]
    out_path = sys.argv[2]
    train(file_path, out_path)