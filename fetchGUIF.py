import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime

# Assuming the model expects images of this size

model = None  # Initially, no model is loaded

def load_model():
    global model
    model_path = filedialog.askopenfilename(title="Load TensorFlow Model", filetypes=[("Model Files", "*.h5")])
    if not model_path:
        return
    model = tf.keras.models.load_model(model_path)
    status_label.config(text="Model Loaded Successfully!")

def open_data_and_predict():
    global model
    if not model:
        status_label.config(text="Please load a model first!")
        return

    file_path = filedialog.askopenfilename(title="load data", filetypes=[("CSV Files", "*.csv")])

    if not file_path:
        return

    reciept_dataframe = pd.read_csv(file_path, usecols=[1], engine='python') 
    # Convert to Numpy Array and Normalize
    reciept_array = reciept_dataframe.values.astype('float32')
    scaler=MinMaxScaler(feature_range=(0, 1))
    normalized_reciept_data= scaler.fit_transform(reciept_array)
    # divide into training and testing sets
    training_rate= int(len(normalized_reciept_data)*0.80)
    train_partition, test_partition = normalized_reciept_data[0:training_rate,:], normalized_reciept_data[training_rate:len(normalized_reciept_data),:]
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

    train_forecast = model.predict(x_train)
    test_forecast = model.predict(x_test)

    train_forecast = scaler.inverse_transform(train_forecast)
    train_target = scaler.inverse_transform([y_train])
    test_forecast = scaler.inverse_transform(test_forecast)
    test_target = scaler.inverse_transform([y_test])

    print(test_forecast)
    # Clear the previous figure
    for widget in image_frame.winfo_children():
        widget.destroy()

    # Create a figure to plot
    fig = Figure(figsize=(8, 4), dpi=100)
    plot = fig.add_subplot(1, 1, 1)

    # Plot the predictions and ground truth
    plot.plot(scaler.inverse_transform(normalized_reciept_data), label='Ground Truth', color='blue')#.argsort() #np.arange(len(y_test))
    plot.plot([item for item in train_forecast], label='training forecast', color='green')#.argsort()
    plot.plot([item+len(train_forecast) for item in range(len(test_forecast))], test_forecast ,label='test_forecast', color='red')#.argsort()
    plot.set_title('Predictions vs Ground Truth')
    plot.legend()

    # Embed the plot into the Tkinter GUI
    canvas = FigureCanvasTkAgg(fig, master=image_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

app = tk.Tk()
app.title("Time Series Prediction")

# Create main frame
main_frame = tk.Frame(app)
main_frame.pack(padx=10, pady=10)

# Button to load model
load_model_button = tk.Button(main_frame, text="Load Model", command=load_model)
load_model_button.pack(pady=10)

# Button to open the image
open_button = tk.Button(main_frame, text="Load Data and Predict", command=open_data_and_predict)
open_button.pack(pady=10)

# Label to display model status
status_label = tk.Label(main_frame, text="", font=("Arial", 12))
status_label.pack(pady=10)

# Frame to display the plot
image_frame = tk.Frame(main_frame)
image_frame.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

app.mainloop()
