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
IMAGE_SIZE = (25, 1)
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

    df = pd.read_csv(file_path, parse_dates=True)
    df.columns=['# Date','Receipt_Count']
    df['Date']=pd.to_datetime(df['# Date'], format='%Y-%m-%d')
    df1=df.drop(['# Date'], axis=1)
    

    def date_to_day_of_year(date_string, date_format='%m/%d'):

        day_of_year = date_string.timetuple().tm_yday
    
        return day_of_year
    df1['dayOfYear'] = df1['Date'].apply(date_to_day_of_year, date_format='%m/%d/%Y')
  
    training_rate=int(0.8 * len( df1['dayOfYear'])) 
    x_train= df1['dayOfYear'][:training_rate]
    y_train=df1['Receipt_Count'][:training_rate]
    x_test= df1['dayOfYear'][training_rate:] 
    y_test=df1['Receipt_Count'][training_rate:]
    
 
    scaler = MinMaxScaler(feature_range=(-1, 1)) #0,1
    y_train = scaler.fit_transform(np.reshape(y_train,((-1, 1))))
    x_test=np.expand_dims(x_test,axis=-1)
    x_test=np.expand_dims(x_test,axis=-1)

    y_test_normalized = scaler.fit_transform(np.reshape(y_test,((-1, 1))))

    predictions_normalized = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions_normalized)

    print(predictions)
    # Clear the previous figure
    for widget in image_frame.winfo_children():
        widget.destroy()

    # Create a figure to plot
    fig = Figure(figsize=(6, 4), dpi=100)
    plot = fig.add_subplot(1, 1, 1)

    # Plot the predictions and ground truth
    plot.plot(np.squeeze(x_test), np.array(y_test), label='Ground Truth', color='blue')#.argsort() #np.arange(len(y_test))
    plot.plot(np.squeeze(x_test), np.array(predictions).flatten(), label='Predictions', color='red')#.argsort()
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
