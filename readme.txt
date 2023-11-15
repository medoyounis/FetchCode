1)To train the model, use the command shell and navigate to the directory, and type :
python train_model.py data_daily.csv data

2)to use the model for inference, run fetchGUIF.py
the first step is loading the model (file name ends with .h5)
then load the data (file name ends with .csv)
the code will split the data for training and testing, and it will plot the predictions.