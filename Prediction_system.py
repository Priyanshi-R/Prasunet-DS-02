import pickle;
import numpy as np;
import pandas as pd;
import sklearn;

# Load the trained model
with open('C:/Users/KIIT/Desktop/PrasunetTask2/random_forest_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Load the scaler
with open('C:/Users/KIIT/Desktop/PrasunetTask2/standard_scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

# Define a function to make predictions
def predict_purchase(input_data):
    input_array = np.asarray(input_data).reshape(1, -1) 
    input_scaled = loaded_scaler.transform(input_array)  
    prediction = loaded_model.predict(input_scaled)
    return prediction[0]

input_data = [0, 3, 10, 2, 0, 3, 8, 16, 0, 0]  

predicted_purchase = predict_purchase(input_data)
print(f'Predicted Purchase Amount: {predicted_purchase:.2f}')