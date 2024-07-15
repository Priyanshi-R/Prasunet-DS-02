import numpy as np;
import streamlit as st;
import pickle;

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

def main():
    
    # Streamlit app interface
    st.title('Black Friday Sales Prediction')

    # Get user input
    Gender = st.text_input('Gender (0 for Female, 1 for Male)')
    Age = st.text_input('Age (1 for 0-17, 2 for 18-25, 3 for 26-35, 4 for 36-45, 5 for 46-50, 6 for 51-55, 7 for 55+)')
    Occupation = st.text_input('Occupation')
    Stay_In_Current_City_Years = st.text_input('Stay In Current City Years')
    Marital_Status = st.text_input('Marital Status (0 for Single, 1 for Married)')
    Product_Category_1 = st.text_input('Product Category 1')
    Product_Category_2 = st.text_input('Product Category 2')
    Product_Category_3 = st.text_input('Product Category 3')
    City_B = st.text_input('City B (0 or 1)')
    City_C = st.text_input('City C (0 or 1)')

    # Predict button
    if st.button('Predict Purchase Amount'):
    # Prepare input data
        input_data = [
            int(Gender), int(Age), int(Occupation), int(Stay_In_Current_City_Years), int(Marital_Status), 
            int(Product_Category_1), float(Product_Category_2), float(Product_Category_3), int(City_B), int(City_C)
        ]

        # Make prediction
        predicted_purchase = predict_purchase(input_data)
    
        # Display the prediction
        st.write(f'Predicted Purchase Amount: {predicted_purchase:.2f}')

    
if __name__ == '__main__':
    main()
