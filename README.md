# Black Friday Sales Prediction System

## Project Overview

This project involves building a machine learning model to predict purchase amounts based on the Black Friday sales dataset. The system is deployed using Streamlit, providing an interactive and user-friendly interface for predictions.

## Features

- Data Preprocessing: Handling missing values, encoding categorical data, and scaling features.
- Model Training: Using a Random Forest Regressor for accurate predictions.
- Deployment: Implemented with Streamlit for an interactive interface.
- EDA: Detailed exploratory data analysis to understand data patterns and insights.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Jupyter Notebook

## Data Preprocessing

Steps taken in data preprocessing:

1. Encoding categorical features like `Gender`, `Age`, and `City_Category`.
2. Handling missing values in `Product_Category_2` and `Product_Category_3`.
3. Scaling features using `StandardScaler`.

## EDA (Exploratory Data Analysis)

The Jupyter notebook contains detailed exploratory data analysis to understand the data better. It includes:

- Visualizations of various features
- Distribution of purchase amounts
- Correlation analysis

## Model Training

A Random Forest Regressor is used for training the model. The dataset is split into training and testing sets with a ratio of 80:20.

## Deployment with Streamlit

Streamlit is used to create an interactive interface where users can input features and get purchase predictions. The application takes inputs for the following features:

- Gender
- Age
- Occupation
- Stay in Current City Years
- Marital Status
- Product Category 1
- Product Category 2
- Product Category 3
- City Category (B, C)

## How to Run the Project

1. **Clone the Repository:**
    ```bash
    git clone <repository-link>
    cd <repository-name>
    ```

2. **Install the Required Libraries:**
    ```bash
    pip install pandas numpy scikit-learn streamlit
    ```

3. **Run the Jupyter Notebook for Data Preprocessing and EDA:**
    ```bash
    jupyter notebook task2.ipynb
    ```

4. **Run the Streamlit App:**
    ```bash
    streamlit run WebApp.py
    ```

## Usage

1. Open the Streamlit app in your browser.
2. Enter the required feature values.
3. Click the 'Predict Purchase' button to get the predicted purchase amount.

## Code Structure

- `task2.ipynb`: Contains the code for data preprocessing and exploratory data analysis.
- `WebApp.py`: Contains the Streamlit application code.
- `Prediction_system.py`: Contains the code for training the Random Forest model.
