# CO2 Emission Prediction using Simple Linear Regression

This repository contains a Jupyter Notebook for predicting CO2 emissions from vehicles using simple linear regression. The dataset used for training and testing the model is "FuelConsumption.csv".

## Dataset Information

The dataset contains information about various vehicles including their engine size, cylinders, fuel consumption, and CO2 emissions.

## Simple Linear Regression

Simple linear regression is a statistical method used to model the relationship between a single independent variable (feature) and a dependent variable (target). In this project, we use simple linear regression to predict CO2 emissions based on engine size and fuel consumption.

## Dependencies

- pandas
- numpy
- matplotlib
- scikit-learn
- Jupyter Notebook

## Usage

1. Clone the repository:

```
git clone https://github.com/your_username/Co2Emission_prediction-Simple-Linear-Regression.git
```

2. Install the required dependencies:

```
pip install -r requirements.txt
```

3. Open and run the Jupyter Notebook:

```
jupyter notebook co2Emission_prediction(simple_Linear_Regression).ipynb
```

## Description

- The notebook first loads the dataset and explores its structure.
- Data visualization is performed to understand the relationship between different features and CO2 emissions.
- The dataset is split into training and testing sets.
- Simple linear regression models are trained for predicting CO2 emissions based on engine size and fuel consumption.
- Model evaluation metrics such as mean absolute error, mean squared error, and R-squared score are calculated.
- Finally, the trained models are used to predict CO2 emissions for new input values.

## Kaggle Link

You can find this project on Kaggle: [CO2 Emission Prediction using Simple Linear Regression](https://www.kaggle.com/code/saragadamsaiprasad/co2emission-prediction-simple-linear-regression/)

## Dataset

The dataset used for this project can be downloaded from [FuelConsumption Dataset](https://www.kaggle.com/datasets/saragadamsaiprasad/fuelconsumption/).

## Files

- `co2Emission_prediction(simple_Linear_Regression).ipynb`: Jupyter Notebook for CO2 emission prediction using simple linear regression.
- `FuelConsumption.csv`: Dataset containing vehicle information.

## Results

The trained models achieved the following performance metrics:

### Engine Size - CO2 Emission Model
- Mean Absolute Error: 24.08
- Mean Squared Error (MSE): 988.61
- R-squared Score: 0.75

### Fuel Consumption - CO2 Emission Model
- Mean Absolute Error: 21.53
- Mean Squared Error (MSE): 837.49
- R-squared Score: 0.79

