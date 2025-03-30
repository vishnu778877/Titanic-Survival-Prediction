# Titanic Survival Prediction

## Overview
This project builds a machine learning model to predict whether a passenger survived the Titanic disaster based on available passenger data.

## Dataset
The dataset contains the following features:
- `PassengerId`: Unique identifier for each passenger
- `Survived`: Target variable (1 = survived, 0 = did not survive)
- `Pclass`: Ticket class (1st, 2nd, 3rd)
- `Name`: Passenger name
- `Sex`: Gender
- `Age`: Age of the passenger (some missing values)
- `SibSp`: Number of siblings/spouses aboard
- `Parch`: Number of parents/children aboard
- `Ticket`: Ticket number
- `Fare`: Price of the ticket (some missing values)
- `Cabin`: Cabin number (many missing values)
- `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Steps Included
1. **Data Preprocessing**
   - Handling missing values (Age, Fare, Cabin)
   - Encoding categorical variables (Sex, Embarked)
   - Dropping unnecessary columns (Cabin, Name, Ticket)
   - Normalizing numerical features (Age, Fare)
2. **Feature Selection**
3. **Train-Test Split**
4. **Model Training** using `RandomForestClassifier`
5. **Model Evaluation**
   - Accuracy, Precision, Recall, F1 Score
   - Confusion Matrix Visualization
6. **Saving the Model**

## Installation
To run this project, install the required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## Running the Code
Run the Jupyter Notebook or Python script to:
- Load and preprocess the dataset
- Train and evaluate the model
- Save the trained model

## Model Performance
The model is evaluated using various metrics, and the trained model is saved as `titanic_survival_model.pkl`.

## Repository Structure
```
|-- Titanic-Survival-Prediction
    |-- titanic_survival.ipynb  # Jupyter Notebook with code
    |-- tested.csv              # Dataset
    |-- README.md               # Project documentation
```
