# Diabetes Prediction

This project utilizes machine learning techniques to predict the likelihood of diabetes in patients based on various health parameters.

## Dataset

The dataset is sourced from the from the web for project purpose.
Datasetcontain
- 100,000 patient records  
- Features: age, gender, BMI, blood glucose, HbA1c, hypertension, heart disease, smoking history  
- Target: Diabetes diagnosis (0 = No, 1 = Yes)

## Approach

- Exploratory Data Analysis (EDA)
- Feature Engineering (scaling, encoding, handling class imbalance with SMOTE)
- Model Training: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, AdaBoost
- Evaluation: Accuracy, Precision, Recall, F1-score

## Results

**Best Model:** The Random Forest model achieved the highest accuracy of 96% 
**F1 Score:** 0.76  
**Key Features:** HbA1c, Glucose, BMI, Age

## Recommendations

- Integrate with EHR for real-time risk scoring  
- Target high-BMI or hypertensive patients for screening  
- Retrain model quarterly with new data

## Modeling

Implemented models include Logistic Regression, Random Forest, and Support Vector Machines.


## Usage

To run the notebook:
1. Install the required packages listed in `requirements.txt`.
2. Open and execute `diabetes_prediction.ipynb`.
