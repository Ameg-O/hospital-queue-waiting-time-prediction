# Hospital Queue Waiting-Time Prediction

## Project Overview

This project uses Machine Learning to predict the estimated waiting time of patients in a hospital queue.

It considers patient details, queue conditions, staff availability, and other relevant factors to generate waiting-time predictions.

The project also includes an interactive web application built using Gradio.

## Objectives

- Predict estimated hospital patient waiting times.
- Perform Exploratory Data Analysis (EDA).
- Handle missing values and preprocess data.
- Apply feature engineering techniques.
- Train and evaluate Machine Learning models.
- Build an interactive prediction application using Gradio.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Gradio

## Machine Learning Workflow

1. Data Loading
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Train-Test Splitting
6. Model Training
7. Hyperparameter Tuning
8. Model Evaluation
9. Gradio Application Development

## Model Used

**Random Forest Regressor**

The project uses a machine learning pipeline for preprocessing and prediction, along with GridSearchCV and TimeSeriesSplit for hyperparameter tuning and validation.

## Project Structure

```text
Hospital-Queue-Waiting-Time-Prediction/
│
├── app.py
├── Hospital_Queue_Waiting_Time_Project_Report.ipynb
├── hospital_queue_prediction_advanced_with_nulls_50000.csv
├── requirements.txt
└── README.md
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/Ameg-O/hospital-queue-waiting-time-prediction.git
```

### 2. Navigate to the project folder

```bash
cd hospital-queue-waiting-time-prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Gradio application

```bash
python app.py
```

### 5. Open the application

Open the local Gradio URL displayed in your terminal.

## Disclaimer

This project provides estimated patient waiting times based on the available dataset. Predictions are estimates and should not be used as the sole basis for clinical or hospital operational decisions.

## Author

**Ameg O**

[GitHub Repository](https://github.com/Ameg-O/hospital-queue-waiting-time-prediction)