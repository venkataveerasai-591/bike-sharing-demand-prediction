# Bike Sharing Demand Prediction

A Machine Learning project that predicts the total number of bike rentals using the **Bike Sharing Dataset**. The project uses **Polynomial Regression** to model the relationship between bike rental demand and different environmental, seasonal, and time-related features.

## Project Overview

Bike-sharing systems generate data containing information about weather, seasons, working days, hours, and bike rental counts.

The goal of this project is to build a regression model that predicts the total number of bike rentals (`cnt`) based on the available features.

### Dataset

The project uses the `hour.csv` dataset from the Bike Sharing Dataset.

The target variable is:

* `cnt` — Total number of bike rentals

The columns `casual` and `registered` are removed because they directly contribute to the total rental count.

## Features Used

The original date column `dteday` is converted into useful date-based features:

* `day`
* `month`
* `year`

Other available features from the dataset are also used for prediction.

The following columns are removed from the input features:

```text
cnt
casual
registered
dteday
```

## Machine Learning Algorithm

The project uses:

**Polynomial Regression with Degree 5**

Polynomial features are generated using Scikit-learn's `PolynomialFeatures`.

The workflow is:

```text
Dataset
   ↓
Data Preprocessing
   ↓
Date Feature Extraction
   ↓
Train/Test Split
   ↓
Polynomial Features (Degree 5)
   ↓
Linear Regression
   ↓
Model Evaluation
```

## Libraries Used

* Python
* NumPy
* Pandas
* Scikit-learn

## Model Evaluation

The model is evaluated using the **R² (R-squared) score**.

Current results:

```text
Training R² Score: 0.6268834980305142
Testing R² Score: 0.6125457933509004
```

### Interpretation

The testing R² score is approximately:

**0.613**

This means the model explains about **61.3% of the variation** in bike rental demand on the test data.

The training and testing scores are relatively close, indicating that the model does not show a large train-test performance gap.

## Project Structure

```text
bike-sharing-demand-prediction/
│
├── hour.csv
├── bike_sharing_prediction.py
├── README.md
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/bike-sharing-demand-prediction.git
```

Move into the project directory:

```bash
cd bike-sharing-demand-prediction
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Requirements

Create a `requirements.txt` file containing:

```text
numpy
pandas
scikit-learn
```

## Running the Project

Run the Python script:

```bash
python bike_sharing_prediction.py
```

The program will preprocess the dataset, train the Polynomial Regression model, and display the training and testing R² scores.

## Future Improvements

Possible improvements for this project include:

* Compare Linear Regression with Polynomial Regression
* Try different polynomial degrees
* Test Random Forest Regression
* Try Gradient Boosting models
* Perform hyperparameter tuning
* Add MAE and RMSE evaluation
* Perform feature importance analysis
* Create visualizations of actual vs predicted rentals
* Use cross-validation
* Experiment with XGBoost or LightGBM

## Conclusion

This project demonstrates how machine learning can be used to predict bike rental demand from historical and environmental data.

The Polynomial Regression model achieved a testing R² score of approximately **0.613**, providing a useful baseline for further model improvement.

## Author

**Your Name**

GitHub: https://github.com/venkataveerasai-591
