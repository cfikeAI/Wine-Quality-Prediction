# Wine Quality Prediction Using K-Nearest Neighbors

## Overview
This project applies K-Nearest Neighbors (KNN) classification to predict wine quality based on various chemical properties.  
The dataset is preprocessed using MinMaxScaler, and different vineyard samples are compared for quality assessment.

## Features
- Scales numerical features using `MinMaxScaler`
- Predicts wine quality using `KNeighborsClassifier`
- Works with a real-world wine data (CSV dataset)
- Easily modifiable for further experimentation

## Dataset
The dataset consists of 11 chemical features (e.g., acidity, pH, alcohol content, etc) and a quality score as the target variable.  
- `winedata.csv` contains wine samples with known quality ratings.

## Setup & Installation

### 1. Clone the Repository
git clone https://github.com/cfikeIT/Wine-Quality-Prediction.git
cd Wine-Quality-Prediction
