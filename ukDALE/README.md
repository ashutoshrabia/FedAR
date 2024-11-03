# Appliance Prediction using FedAR+ on ukDALE Dataset

## Overview

This project focuses on predicting the energy consumption of various household appliances using the UK-DALE dataset. The dataset was originally in a `.h5` format, and this project details the steps taken to extract, preprocess, and analyze the data using various machine learning methods.

## Workflow

The workflow consists of several steps, each handled by specific Python scripts. Below is a brief overview of each step:

### 1. Data Extraction

- **NILMTK API**: The NILMTK (Non-Intrusive Load Monitoring Toolkit) API was utilized to extract CSV files from the UK-DALE `.h5` file. The API is specifically designed for handling energy consumption data and provides a robust method for data extraction.

### 2. CSV Conversion

- **`read_ukDale.py`**: This script converts the extracted data into CSV format. It processes the raw data and saves it in a structured manner for easier analysis.

### 3. Data Preprocessing

- **`preprocessing.py`**: After conversion, the CSV files undergo preprocessing to clean the data, remove NaN values, and format it for model training. This step ensures that the dataset is ready for analysis and modeling.

### 4. Training Data Preparation

- **`final_dataset.py`**: This script concatenates preprocessed data from multiple houses and formats it into a training dataset specifically for five appliances: fridge, dishwasher, washing machine, boiler, and kettle. 

### 5. Prediction

- **Prediction Methods**: Three different methods were employed for prediction:
  - **Decision Tree (DT)**: Implemented in `prediction_DT.py`, this method uses a decision tree classifier to predict appliance consumption.
  - **Long Short-Term Memory (LSTM)**: The script `prediction_LSTM.py` utilizes LSTM neural networks to model time-series data for more accurate predictions.
  - **Federated Averaging (FedAR+)**: The script `prediction_FedAR+.py` implements the FedAR+ algorithm, enabling federated learning for appliance energy prediction.

## Requirements

To run the scripts in this project, you need the following Python packages:

- `numpy`
- `pandas`
- `tensorflow`
- `sklearn`
- `NILMTK`

## Usage

1. **Data Extraction**: Use the NILMTK API to extract the required data from the UK-DALE `.h5` file.
2. **Convert to CSV**: Run `read_ukDale.py` to convert the extracted data to CSV files.
3. **Preprocess Data**: Execute `preprocessing.py` to clean and prepare the data.
4. **Prepare Training Data**: Use `final_dataset.py` to create the final dataset for training.
5. **Run Predictions**: Execute one of the prediction scripts (`prediction_DT.py`, `prediction_LSTM.py`, `prediction_FedAR+.py`) to make predictions based on the training data.

## Contact

For further information or access to larger datasets, please contact me at **jangirpoorab@gmail.com**.

## References

- UK-DALE Dataset: [UK-DALE Dataset](https://data.ukdale.org/)
- NILMTK Documentation: [NILMTK GitHub](https://github.com/nilmtk/nilmtk)
