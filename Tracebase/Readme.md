# Appliance Prediction using Tracebase Dataset

## Overview

This project focuses on predicting the energy consumption of household appliances using the Tracebase dataset. The dataset contains detailed appliance-level energy consumption data, and this project details the steps taken to preprocess, analyze, and model the data using various machine learning techniques.

### 1. Data Extraction

- **Tracebase Data**: This data is taken from https://github.com/areinhardt/tracebase/tree/master.

### 2. Data Preprocessing

Data preprocessing is a critical step to ensure the quality and usability of the dataset for modeling. It includes the following steps:

#### 2.1 Terminology

- **Switch Point**: For a time series collection \(X = \{x_1, x_2, \ldots, x_n\}\), a time instance \(t\) is a switch point if:

  1. The difference \(\delta(t) = |X(t) - X(t - 1)| > \phi_1\), a predefined threshold.
  2. The rate of change in power readings \(\delta_r(t) = \delta(t)/X(t) > \phi_2\), another threshold.

  Empirically, we found suitable thresholds to be \(\phi_1 = 30\) watts and \(\phi_2 = 0.2\) (i.e., 20%). These thresholds may be adjusted based on the specific operating environment of the appliances.

- **Steady Point**: A time point \(t\) is considered steady if \(\delta_r(t) < \phi_2\).

- **Steady Period**: Given a time series \(X\), a steady period is defined as a subsequence \(X*{t:m} = \{x*{t+1}, x*{t+2}, \ldots, x*{t+m}\}\) if all time points within it are steady, where \(t\) is a switch point and \(m\) is the length of the steady period.

#### 2.2 Appliance Footprint Definition

For a given steady period \(X\_{t:m}\) corresponding to the ON state of the appliance, the appliance footprint is defined as:

\[
X*{af} = \{X*{t:m}(i) - X\_{t:m}(i - 1) | 1 \leq i \leq m\}
\]

This calculation captures the single-order differences between consecutive data points, revealing better identifiable patterns in power consumption when the appliance is active.

#### 2.3 Footprint Extraction Steps

The extraction of appliance footprints involves the following steps:

1. **Identify Switch Points**: Detect switch points in \(X\) using thresholds \(\phi_1\) and \(\phi_2\).
2. **Identify Steady Periods**:
   - For each identified switch point \(t\), search for a steady period of length \(m\) after \(t\):
     - Let \(X*{t:m} = \{x*{t+1}, x*{t+2}, \ldots, x*{t+m}\}\) be the steady period obtained.
     - If \(X(t) - X(t + m) < 0\), then \(X\_{t:m}\) corresponds to the ON state of the appliance; otherwise, it is considered OFF state.
3. **Store Appliance Footprints**: The steady periods corresponding to ON states are used to obtain appliance footprints, which, along with their labels (appliance names), are stored as instances for further analysis.

### 3. Preparing Training Data

- **Training Dataset Creation**: The preprocessed data is split into training and testing datasets. This step is crucial for ensuring the model's performance can be evaluated accurately.

### 4. Model Training

This project employs a Deep Neural Network (DNN) architecture to predict appliance energy consumption. The DNN consists of three sequential one-dimensional convolutional layers, followed by a flatten layer and a fully connected (FC) layer, as illustrated in Figure 3.

#### 4.1 DNN Architecture Overview

- **Input Dataset**: Let 𝒟 = {𝒳, 𝒴} be the local training dataset available with a client, where 𝒳 represents the features (appliance footprints) and 𝒴 represents the labels.
- **Convolutional Layers**:
  - Three one-dimensional convolutional layers, each with 128 filters of size 1 × 1.
  - Input shape is (1, 𝑚), where 𝑚 denotes the number of data points in each instance.
- **Flatten Layer**: Converts the 3D output of the convolutional layers to 1D for the FC layer.
- **Fully Connected Layer**: Contains 𝐶 neurons, where 𝐶 is the total number of class labels in the dataset 𝒟.
- **Output**: A softmax function is applied to the output of the FC layer to yield class probabilities.

#### 4.2 Model Training

To train the DNN model, execute the following script:
````
bash
python file_name.py

````

### Noise Handling Method

This project employs an adaptive noise handling method to effectively learn from mislabeled training data in the appliance energy consumption recognition model. The method consists of three key steps:

1. **Learn Weight Parameters \( \theta \)**:

   - Train the DNN using the cross-entropy loss function on the dataset \( \mathcal{D} \) to obtain initial weight parameters \( \theta \).

2. **Estimate Label Distributions \( \mathcal{Y}\_d \)**:

   - Utilize the trained model to estimate label distributions for each instance. The model outputs probability distributions over class labels instead of relying solely on the most probable label.

3. **Optimize \( \theta \) and \( \mathcal{Y}\_d \)**:
   - Update the loss function to include label distributions using Kullback-Leibler (KL) divergence, optimizing both \( \mathcal{Y}\_d \) and \( \theta \) iteratively until convergence.

This method enhances the model's robustness against noisy labels, leading to improved performance in recognizing appliance energy consumption patterns.

## Requirements

To run the scripts in this project, you need the following Python packages:

- numpy
- pandas
- sklearn
- matplotlib
- tensorflow

You can install these packages using pip:
```
bash
pip install numpy pandas scikit-learn matplotlib tensorflow
````