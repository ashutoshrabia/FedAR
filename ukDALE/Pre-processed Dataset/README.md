# Pre-Processed Dataset

## Overview

This repository contains a pre-processed version of the dataset derived from the UK-DALE project. The dataset is specifically formatted for machine learning tasks, including energy disaggregation and appliance recognition. Pre-processing steps have been applied to ensure that the data is clean, structured, and ready for analysis.

## Dataset Description

The pre-processed dataset consists of sequences representing power consumption readings from various appliances in a residential setting. The data is structured to facilitate machine learning modeling.

### Features

- **Footprint**: Each entry contains a sequence of power consumption values.
- **Appliance**: Each sequence is associated with an appliance representing the corresponding power consumption.

### Format

The dataset is stored in a CSV file with the following columns:

- **footprint**: A list of power consumption readings as numerical values.
- **appliance**: The appliance name (e.g., 'boiler', 'refrigerator', etc.).

**Example of the CSV structure**:

```
footprint,appliance
[1500, 1600, 1400, ...],refrigerator
[1200, 1300, 1250, ...],boiler
```

## Pre-Processing Steps

The following steps were undertaken to prepare the dataset:

1. **Data Cleaning**:
   - Removal of any corrupted or malformed entries.
   - Handling of missing values by discarding or imputing data as necessary.

2. **Data Transformation**:
   - Conversion of power consumption readings into a structured list format.
   - Padding sequences to ensure uniform length for modeling purposes.

3. **Appliance Encoding**:
   - Appliances were transformed into a numerical format suitable for machine learning algorithms.

## Usage

1. **Requirements**:
   - Ensure that the required libraries such as `pandas`, `numpy`, and `tensorflow` are installed in your Python environment.

   ```bash
   pip install pandas numpy tensorflow
   ```

2. **Loading the Dataset**:
   - You can load the pre-processed dataset using `pandas` in Python:

   ```python
   import pandas as pd

   data = pd.read_csv('path/to/pre_processed_dataset.csv')
   ```

3. **Model Training**:
   - This dataset is ready for training machine learning models. Use appropriate modeling techniques based on the specific requirements of your analysis.

## Contact Information

For any inquiries or requests for additional information regarding the dataset, please reach out to:

**Email**: [jangirpoorab@gmail.com](mailto:jangirpoorab@gmail.com)

## Note

Currently, the dataset provided includes data for only **house_3**. Other houses cannot be shared due to their large size. For detailed datasets, kindly email the contact provided above.

## References

- **UK-DALE Dataset**:
  - A. H. M. Z. Abdurrazak, J. S. Newbold, and A. S. K. Das. "UK-Dale: A Dataset for Energy Disaggregation Research." [UK-DALE Website](https://data.uk-dale.net).
  
- **Data Pre-Processing Techniques**:
  - Refer to standard data science practices and documentation for preprocessing methods utilized in this dataset.
