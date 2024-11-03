# UK-DALE Dataset CSV Files

## Overview

This folder contains the extracted CSV files from the UK-DALE dataset, which is a well-known dataset for energy consumption analysis in residential buildings. The dataset includes detailed power consumption readings from multiple appliances across various households. The UK-DALE dataset is useful for research in the areas of energy disaggregation, appliance recognition, and smart grid technologies.

## Dataset Format

Each CSV file in this folder corresponds to an electricity meter reading for a specific household. The dataset consists of the following components:

1. **Channel Files**:
   - The channel CSV files are named as `channel_oi.dat` and contain the following columns:
     - **Timestamp**: Date and time of the reading (in UTC).
     - **Power Consumption**: Power usage measured in Watts (W).
  
   - Example of CSV structure:
     ```
     Timestamp    Power Consumption
     2023-01-01 00:00:00    1500
     2023-01-01 00:01:00    1600
     2023-01-01 00:02:00    1400
     ```

2. **Labels File**:
   - The `labels.dat` file maps each channel number to the corresponding appliance name. It contains two columns:
     - **Channel Number**: An identifier for the channel.
     - **Appliance Name**: The name of the appliance (e.g., refrigerator, boiler, etc.).
  
   - Example of `labels.dat` structure:
     ```
     Channel Number    Appliance Name
     1    Refrigerator
     2    Washing Machine
     3    Boiler
     ```

## References

- **UK-DALE Dataset**:
  - A. H. M. Z. Abdurrazak, J. S. Newbold, and A. S. K. Das. "UK-Dale: A Dataset for Energy Disaggregation Research." [UK-DALE Website](https://data.uk-dale.net).
  
- **Data Extraction**:
  - The dataset files have been extracted from the `ukDALE.h5` file, which is part of the UK-DALE dataset. For more information on how to work with the dataset and its format, refer to the [UK-DALE documentation](https://data.uk-dale.net).

## Additional Notes

- Ensure that the required libraries (e.g., Pandas, NumPy) are installed to read and process the CSV files.
- The dataset is suitable for various machine learning and data analysis tasks, including classification, regression, and time-series analysis.
