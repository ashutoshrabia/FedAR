# Final Dataset for Appliance Power Consumption

## Overview

This repository contains the `final_dataset`, which is a comprehensive collection of preprocessed power consumption data for five different houses. The dataset focuses on five specific appliances: fridge, dishwasher, washing machine, boiler, and kettle. 

The data has been cleaned, with NaN values removed to ensure high-quality inputs for further analysis or machine learning tasks.

## Dataset Details

- **Number of Houses**: 5
- **Appliances Included**:
  - Fridge
  - Dishwasher
  - Washing Machine
  - Boiler
  - Kettle

### Data Structure

The dataset is stored in CSV format. Each row represents power consumption readings for a specific appliance at a given timestamp. The columns include:

- **Timestamp**: The time at which the reading was taken.
- **Appliance**: The name of the appliance (e.g., fridge, dishwasher, etc.).
- **Power Consumption**: The measured power consumption for that appliance (in watts).

### Preprocessing Steps

1. **Concatenation**: Data from all five houses has been combined into a single dataset.
2. **NaN Removal**: All rows containing NaN values have been removed to ensure the integrity of the dataset.

## Usage

This dataset is intended for research and development in the field of appliance energy consumption analysis, load forecasting, and machine learning applications. It can be used to build predictive models, analyze consumption patterns, and explore energy-saving opportunities.

## References

- UK-DALE Dataset: [UK-DALE Dataset](https://data.ukdale.org/)
- Relevant literature on energy consumption analysis and forecasting.
