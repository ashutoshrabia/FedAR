import os
import pandas as pd
import numpy as np

# Threshold values
phi1 = 30  # in watts
phi2 = 0.2  # 20%

def load_tracebase_data(directory_path):
    all_data = []
    labels = []

    complete_path = directory_path
    for appliance_type in os.listdir(complete_path):
        appliance_path = os.path.join(complete_path, appliance_type)
        
        if os.path.isdir(appliance_path):
            for file in os.listdir(appliance_path):
                file_path = os.path.join(appliance_path, file)
                if file.endswith('.csv'):
                    data = pd.read_csv(file_path, delimiter=';', header=None)
                    data.columns = ['Timestamp', 'Power_1s', 'Power_8s']
                    data['Power'] = data['Power_1s']
                    #data['Timestamp'] = pd.to_datetime(data['Timestamp'],format='mixed') #, format='%d/%m/%Y %H:%M:%S'
                    data['Timestamp'] = pd.to_datetime(data['Timestamp'],format='mixed', errors='coerce')
                    data = data.sort_values(by='Timestamp')
                    
                    if not data.empty:
                        all_data.append(data)
                        labels.append(appliance_type)

    return all_data, labels

def identify_switch_points(data):
    switch_points = []
    
    
    for t in range(1, len(data)):
        delta = abs(data['Power'].iloc[t] - data['Power'].iloc[t - 1])
        delta_r = delta / data['Power'].iloc[t] if data['Power'].iloc[t] != 0 else 0
        
        if delta > phi1 and delta_r > phi2:
            switch_points.append(t)
        
    return switch_points

def extract_steady_period(data, start_index):
    m = 0
    while start_index + m + 1 < len(data) - 1:
        # Calculate delta_r for the period after the switch point
        current_power = data['Power'].iloc[start_index + m]
        next_power = data['Power'].iloc[start_index + m + 1]
        
        # Handle division by zero by skipping or setting delta_r to 0
        if next_power != 0:
            delta_r = abs(next_power - current_power) / next_power
        else:
            delta_r = 0  # If next_power is zero, set delta_r to zero to continue checking
        
        # Check if the relative change exceeds the threshold
        if delta_r >= phi2:
            break
        m += 1
    
    return m

def extract_appliance_footprints(data, switch_points):
    footprints = []
    
    for t in switch_points:
        m = extract_steady_period(data, t)
        #print(m)
        if m > 0 and t + m < len(data) and data['Power'].iloc[t] - data['Power'].iloc[t + m] < 0:
            footprint = np.diff(data['Power'].iloc[t:t + m + 1].values)
            #print(footprint)
            footprints.append(footprint)
    #print(footprints)
    return footprints

def preprocess_tracebase_data(directory_path):
    all_data, labels = load_tracebase_data(directory_path)
    dataset_X = []
    dataset_Y = []
    
    for i, (data, label) in enumerate(zip(all_data, labels)):
        #print(f"Processing appliance {label} with {len(data)} data points (#{i + 1}/{len(all_data)})...")
        switch_points = identify_switch_points(data)
        #print(switch_points)
        footprints = extract_appliance_footprints(data, switch_points)
        
        for footprint in footprints:
            dataset_X.append(footprint)
            dataset_Y.append(label)
    
    df = pd.DataFrame({
        'Footprint': [list(fp) for fp in dataset_X],
        'Label': dataset_Y
    })
    return df

# Example usage
directory_path = r'C:\Users\niles\OneDrive\Desktop\tracebase\australia'

print(f"Loading data from: {directory_path}")
df = preprocess_tracebase_data(directory_path)

if not df.empty:
    output_file = 'processed_dataset2.csv'
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
else:
    print("No data was processed. The output CSV will be empty.")
