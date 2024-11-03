import numpy as np
import pandas as pd

final_df = []

for i in range(1, 6):
    print(f"Appending House {i}...")
    csv_name = f"train_house_{i}.csv"
    data = pd.read_csv(csv_name)
    data = data.dropna(subset=['footprint'])
    data = data[data['appliance'].isin(['Fridge', 'Dish washer', 'Washing machine', 'Kettle', 'Boiler'])]
    final_df.append(data)
  
final_df = pd.concat(final_df, ignore_index=False)
final_df.to_csv('train_data.csv', index=False)
print("Final mereged dataset created.\n")
print(final_df.shape)