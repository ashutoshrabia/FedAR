import pandas as pd
import numpy as np

for j in range(1, 6):

    initial_csv = f"house_{j}.csv"
    data = pd.read_csv(initial_csv) 

    data.fillna(0, inplace=True)

    phi1 = 30  
    phi2 = 0.2  

    appliance_footprints = []
    print(f"Processing house_{j}...")

    for appliance in data.columns[1:]: 
        data['delta'] = data[appliance].diff().abs()
        data['delta_r'] = data['delta'] / data[appliance].shift(1)
        data['switch_point'] = (data['delta'] > phi1) & (data['delta_r'] > phi2)

        for idx, row in data[data['switch_point']].iterrows():
            steady_period = []
            for i in range(idx + 1, len(data)):
                if data.loc[i, 'delta_r'] < phi2:
                    steady_period.append(data.loc[i, appliance])
                else:
                    break

            if steady_period:
                if row[appliance] < steady_period[-1]:  
                    footprint = np.diff(steady_period)  
                    appliance_footprints.append({
                        'appliance': appliance,
                        'footprint': footprint
                    })

    footprints_df = pd.DataFrame(appliance_footprints)

    # Save each house's data to a separate CSV file
    csv_filename = f"train_house_{j}.csv"
    footprints_df.to_csv(csv_filename, index=True)
    print(f'Saved {csv_filename}')
    print(f'House {j} completed.\n')
