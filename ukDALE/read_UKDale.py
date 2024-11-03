import dateutil
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import nilmtk as ntk
import utility as ut

FILENAME = "ukdale.h5"

ds = ntk.DataSet(FILENAME)
# Appliances = ['fridge freezer', 'washer dryer', 'kettle', 'dish washer', 'boiler']
START_TS = '2012-11-09 22:28:15'
END_TS = '2017-04-26 18:35:53'

STR_TITLE_TS = START_TS + " To " + END_TS

ds.set_window(start=START_TS, end=END_TS)

for i in range(1, 6):
    house = ds.buildings[i].elec
    house_mains = house.mains()
    raw_data = next(house_mains.load(sample_period=6))

    view_df_mains = raw_data.copy()
    view_df_mains.columns = ["_".join(pair) for pair in view_df_mains.columns] 
    view_df_mains = view_df_mains.rename(columns={"power_active": "m_active"})
    view_df_mains = view_df_mains.rename(columns={"voltage_": "m_voltage"})
    view_df_mains = view_df_mains.rename(columns={"power_apparent": "m_apparent"})

    top_5_house_data = house.submeters().select_top_k(k=5)
    raw_df_appliances_top5 = top_5_house_data.dataframe_of_meters()

    print("Is there any null value in dataframe = {}.\n".format(raw_df_appliances_top5.isnull().values.any()))
    raw_df_appliances_top5.columns = house.get_labels(raw_df_appliances_top5.columns)
    print(raw_df_appliances_top5.head())

    # Save each house's data to a separate CSV file
    csv_filename = f"house_{i}.csv"
    raw_df_appliances_top5.to_csv(csv_filename, index=True)
    print(f'Saved {csv_filename}')

    print(f'House {i} completed...\n')
