import pandas as pd
import os

csv = '../data'    # modify as you see fit

df = pd.read_csv('./DataSummary.csv')
# paths for train / test csv
df['TrainPath'] = df['Name'].apply(lambda name: os.path.join(csv, f"{name}_TRAIN.csv"))
df['TestPath'] = df['Name'].apply(lambda name: os.path.join(csv, f"{name}_TEST.csv"))

# convert DataFrame to a list of dictionaries for intuitive management
META = df.to_dict('records')
