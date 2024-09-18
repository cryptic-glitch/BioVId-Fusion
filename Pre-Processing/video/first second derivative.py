import pandas as pd
import numpy as np

file_path = "/home/prashant/Desktop/all/Datasets/BioVid_pain/PartA/not fianl dataset/RAW video csv/c vxcv/features (1).csv"  # Update this path to the correct location
df = pd.read_csv(file_path)

df = df[(df['Frame'] >= 0) & (df['Frame'] <= 137)]

columns_to_derivative = ['Eye-Brow Distance', 'Eye Closure', 'Mouth Height',
                         'Yaw', 'Pitch', 'Roll', 'Translation_X', 'Translation_Y', 'Translation_Z']

for column in columns_to_derivative:

    df[f'{column} derivative1'] = np.gradient(df[column].values)


    df[f'{column} derivative2'] = np.gradient(df[f'{column} derivative1'].values)

df.to_csv('yooo.csv', index=False)

