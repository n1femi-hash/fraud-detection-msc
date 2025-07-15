import pandas as pd

dataframe = pd.read_csv('../assets/fraudTrain.csv')

print(dataframe.columns)
print(dataframe.head())
print(dataframe.shape)
