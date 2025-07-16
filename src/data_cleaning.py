import pandas as pd

dataframe = pd.read_csv('../assets/fraudTrain.csv')
dataframe = dataframe.drop([dataframe.columns[0], 'cc_num'], axis=1)

print(dataframe.columns)
print(dataframe.head())
print(dataframe.duplicated().value_counts())
print(dataframe.isna().value_counts())
print(dataframe["is_fraud"].value_counts())
