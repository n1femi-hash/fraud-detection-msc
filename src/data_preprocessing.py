import pandas as pd

dataframe_augmented = pd.read_csv('../assets/augmented_fraud_train.csv')
dataframe_test_augmented = pd.read_csv('../assets/augmented_fraud_test.csv')


def data_downsampling(dataframe):
    minimum_count = dataframe['is_fraud'].value_counts().min()
    grouped_dataframe = dataframe.groupby('is_fraud')
    grouped_dataframe = grouped_dataframe.apply(lambda x: x.sample(minimum_count))
    dataframe = grouped_dataframe.reset_index(drop=True)
    return dataframe


dataframe_augmented = data_downsampling(dataframe_augmented)
dataframe_test_augmented = data_downsampling(dataframe_test_augmented)
