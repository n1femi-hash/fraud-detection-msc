import pandas as pd
import numpy as np

dataframe_augmented = pd.read_csv('../assets/augmented_fraud_train.csv')
dataframe_test_augmented = pd.read_csv('../assets/augmented_fraud_test.csv')


def data_downsampling(dataframe):
    minimum_count = dataframe['is_fraud'].value_counts().min()
    grouped_dataframe = dataframe.groupby('is_fraud')
    grouped_dataframe = grouped_dataframe.apply(lambda x: x.sample(minimum_count))
    dataframe = grouped_dataframe.reset_index(drop=True)
    return dataframe


def object_encoding(data, mappings=None):
    if mappings is None:
        mappings = {}
    try:
        return mappings[data]
    except:
        return mappings['unknown']


def generate_mappings(vocab):
    mappings = {}
    for i in range(len(vocab)):
        mappings[vocab[i]] = i
    mappings['unknown'] = len(mappings.values())
    return mappings


def data_vectorisation(dataframe):
    columns = list(dataframe.columns)
    for column in columns:
        data_type = dataframe[column].dtype
        if data_type == object:
            vocab = np.unique(dataframe[column])
            mappings = generate_mappings(vocab)
            dataframe[column] = dataframe[column].apply(lambda d: object_encoding(d, mappings))
    return dataframe


dataframe_augmented = data_downsampling(dataframe_augmented)
dataframe_test_augmented = data_downsampling(dataframe_test_augmented)

dataframe_augmented = data_vectorisation(dataframe_augmented)
dataframe_test_augmented = data_vectorisation(dataframe_test_augmented)

dataframe_augmented.to_csv('../assets/preprocessed_fraud_train.csv', index=False)
dataframe_test_augmented.to_csv('../assets/preprocessed_fraud_test.csv', index=False)
