import pandas as pd
import warnings

from src.compile_file_path import get_file_path

# to suppress 'group_by' deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning)

dataframe_augmented = pd.read_csv(get_file_path('assets/cleaned_fraud_train.csv'))
dataframe_test_augmented = pd.read_csv(get_file_path('assets/cleaned_fraud_test.csv'))

# for display purposes
print("Before Downsampling")
print("==================================")
print("Train: ")
print(dataframe_augmented["is_fraud"].value_counts())
print()
print("Test: ")
print(dataframe_test_augmented["is_fraud"].value_counts())
print()
print("Before Encoding")
print("==================================")
print("Train: ")
print(dataframe_augmented.head())
print()
print("Test: ")
print(dataframe_test_augmented.head())
print()


# to make sure the classes are balanced
def data_downsampling(dataframe):
    minimum_count = dataframe['is_fraud'].value_counts().min()
    grouped_dataframe = dataframe.groupby('is_fraud')
    grouped_dataframe = grouped_dataframe.apply(lambda x: x.sample(minimum_count))
    dataframe = grouped_dataframe.reset_index(drop=True)
    return dataframe


# categorises the data into 1's and 0's
def data_vectorisation(dataframe):
    columns = list(dataframe.columns)
    for column in columns:
        if column != "is_fraud":
            # 'prefix' helps to add the name of the column to the beginning to avoid conflict
            # creating dummies spreads the categories into columns with values of ones and zeros to identify
            # the existence of the value corresponding to the column name
            dummies = pd.get_dummies(dataframe[column], prefix=column, prefix_sep="_", dtype=float)
            # the column used to generate the dummies is dropped to avoid redundancy
            dataframe = dataframe.drop(column, axis=1)
            # the dummies are concatenated on the first axis with the existing dataframe to create a new dataframe
            dataframe = pd.concat([dataframe, dummies], axis=1)
    return dataframe


dataframe_augmented = data_downsampling(dataframe_augmented)
dataframe_test_augmented = data_downsampling(dataframe_test_augmented)

dataframe_augmented = data_vectorisation(dataframe_augmented)
dataframe_test_augmented = data_vectorisation(dataframe_test_augmented)

# to make sure that the columns are of equal length
full_train_columns = dataframe_augmented.columns
full_test_columns = dataframe_test_augmented.columns
full_columns = []
full_columns.extend(full_train_columns)
full_columns.extend(full_test_columns)
full_columns = list(set(full_columns))


def dataframe_reindex(dataframe):
    return dataframe.reindex(columns=full_columns, fill_value="0")


dataframe_augmented = dataframe_reindex(dataframe_augmented)
dataframe_test_augmented = dataframe_reindex(dataframe_test_augmented)

# for display purposes
print("After Downsampling")
print("==================================")
print("Train: ")
print(dataframe_augmented["is_fraud"].value_counts())
print()
print("Test: ")
print(dataframe_test_augmented["is_fraud"].value_counts())
print()
print("After Encoding")
print("==================================")
print("Train: ")
print(dataframe_augmented.head())
print()
print("Test: ")
print(dataframe_test_augmented.head())
print()

print()
print()
# dataframe_augmented.to_csv('../assets/preprocessed_fraud_train.csv', index=False)
# dataframe_test_augmented.to_csv('../assets/preprocessed_fraud_test.csv', index=False)
