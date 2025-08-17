import pandas as pd

from src.compile_file_path import get_file_path


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


full_columns = []
with open(get_file_path('misc/columns.txt'), 'r') as file:
    output = file.read()
    full_columns.extend(output.split("."))
    file.close()


def dataframe_reindex(dataframe):
    return dataframe.reindex(columns=full_columns, fill_value="0")
