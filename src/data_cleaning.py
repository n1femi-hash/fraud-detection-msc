import json
import pandas as pd
from haversine import haversine
from datetime import date
from pathlib import Path

dataframe = pd.read_csv('../assets/fraudTrain.csv')
dataframe_test = pd.read_csv('../assets/fraudTest.csv')
columns_to_remove = [dataframe.columns[0], 'cc_num', 'trans_num', 'unix_time', 'dob', 'trans_date_trans_time', 'lat',
                     'long', 'merch_lat', 'merch_long', 'job']

job_sectors = json.load(Path('../misc/job_sector.json').open())


def extract_date(data):
    split_data = data.split(" ")
    trans_date = split_data[0]
    time = split_data[1]
    trans_date = trans_date.split("-")[1]
    time = time.split(":")[0]
    return pd.Series([int(trans_date), int(time)])


def calculate_distance(data):
    user_coordinates = (data["lat"], data["long"])
    merch_coordinates = (data["merch_lat"], data["merch_long"])
    distance = haversine(user_coordinates, merch_coordinates, unit='mi')
    return distance


def extract_age(data):
    dob = [int(d) for d in data.split("-")]
    difference = date.today() - date(dob[0], dob[1], dob[2])
    return difference.days // 365


def set_job_sector(data):
    try:
        selected_sector = [k for k, v in job_sectors.items() if data in v][0]
    except:
        selected_sector = "Other"
    return selected_sector


def clean_dataframe(dataframe_copy):
    dataframe_copy["age"] = dataframe_copy["dob"].apply(extract_age)
    dataframe_copy["job_sector"] = dataframe_copy["job"].apply(set_job_sector)
    dataframe_copy[["trans_month", "trans_hour"]] = dataframe_copy["trans_date_trans_time"].apply(extract_date)
    dataframe_copy["trans_distance"] = dataframe_copy.apply(calculate_distance, axis=1)
    dataframe_copy = dataframe_copy.drop(columns_to_remove, axis=1)
    return dataframe_copy


dataframe = clean_dataframe(dataframe)
dataframe_test = clean_dataframe(dataframe_test)

dataframe.to_csv('../assets/cleaned_fraud_train.csv', index=False)
dataframe_test.to_csv('../assets/cleaned_fraud_test.csv', index=False)
