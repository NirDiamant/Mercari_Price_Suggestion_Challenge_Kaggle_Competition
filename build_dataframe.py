import os
import numpy as np
from pandas import read_csv


def create_relevant_df_from_data(folder_name, file_name):
    """ :param folder_name:folder where the csv data exists
        :param: file_name: name of the csv file
        output: a pandas dataframe, containing names of the columns, with cleaned data"""
    train_data = os.path.join(folder_name,file_name)
    dataframe = read_csv(train_data, header=None, delimiter='\t')
    dataset = dataframe.values
    names = dataset[0, :]

    dataframe = read_csv(train_data, header=None, delimiter='\t', names=names)
    dataframe = dataframe[[(len(str(x)) < 8) for x in dataframe['price']]]  # discard all entries where there is no price
    dataframe = dataframe[[type(x) == str for x in dataframe['brand_name']]]  # discard all entries where brand is none
    dataset = dataframe.values

    all_data_filtered = dataset[1:, 1:]  # filter first row - titles, filter first column - query id
    with open('all_data_filtered.npy', 'wb') as f:
        np.save(f, all_data_filtered)
