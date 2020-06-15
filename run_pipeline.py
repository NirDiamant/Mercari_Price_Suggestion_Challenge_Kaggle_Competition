""" whole model pipeline"""

from build_dataframe import create_relevant_df_from_data
from keras_model_embedded import nn_model
from shorten_sentence import ShortenSentences
from words2indices import Words2Indices
from misc import load_obj
from analyze_results import analyze_results

''' define hyper-parameters  '''
num_rows = 100000
all_data_filtered_name = 'all_data_filtered'
num_best_words = 2
create_short_sentences_table_name = 'short_sentences_table'
dict_name = 'word_dict'
folder_name = "data"
file_name = "train.tsv"
final_matrix_name = 'final_matrix'
train_test_split = 80000
is_dict_existing = True

''' start pipeline'''
create_relevant_df_from_data(folder_name, file_name) # comment this line after creating the file
short_sentences_table_class = ShortenSentences(all_data_filtered_name, num_best_words, num_rows)  # comment after creating the file
short_sentences_table = short_sentences_table_class.create_short_sentences_database() # comment after creating the file
# short_sentences_table = load_obj(create_short_sentences_table_name)  # uncomment to load existing file
words_2_indicies_class = Words2Indices(short_sentences_table, dict_name, final_matrix_name)
final_matrix = words_2_indicies_class.create_indices_matrix(is_dict_existing)
my_nn_model = nn_model(final_matrix_name, dict_name, train_test_split)
my_nn_model.train_model()  # comment this line after training the model
df_test = my_nn_model.predict_model()
analyze_results(df_test)
