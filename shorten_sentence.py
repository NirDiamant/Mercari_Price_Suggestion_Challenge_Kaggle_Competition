from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from misc import save_obj


class ShortenSentences:
    """ class that receives a table of data, number of best words to represent each sentence, and how many rows of
    the data to use"""
    def __init__(self, all_data_filtered_name, num_best_words, num_rows_to_use):
        self.all_data_filtered = np.load(all_data_filtered_name + ".npy", allow_pickle=True)
        self.shorten_table = self.all_data_filtered[0: num_rows_to_use, :]
        self.num_of_best = num_best_words

    def get_list_of_two_most_important_word_in_sentence(self, list_of_sentences):
        """ :param list_of_sentences: the function receives a list of #rows sentences
            output: np array of the top #num best words as a replacement for each of the sentences
        """
        list_of_sentences = list(list_of_sentences.T)
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(list_of_sentences)
        feature_names = vectorizer.get_feature_names()
        dense = vectors.todense()
        denselist = dense.tolist()
        df = pd.DataFrame(denselist, columns=feature_names)

        max_vals = df.idxmax(axis=1)
        if self.num_of_best == 1:
            total = np.array(list(max_vals))
            return total
        else:  # num of best == 2
            second_max_vals = df.T.apply(lambda x: x.nlargest(2).idxmin())
            short_sentence = max_vals + ' ' + second_max_vals
            total = np.array(short_sentence)
            return total

    def create_short_sentences_database(self):
        """ outout: function creates short sentences from the relevant df columns and than concatenates all columns
        back to matrix """
        names = np.expand_dims(self.get_list_of_two_most_important_word_in_sentence(self.shorten_table[:, 0].astype('str'), self.num_best_words), axis=1)
        item_conditions = np.expand_dims(self.shorten_table[:, 1], axis=1)
        category_names = np.expand_dims(self.shorten_table[:, 2].astype('str'), axis=1)
        brand_names = np.expand_dims(self.shorten_table[:, 3].astype('str'), axis=1)
        prices = np.expand_dims(self.shorten_table[:, 4],axis=1)
        is_shipping = np.expand_dims(self.shorten_table[:, 5], axis=1)
        #item_descriptions = np.expand_dims(self.get_list_of_two_most_important_word_in_sentence(sel.foriginal_table[:, 6].astype('str'), self.num_best_words), axis=1) # decided not to use it

        short_sentence_table = np.concatenate((names,item_conditions, category_names,brand_names, prices, is_shipping), axis=1)
        save_obj(short_sentence_table, 'short_sentences_table')
        return short_sentence_table
