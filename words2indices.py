import numpy as np
from collections import defaultdict
from torchtext.vocab import Vocab
from collections import Counter
from misc import save_obj, load_obj

class Words2Indices:
    """ class that receives a table containing words / sentences, creates a vocabulary of the words in the table
    and creates a matrix with the numbers which represent the words"""
    def __init__(self, short_sentence_table,dict_name, final_matrix_name):
        self.short_sentences_table = short_sentence_table
        self.dict_name = dict_name
        self.final_matrix_name = final_matrix_name
        pass

    def voc2index(self):
        """ creates a vocabulary of the words in the table"""
        total_cells = self.short_sentences_table.shape[0] * self.short_sentences_table.shape[1]
        counter = 0
        word_dict = defaultdict(int)
        for row in range(self.short_sentences_table.shape[0]):
            for col in range(self.short_sentences_table.shape[1]):
                counter += 1
                if counter % 100 == 0 :
                    print(f'done {counter * 100 / total_cells} %')
                if col == 1 or col == 4 or col ==5: # already numerical values
                    continue
                if col == 0: # name has two words
                    sentence = self.short_sentences_table[row,col]
                    split_words = sentence.split()
                    for word in split_words:
                        word_dict[word] += 1
                else: # name has one word
                    sentence = self.short_sentences_table[row, col]
                    word = str(sentence)
                    word_dict[word] += 1

        index_dict_word = Vocab(Counter(word_dict))
        save_obj(index_dict_word.stoi, self.dict_name)
        return index_dict_word.stoi

    def create_indices_matrix(self, is_dict_existing):
        """ :param is_dict_existing: if a dictionary already exists (from previous calls), skip creating one"""
        if is_dict_existing:
            word_dict = load_obj(self.dict_name)
        else: # create a new vocabulary
            word_dict = self.voc2index()

        word_2_num_sentence = lambda t: [word_dict[word] for word in t.split()] # replace every word in the cell with the matching vocab number
        word_2_num_one_word = lambda t: [word_dict[t]] # refer to the cell content as one string and replace this string with the matching vocab number

        ''' for each column of the table, replace (if needed) the words / sentence with the matching index from the vocabulary'''
        names_indices = np.array([word_2_num_sentence(t) for t in self.short_sentences_table[:, 0]])
        item_conditions = np.expand_dims(self.short_sentences_table[:, 1].astype('float'), axis=1)
        category_names_indices = np.array([word_2_num_one_word(t) for t in self.short_sentences_table[:, 2]])
        brand_names_indices = np.array([word_2_num_one_word(t) for t in self.short_sentences_table[:, 3]])
        price = np.expand_dims(self.short_sentences_table[:, 4].astype('float'), axis=1)
        is_shipping = np.expand_dims(self.short_sentences_table[:, 5].astype('float'), axis=1)
        indices_matrix = np.concatenate(
            (names_indices, item_conditions, category_names_indices, brand_names_indices, price, is_shipping), axis=1)
        save_obj(indices_matrix, self.final_matrix_name)
        return indices_matrix