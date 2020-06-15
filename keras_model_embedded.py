import pandas as pd
import keras
from keras.layers import Input, Embedding, Dense
from keras.models import Model
from misc import load_obj
from keras.models import model_from_json



class nn_model:
    """ class the receives the final numerical matrix, dictionary name (for embedding purposes)
    and train/test split value.
    the class uses a 3 layer fully connected neural network with embedding layer at the beginning to train a
    regression model for prediction the price of an item"""

    def __init__(self, final_matrix_name, word_dict_name, train_test_split):
        self.final_matrix = load_obj(final_matrix_name).astype('float')
        self.word_dict = load_obj(word_dict_name)
        self.num_words = len(self.word_dict)
        self.train_test_split = train_test_split
        self.X_train = self.final_matrix[:self.train_test_split, :]
        self.X_test = self.final_matrix[self.train_test_split:, :]

    def train_model(self):
        """ creates dataframe from the table, split to numerical, categorical and label
         then trains the nn-model"""

        ''' split data and create pd dafaframe'''
        train = pd.DataFrame(self.X_train,
                             columns=['name1', 'name2', 'condition', 'category', 'brand', 'price', 'shipping'])
        val = pd.DataFrame(self.X_test,
                           columns=['name1', 'name2', 'condition', 'category', 'brand', 'price', 'shipping'])

        continuous_cols = ['condition', 'shipping']
        categorical_cols = ['name1', 'name2', 'category', 'brand']
        y_col = ['price']

        X_train_continuous = train[continuous_cols]
        X_train_categorical = train[categorical_cols]
        y_train = train[y_col]


        X_val_continuous = val[continuous_cols]
        X_val_categorical = val[categorical_cols]
        y_val = val[y_col]

        # normalization #
        train_mean = X_train_continuous.mean(axis=0)
        train_std = X_train_continuous.std(axis=0)

        X_train_continuous = X_train_continuous - train_mean
        X_train_continuous /= train_std

        X_val_continuous = X_val_continuous - train_mean
        X_val_continuous /= train_std

        ''' define categorical inputs'''
        name1_input = Input(shape=(1,), dtype='float')
        name2_input = Input(shape=(1,), dtype='float')
        category_input = Input(shape=(1,), dtype='float')
        brand_input = Input(shape=(1,), dtype='float')

        '''create the embedding vectors for the categorical inputs and contcat them to the continues ones'''
        embeddings_output = 50

        name1_input_embedings = Embedding(output_dim=embeddings_output, input_dim=self.num_words + 1, input_length=1)(
            name1_input)
        name1_input_embedings = keras.layers.Reshape((embeddings_output,))(name1_input_embedings)

        name2_input_embedings = Embedding(output_dim=embeddings_output, input_dim=self.num_words + 1, input_length=1)(
            name2_input)
        name2_input_embedings = keras.layers.Reshape((embeddings_output,))(name2_input_embedings)

        category_input_embedings = Embedding(output_dim=embeddings_output, input_dim=self.num_words + 1,
                                             input_length=1)(category_input)
        category_input_embedings = keras.layers.Reshape((embeddings_output,))(category_input_embedings)

        brand_input_embedings = Embedding(output_dim=embeddings_output, input_dim=self.num_words + 1, input_length=1)(
            brand_input)
        brand_input_embedings = keras.layers.Reshape((embeddings_output,))(brand_input_embedings)

        # Define the continuous variables input
        continuous_input = Input(shape=(X_train_continuous.shape[1],))

        # Concatenate continuous and embeddings inputs
        all_input = keras.layers.concatenate(
            [continuous_input, name1_input_embedings, name2_input_embedings, category_input_embedings,
             brand_input_embedings])

        ''' net architecture'''
        units = 25
        dense1 = Dense(units=units, activation='relu')(all_input)
        dense2 = Dense(units, activation='relu')(dense1)
        dense3 = Dense(units, activation='relu')(dense2)
        predictions = Dense(1)(dense3)
        model = Model(inputs=[continuous_input, name1_input, name2_input, category_input, brand_input],
                      outputs=predictions)

        '''define nn hyper-parameters and train the model'''
        lr = .08
        beta_1 = 0.9
        beta2 = 0.999
        decay = 1e-03
        epochs = 3
        batch_size = 128
        verbose = 1

        model.compile(loss='mean_squared_logarithmic_error',
                      optimizer=keras.optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta2, decay=decay, amsgrad=True),
                      metrics=['mean_squared_error'])

        # Note continuous and categorical columns are inserted in the same order as defined in all_inputs
        history = model.fit([X_train_continuous, X_train_categorical['name1'], X_train_categorical['name2'],
                             X_train_categorical['category'], X_train_categorical['brand']], y_train,
                            epochs=epochs, batch_size=batch_size, verbose=verbose,
                            validation_data=([X_val_continuous, X_val_categorical['name1'], X_val_categorical['name2'],
                                              X_val_categorical['category'], X_val_categorical['brand']], y_val))

        '''save the model'''
        model_json = model.to_json()
        with open("model_all.json", "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights("model_all.h5")

    def predict_model(self):
        """ function predicts a trained model on the validation data """

        ''' split data and create pd dafaframe'''

        train = pd.DataFrame(self.X_train,
                             columns=['name1', 'name2', 'condition', 'category', 'brand', 'price', 'shipping'])
        val = pd.DataFrame(self.X_test,
                           columns=['name1', 'name2', 'condition', 'category', 'brand', 'price', 'shipping'])

        continuous_cols = ['condition', 'shipping']
        categorical_cols = ['name1', 'name2', 'category', 'brand']
        y_col = ['price']

        X_train_continuous = train[continuous_cols]
        X_val_continuous = val[continuous_cols]
        X_val_categorical = val[categorical_cols]
        y_val = val[y_col]

        # normalization #
        train_mean = X_train_continuous.mean(axis=0)
        train_std = X_train_continuous.std(axis=0)

        X_val_continuous = X_val_continuous - train_mean
        X_val_continuous /= train_std

        '''load model'''
        json_file = open('model_all.json', 'r')

        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights("model_all.h5")

        '''create dataframe for prediction results and predict model'''

        df_test = y_val.copy()
        df_test['pred'] = loaded_model.predict(
            [X_val_continuous, X_val_categorical['name1'], X_val_categorical['name2'], X_val_categorical['category'],
             X_val_categorical['brand']])

        return df_test
