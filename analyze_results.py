import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error

def analyze_results(df_test):
 """:param df_test: the function receives a data frame with the values of the real prices and the predicted ones"""

        ''' adding a column of the difference between the values'''
        df_test['diff'] = df_test['pred'] - df_test['price']
        diff_mean, diff_std = df_test['diff'].mean(), df_test['diff'].std()

        print("The mean diff is: ({mean}) with std. {std}.".format(mean=round(diff_mean, 2), std=round(diff_std, 2)))
        min_price = df_test['price'].min()
        max_price = df_test['price'].max()

        min_price_pred = df_test['pred'].min()
        max_price_pred = df_test['pred'].max()
        print(f'min_price:{min_price}, max_price: {max_price}, min_price_pred:{min_price_pred}, max_price_pred:{min_price_pred}')

        ''' plot graphs'''

        num_bins = 500
        plt.figure(1)
        n, bins, patches = plt.hist(df_test['price'], bins=num_bins)
        plt.savefig('test_price')

        plt.figure(2)
        n, bins, patches = plt.hist(df_test['pred'], bins=num_bins)
        plt.savefig('pred_price')

        plt.figure(3)
        plt.plot(df_test['price'])
        plt.plot(df_test['pred'])
        val_mean_squared_log_error = mean_squared_log_error(df_test['price'], df_test['pred'])
        plt.title('msle:' +str(val_mean_squared_log_error))
        plt.savefig('real_and_predict_prices_val')

        '''eliminating noisy samples (0.455 percent) in this case'''
        df_test_lower_than_max_pred = df_test[[x < max_price_pred for x in df_test['price']]]
        print(f'number of elements lower than {max_price_pred}: {df_test_lower_than_max_pred.shape[0]}')
        plt.figure(4)
        plt.plot(df_test_lower_than_max_pred['price'])
        plt.plot(df_test_lower_than_max_pred['pred'])
        df_test_lower_than_max_pred['diff'] = df_test_lower_than_max_pred['pred'] - df_test_lower_than_max_pred['price']
        diff_mean_no_noise, diff_std_no_noise = df_test_lower_than_max_pred['diff'].mean(), df_test_lower_than_max_pred['diff'].std()
        plt.title('mean diff:' +str(diff_mean_no_noise)+', std of diff:' + str(diff_std_no_noise))
        plt.savefig('real_and_predict_prices_val_no_noise')

        '''adding mean difference to the model predictions'''

        plt.figure(5)
        df_test_lower_than_max_pred['pred'] -= diff_mean_no_noise
        plt.plot(df_test_lower_than_max_pred['price'])
        plt.plot(df_test_lower_than_max_pred['pred'])
        df_test_lower_than_max_pred['diff'] = df_test_lower_than_max_pred['pred'] - df_test_lower_than_max_pred['price']
        diff_mean_no_noise_mean_added, diff_std_no_noise_mean_added = df_test_lower_than_max_pred['diff'].mean(), df_test_lower_than_max_pred[
                'diff'].std()
        plt.title('mean diff:' + str(diff_mean_no_noise_mean_added) + ', std of diff:' + str(diff_std_no_noise_mean_added))
        plt.savefig('real_and_predict_prices_val_no_noise_mean_added')
