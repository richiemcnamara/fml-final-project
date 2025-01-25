import pandas as pd
import numpy as np
from CARTLearner import CARTLearner
from PERTLearner import PERTLearner
from BootstrapLearner import BootstrapLearner
from data_engineering import clean_yankee_data
from data_engineering import get_yankee_and_stock_data
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

def stock_selection(model, stocks, leaf_size=1, threshold=50):

    # create the accuracy dataframe
    accuracy = pd.DataFrame(0, index=[f'Test Accuracy > {threshold}%'], columns=stocks)

    # create a counter for each stock
    counter = {}
    for s in stocks:
        counter[s] = 0

    # create three-dimensional array
    data = np.zeros((5, 4, len(stocks)))

    # create the 3-dimensional dataframe
    results_df = pd.DataFrame(data.reshape(-1, data.shape[-1]))

    # assign the row and column labels
    x_values = [2017, 2018, 2019, 2021, 2022]
    y_values = ['Train loss', 'Train accuracy', 'Test loss', 'Test accuracy']
    results_df.index = pd.MultiIndex.from_product([x_values, y_values], names=['x', 'y'])
    results_df.columns = stocks

    for year in x_values:

        df = get_yankee_and_stock_data(f'{year}-01-01', f'{year}-12-31', stocks)

        # split the data into train and test
        n = df.shape[0]
        s = int(n*0.65)

        train_start = df.index[0]
        train_end = df.index[s]
        train = df.loc[train_start:train_end]

        test_start = df.index[s+1]
        test_end = df.index[n-1]
        test = df.loc[test_start:test_end]

        train_x = train.iloc[:,len(stocks):]
        test_x = test.iloc[:,len(stocks):]

        # loop through each stock
        for i in range(len(stocks)):

            train_y = train.iloc[:,i]
            test_y = test.iloc[:,i]

            # choose the model
            if model == 'CART':
                m = CARTLearner(leaf_size=leaf_size)
            else:
                m = BootstrapLearner(PERTLearner, {'leaf_size': leaf_size}, bags=20)

            # train the model
            m.train(train_x, train_y)

            # test the model
            train_predictions = m.test(train_x)
            test_predictions = m.test(test_x)

            # calculate the loss and accuracy
            results_df.loc[(year, 'Train loss'), stocks[i]] = log_loss(train_y, train_predictions)
            results_df.loc[(year, 'Train accuracy'), stocks[i]] = accuracy_score(train_y, train_predictions)
            results_df.loc[(year, 'Test loss'), stocks[i]] = log_loss(test_y, test_predictions)
            results_df.loc[(year, 'Test accuracy'), stocks[i]] = accuracy_score(test_y, test_predictions)

            # check if the accuracy is greater than the threshold
            if results_df.loc[(year, "Test accuracy"), stocks[i]] > (threshold/100):
                counter[stocks[i]] += 1

    
    print(results_df)

    # print the accuracy stats
    for s in stocks:
        accuracy.loc[f'Test Accuracy > {threshold}%', s] = counter[s]
        accuracy.loc[f'Test Accuracy > {threshold}%', s] = counter[s]

    print('-------- Accuracy Stats --------')
    print(accuracy)
    
    # print the selected stocks
    print('--------- Stock Selection ---------')
    filtered_columns = accuracy.columns[accuracy.loc[f'Test Accuracy > {threshold}%'] >= 3]
    selected = filtered_columns.tolist()
    print(selected)

    return selected



    