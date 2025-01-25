import numpy as np
import pandas as pd
from CARTLearner import CARTLearner
from PERTLearner import PERTLearner
from BootstrapLearner import BootstrapLearner
from Stock_Selection import stock_selection
from data_engineering import clean_yankee_data
from data_engineering import get_yankee_and_stock_data
from sklearn.metrics import accuracy_score


def model_selection(model, stocks, years, max_leaf_size=20):

    year_stock_leaf_size_tracker = {}
    leaf_sizes = range(1, max_leaf_size)

    for y in years:

        stock_leaf_size_counter = {}
        clean_yankee_data(y)
        df = get_yankee_and_stock_data(f'{y}-01-01', f'{y}-12-31', stocks)

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

        for i in range(len(stocks)):
            s = stocks[i]

            train_y = train.iloc[:,i]
            test_y = test.iloc[:,i]

            accuracies = []

            for l in leaf_sizes:
                
                if model == 'CART':
                    m = CARTLearner(leaf_size=l)
                else:
                    m = BootstrapLearner(PERTLearner, {'leaf_size': l}, bags=20)

                m.train(train_x, train_y)
                test_predictions = m.test(test_x)
                accuracies.append(accuracy_score(test_y, test_predictions))

            max_idx = accuracies.index(max(accuracies))

            stock_leaf_size_counter[s] = (leaf_sizes[max_idx], accuracies[max_idx])

        year_stock_leaf_size_tracker[y] = stock_leaf_size_counter

    print(year_stock_leaf_size_tracker)

    return year_stock_leaf_size_tracker







