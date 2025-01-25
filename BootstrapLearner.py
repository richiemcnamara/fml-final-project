import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class BootstrapLearner:

    def __init__(self, constituent, kwargs, bags = 20):
        # set up your object
        self.constituents = []
        for i in range(bags):
            self.constituents.append(constituent(**kwargs))


    def train(self, x, y):
        n = x.shape[0]

        for constituent in self.constituents:
            sampled_rows_indexes = np.random.choice(n, n, replace=True)

            sampled_x = x.iloc[sampled_rows_indexes]
            sampled_y = y[sampled_rows_indexes]
            constituent.train(sampled_x, sampled_y)


    def test(self, x):
        predictions = pd.DataFrame()
        model_num = 1

        for constituent in self.constituents:
            predictions[f"Model: {model_num}"] = constituent.test(x)
            model_num += 1

        return predictions.mode(axis=1).iloc[:,0]
    