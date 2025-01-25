import numpy as np
import pandas as pd
from data_engineering import get_yankee_and_stock_data
from CARTLearner import CARTLearner
from Backtester import assess_strategy
from data_engineering import clean_yankee_data

class YankeeStrategy:
    def __init__(self, *params, **kwparams):
        # Defined so you can call it with any parameters and it will just do nothing.
        pass

    def train(self, *params, **kwparams):
        # Defined so you can call it with any parameters and it will just do nothing.
        pass

    def test(self, trained_models, start_date = '2018-01-01', end_date = '2018-12-31', symbols = ['DIS'], starting_cash = 200000):
        # Inputs represent the date range to consider, the single stock to trade, and the starting portfolio value.
        # Needs a trained model for the given date range
        # Return a date-indexed DataFrame with a single column containing the desired trade for that date.
        # Given the position limits, the only possible values are -2000, -1000, 0, 1000, 2000.
        df = get_yankee_and_stock_data(start_date, end_date, symbols)

        # create three-dimensional array
        data = np.zeros((len(df.index), len(symbols), 1)) # (day, stocks, 1)

        # create the 3-dimensional dataframe
        df_trades = pd.DataFrame(data.reshape(-1, data.shape[-1]))

        # assign the row and column labels
        x_values = df.index
        y_values = symbols
        df_trades.index = pd.MultiIndex.from_product([x_values, y_values], names=['Day', 'Symbol'])
        df_trades.columns = ['Trade']

        # create a dictionary to keep track of positions
        positions = {}

        # iterate through the symbols
        for i in range(len(symbols)):
            x = df.iloc[:,i:]

            # Train the model
            predictions = trained_models[i].test(x)

            # Initialize the position
            positions[symbols[i]] = 0

            ## Calculate optimal trade
            for day in df.index:

                # Get the index of the day
                day_idx = df.index.get_loc(day)
            
                # Get the prediction for the day
                increase = predictions[day_idx]

                # Update the position
                if increase:
                    if positions[symbols[i]] == 0:
                        df_trades.loc[(day, symbols[i]), 'Trade'] = 1000
                    elif positions[symbols[i]] > 0:
                        df_trades.loc[(day, symbols[i]), 'Trade'] = 0
                    else:
                        df_trades.loc[(day, symbols[i]), 'Trade'] = 2000
                    positions[symbols[i]] = 1000
                else:
                    if positions[symbols[i]] == 0:
                        df_trades.loc[(day, symbols[i]), 'Trade'] = -1000
                    elif positions[symbols[i]] > 0:
                        df_trades.loc[(day, symbols[i]), 'Trade'] = -2000
                    else:
                        df_trades.loc[(day, symbols[i]), 'Trade'] = 0
                    positions[symbols[i]] = -1000

        return df_trades