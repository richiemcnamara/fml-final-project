import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tech_ind import price_SMA_ratio
from tech_ind import boll_bands
from tech_ind import boll_bands_percent
from tech_ind import stochastic_oscillator

def get_data(start, end, symbols, column_name="Adj Close", include_spy=True, data_folder="./data"):

    # Construct an empty DataFrame with the requested date range.
    dates = pd.date_range(start, end)
    df = pd.DataFrame(index=dates)

    # Read SPY.
    df_spy = pd.read_csv(f'{data_folder}/SPY.csv', index_col=['Date'], parse_dates=True, na_values=['nan'], usecols=['Date',column_name])

    # Use SPY to eliminate non-market days.
    df = df.join(df_spy, how='inner')
    df = df.rename(columns={column_name:'SPY'})

    # Append the data for the remaining symbols, retaining all market-open days.
    for sym in symbols:
      df_sym = pd.read_csv(f'{data_folder}/{sym}.csv', index_col=['Date'], parse_dates=True, na_values=['nan'], usecols=['Date',column_name])
      df = df.join(df_sym, how='left')
      df = df.rename(columns={column_name:sym})

    # Eliminate SPY if requested.
    if not include_spy: del df['SPY']

    df = df.ffill().bfill()

    return df


class TechnicalStrategy:
    def __init__(self, *params, **kwparams):
        # Defined so you can call it with any parameters and it will just do nothing.
        pass

    def train(self, *params, **kwparams):
        # Defined so you can call it with any parameters and it will just do nothing.
        pass

    def test(self, start_date = '2018-01-01', end_date = '2019-12-31', symbol = 'DIS', starting_cash = 200000):
        # Inputs represent the date range to consider, the single stock to trade, and the starting portfolio value.
        #
        # Return a date-indexed DataFrame with a single column containing the desired trade for that date.
        # Given the position limits, the only possible values are -2000, -1000, 0, 1000, 2000.
        
        ## Create Dataframe with Indicators for each day
        df_adjclose = get_data(start_date, end_date, [symbol], "Adj Close", include_spy=False)
        df_close = get_data(start_date, end_date, [symbol], "Close", include_spy=False)
        df_high = get_data(start_date, end_date, [symbol], "High", include_spy=False)
        df_low = get_data(start_date, end_date, [symbol], "Low", include_spy=False)
        df_close["High"] = df_high[symbol]
        df_close["Low"] = df_low[symbol]
        
        SMA_n = 10
        BB_n = 9

        df_ratio_ind = price_SMA_ratio(df_adjclose, SMA_n, symbol)
        df_BBP_ind = boll_bands_percent(boll_bands(df_adjclose, BB_n, symbol), BB_n, symbol)
        df_SO_ind = stochastic_oscillator(df_close, symbol)

        df_indicators = df_ratio_ind.copy()
        df_indicators["BB %"] = df_BBP_ind["BB %"]
        df_indicators["%K"] = df_SO_ind["%K"]
        df_indicators["%D"] = df_SO_ind["%D"]


        df_trades = pd.DataFrame(index=df_indicators.index, columns=["Position", "Trade"])

        curr_pos = 0
        lower = None
        upper = None

        lower_SO = 20
        upper_SO = 80
        lower_BBp = 40
        upper_BBp = 60

        for date in df_trades.index:

            ratio = df_indicators.loc[date][f"Price to SMA-{SMA_n} Ratio"]
            BBp = df_indicators.loc[date]["BB %"]
            pK = df_indicators.loc[date]["%K"]

            if (pK < lower_SO) and (ratio > 0.98) and (not upper):
                df_trades.loc[date]["Position"] = 1000
                curr_pos = 1000
                lower = True
            elif (lower) and (BBp < upper_BBp) and (BBp > lower_BBp):
                df_trades.loc[date]["Position"] = -1000
                curr_pos = -1000
                lower = False
            elif (pK > upper_SO) and (ratio < 1.02) and (not lower):
                df_trades.loc[date]["Position"] = -1000
                curr_pos = -1000
                upper = True
            elif (upper) and (BBp < upper_BBp) and (BBp > lower_BBp):
                df_trades.loc[date]["Position"] = 1000
                curr_pos = 1000
                upper = False
            else:
                df_trades.loc[date]["Position"] = curr_pos

        ## Calculate the desired Trades based on desired Positions
        df_trades["Trade"] = (df_trades["Position"].shift(-1) - df_trades["Position"])* -1

        df_trades = df_trades.drop(index=df_trades.index[-1])

        df_trades = df_trades[df_trades['Trade'] != 0]

        del df_trades["Position"]

        return df_trades