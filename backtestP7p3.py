import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

## Updated Backtester ##

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


def assess_strategy(start_date, end_date, trade, sym, starting_value = 1000000, fixed_cost = 9.95, floating_cost = 0.005):
    trades = trade.copy()
    trades["Date"] = trades.index
    trades["Symbol"] = sym
    trades["Direction"] = ['BUY' if x > 0 else 'SELL' for x in trades['Trade']]
    trades["Shares"] = abs(trades['Trade'])
    del trades['Trade']
    trades = trades.reset_index(drop=True)

    if len(trades["Date"]) == 0:
        daily_portfolio_values = pd.DataFrame(index=pd.date_range(start_date, end_date))
        daily_portfolio_values["Dollar Value"] = starting_value
        return daily_portfolio_values

    first_day = trades["Date"].iloc[0]
    last_day = trades["Date"].iloc[-1]
    
    stocks_traded = trades["Symbol"].unique()
    stock_df = get_data(first_day, last_day, stocks_traded, include_spy=False)
    stock_df["Date"] = stock_df.index

    dates = pd.DatetimeIndex(stock_df.index).date

    trades["Date"] = pd.to_datetime(trades["Date"]).dt.date

    daily_portfolio_values = pd.DataFrame(index=dates)
    daily_portfolio_values["Dollar Value"] = np.nan
    available_cash = starting_value

    # symbol: volume
    stocks_in_portfolio = {}

    for s in stocks_traded:
       stocks_in_portfolio[s] = 0

    for date in dates:

        # Trade occurs on this date
        if date in trades["Date"].values:

            # Number of trades on this date
            curr_date_trades = trades.loc[trades["Date"] == date]
            n = curr_date_trades.shape[0]
            
            # Go through each trade and update the portfolio 
            for i in range(n):
               symbol = curr_date_trades.iloc[i, 1]
               direction = curr_date_trades.iloc[i, 2]
               num_shares = curr_date_trades.iloc[i, 3]

               stock_amount = stock_df.loc[str(date), symbol] * num_shares

               available_cash -= fixed_cost
               available_cash -= stock_amount * floating_cost

               if direction == "BUY":
                  stocks_in_portfolio[symbol] += num_shares
                  available_cash -= stock_amount
               else:
                  stocks_in_portfolio[symbol] -= num_shares
                  available_cash += stock_amount

        # Calculating portfolio value for the day
        day_portfolio_value = available_cash 

        for symbol, num_shares in stocks_in_portfolio.items():
           day_portfolio_value += stock_df.loc[str(date), symbol] * num_shares

        daily_portfolio_values.loc[date, "Dollar Value"] = day_portfolio_value


    ## Calculate Portfolio for days prior to trading
    df = get_data(start_date, end_date, stocks_traded, include_spy=False)

    start = df.index[0]
    first = daily_portfolio_values.index[0]
    while start.date() < first:
        if start in df.index:
            daily_portfolio_values.loc[start.date()] = starting_value
        start += timedelta(days=1)
        
    daily_portfolio_values = daily_portfolio_values.sort_index()

    ## Calculate Portfoliio for days after trading
    last = daily_portfolio_values.index[-1]
    end = df.index[-1].date()


    while (last <= end):
        if pd.Timestamp(last) in df.index:
            day_portfolio_value = available_cash 
            for symbol, num_shares in stocks_in_portfolio.items():
                day_portfolio_value += df.loc[pd.Timestamp(last)][symbol] * num_shares
            daily_portfolio_values.loc[last] = day_portfolio_value

        last += timedelta(days=1)
        
    daily_portfolio_values = daily_portfolio_values.sort_index()
          
    return daily_portfolio_values
