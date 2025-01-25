import numpy as np
import pandas as pd
import yfinance as yf

def assess_strategy(trades, start, end, starting_value = 200000, fixed_cost = 0, floating_cost = 0):
    trades['Direction'] = trades.apply(lambda row: 'BUY' if row['Trade'] > 0 else 'SELL', axis=1)
    trades["Shares"] = abs(trades['Trade'])
    del trades['Trade']

    # Get the first & last date of the trades
    first_date = trades.index.levels[0][0]
    last_day = trades.index.levels[0][-1]
    
    stocks_traded = list(trades.index.levels[1].unique())

    stock_df = pd.DataFrame()
    # Loop through each ticker symbol
    for symbol in stocks_traded:
        # Get the data for the stock
        tickerData = yf.Ticker(symbol)

        # Get the historical prices for this ticker
        tickerDf = tickerData.history(period='1d', start=start, end=end)

        # Add a column to the dataframe with the ticker symbol
        tickerDf['Symbol'] = symbol

        # Select only the 'Adj Close' column and rename it to the ticker symbol
        closePrices = tickerDf.loc[:, ['Close']].rename(columns={'Close': symbol})

        # Append the data for this ticker to the allData dataframe
        stock_df = pd.concat([stock_df, closePrices], axis=1)

    # Print the data for all tickers
    stock_df = stock_df.tz_localize(None)
    stock_df.index = pd.to_datetime(stock_df.index)

    daily_portfolio_values = pd.DataFrame(index=stock_df.index)
    daily_portfolio_values["Dollar Value"] = np.nan
    available_cash = starting_value

    # symbol: volume
    stocks_in_portfolio = {}

    for s in stocks_traded:
       stocks_in_portfolio[s] = 0

    for date in stock_df.index:

        # Trade occurs on this date
        if date in trades.index.levels[0]:

            for sym in stocks_traded:
                direction = trades.loc[(date, sym), 'Direction']
                num_shares = trades.loc[(date, sym), 'Shares']

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
    return daily_portfolio_values