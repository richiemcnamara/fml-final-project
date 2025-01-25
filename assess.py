from get_data import get_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as spo

def get_cumulative_return(df, name="Portfolio"):
    if name:
        return df[name] / df[name].iloc[0] - 1
    return df / df.iloc[0] - 1

def get_average_daily_return(df, name="Portfolio"):
    return ((df[name] / df[name].shift()) - 1).mean()

def get_stdev_daily_return(df, name="Portfolio"):
    return ((df[name] / df[name].shift()) - 1).std()

def get_sharpe_ratio(sample_freq, average_daily_return, risk_free_rate, stdev_daily_return):
    return np.sqrt(sample_freq) * (average_daily_return - risk_free_rate) / stdev_daily_return

def get_end_value(df, name="Portfolio"):
    return df[name][-1]

def sum_portfolio(df, starting_value):
    df = (df / df.iloc[0])
    df["Portfolio"] = df.sum(axis=1)
    df["Portfolio"] *= starting_value
    return df

def plot_cumulative_return(cumulative_return, spy_data, year, E):
    # spy_data = (spy_data / spy_data.iloc[0]) - 1

    cumulative_return_df = pd.DataFrame(cumulative_return)
    plot_data = cumulative_return_df.join(spy_data)
    plot_data.plot(figsize=(10,6))
    plt.title(f"{year} Yankee-Trader vs. SPY")
    plt.xlabel('Date')
    plt.ylabel("Cumulative Returns (0-based)")
    plt.savefig(f'./images/{E}_{year}', dpi=1200)
    plt.show()

def plot_daily_prices(df, spy_data):
    spy_data = (spy_data / spy_data.iloc[0])
    pd_df = pd.DataFrame(df)
    plot_data = pd_df.join(spy_data)
    plot_data.plot(y=["Portfolio", "SPY"])
    plt.show()

# def assess_portfolio (start_date, end_date, symbols, allocations,
#                       starting_value=1000000, risk_free_rate=0.0,
#                       sample_freq=252, plot_returns=True):
def assess_portfolio (portfolio, bench_cr, E, starting_value=200000, risk_free_rate=0.0, sample_freq=252, plot_returns=True):

    # df = get_data(start_date, end_date, symbols, include_spy=False)
    portfolio = sum_portfolio(portfolio, starting_value)

    cumulative_return = get_cumulative_return(portfolio) # (df["Portfolio"] / df["Portfolio"].iloc[0] - 1)
    average_daily_return = get_average_daily_return(portfolio) # ((df["Portfolio"] / df["Portfolio"].shift()) - 1).mean()
    stdev_daily_return = get_stdev_daily_return(portfolio)
    sharpe_ratio = get_sharpe_ratio(sample_freq, average_daily_return, risk_free_rate, stdev_daily_return) # np.sqrt(sample_freq) * (average_daily_return - risk_free_rate) / stdev_daily_return
    end_value = get_end_value(portfolio) # df["Portfolio"][-1]

    if plot_returns:
        # spy_data = get_data(start_date, end_date, [])
        year = portfolio.index[0].year
        plot_cumulative_return(cumulative_return, bench_cr, year, E)

    return cumulative_return[-1], average_daily_return, stdev_daily_return, sharpe_ratio, end_value


def optimize_portfolio (start_date, end_date, symbols, plot_returns=True):
    your_bounds = np.array([(0.0, 1.0) for i in range(len(symbols))])
    your_constraints = [{ 'type': 'eq', 'fun': lambda allocations: 1.0 - np.sum(allocations) }]
    your_initial_allocations = np.array([1/len(symbols) for i in range(len(symbols))])

    def f(allocations):
        df = get_data(start_date, end_date, symbols, include_spy=False)
        df = (df / df.iloc[0])
        df *= allocations
        df["Portfolio"] = df.sum(axis=1)
        return (-1) * get_sharpe_ratio(252, get_average_daily_return(df), 0.0, get_stdev_daily_return(df))

    result = spo.minimize(f, your_initial_allocations, method='SLSQP', bounds=your_bounds, constraints=your_constraints) # args=(252, get_average_daily_return(df), 0.0, get_stdev_daily_return(df))

    allocations = result.x

    df = get_data(start_date, end_date, symbols, include_spy=False)
    df = (df / df.iloc[0])
    df *= allocations
    df["Portfolio"] = df.sum(axis=1)

    cumulative_return = get_cumulative_return(df)
    average_daily_return = get_average_daily_return(df)
    stdev_daily_return = get_stdev_daily_return(df)

    sharpe_ratio = (-1) * result.fun

    if plot_returns:
        spy_data = get_data(start_date, end_date, [])
        plot_daily_prices(df, spy_data)

    return allocations, cumulative_return[-1], average_daily_return, stdev_daily_return, sharpe_ratio


if __name__ == "__main__":
    assess_portfolio("2010-01-01", "2010-12-31", ["GOOG", "AAPL", "GLD", "XOM"], [0.2, 0.3, 0.4, 0.1], plot_returns=True) # Test Case 1
    assess_portfolio("2015-06-30", "2015-12-31", ["MSFT", "HPQ", "IBM", "AMZN"], [0.1, 0.1, 0.4, 0.4], starting_value=10000, risk_free_rate=0.0022907680801) # Test Case 2
    assess_portfolio("2020-01-01", "2020-06-30", ['NFLX', 'AMZN', 'XOM', 'PTON'], [0.0, 0.35, 0.35, 0.3], starting_value=500000, sample_freq=52) # Test Case 3
    
    optimize_portfolio("2010-01-01 00:00:00", "2010-12-31 00:00:00", ["GOOG", "AAPL", "GLD", "XOM"]) # Test Case 1
    optimize_portfolio("2004-01-01 00:00:00", "2006-01-01 00:00:00", ['AXP', 'HPQ', 'IBM', 'HNZ']) # Test Case 2
    optimize_portfolio("2011-01-01 00:00:00", "2012-01-01 00:00:00", ['AAPL', 'GLD', 'GOOG', 'XOM']) # Test Case 3
