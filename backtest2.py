from get_data import get_data
import numpy as np
import pandas as pd
from datetime import timedelta
import assess

def update_column(purchase_prices, index, portfolio, col_name, fixed_cost=0.0, floating_cost=0.0):
    total_transaction_value = purchase_prices[0]
    prev_day = index - timedelta(days=1)
    first_part = portfolio.loc[:prev_day,col_name]
    second_part = portfolio.loc[index:,col_name] + purchase_prices
    second_part -= (fixed_cost + np.abs(total_transaction_value * floating_cost))
    full_series = pd.concat([first_part, second_part], axis=0)
    return full_series

def calculate_cash_adjustment(purchase_prices):
    return np.repeat(purchase_prices[0], purchase_prices.shape[0])

def calculate_benchmarks(start_date, end_date, symbol, starting_cash):
    benchmark_df = get_data(str(start_date), str(end_date), [], include_spy=True)
    benchmark_starting_cash = starting_cash - benchmark_df[symbol][0] * 1000
    benchmark_df[symbol] = benchmark_df[symbol] * 1000 + benchmark_starting_cash
    adr = assess.get_average_daily_return(benchmark_df, symbol)
    cr = assess.get_cumulative_return(benchmark_df,symbol)
    sd = assess.get_stdev_daily_return(benchmark_df, symbol)
    sharpe_ratio = assess.get_sharpe_ratio(sample_freq=252, average_daily_return=adr, risk_free_rate=0.0, stdev_daily_return=sd)
    final_value = benchmark_df.loc[str(end_date), symbol]
    return sharpe_ratio, adr, cr, sd, final_value

def print_statistics(start_date, end_date, portfolio, symbol, starting_cash):
    adr = assess.get_average_daily_return(portfolio)
    cr = assess.get_cumulative_return(portfolio)[-1]
    sd = assess.get_stdev_daily_return(portfolio)
    final_value = portfolio.loc[str(end_date), "Portfolio"]
    sharpe_ratio = assess.get_sharpe_ratio(sample_freq=252, average_daily_return=adr, risk_free_rate=0.0, stdev_daily_return=sd)
    
    bench_sharpe_ratio, bench_adr, bench_cr, bench_sd = calculate_benchmarks(start_date, end_date, symbol, starting_cash)
    
    print("Start date: ", start_date)
    print("End date: ", end_date)
    print("Portfolio Sharpe Ratio: ", sharpe_ratio)
    print("Benchmark Sharpe Ratio: ", bench_sharpe_ratio)
    print("Portfolio ADR: ", adr)
    print("Benchmark ADR: ", bench_adr)
    print("Portfolio CR: ", cr)
    print("Benchmark CR: ", bench_cr)
    print("Portfolio SD: ", sd)
    print("Benchmark SD: ", bench_sd)
    print(f"Final Portfolio Value: ${final_value:.2f}")


def assess_strategy(df_trades, symbol = 'DIS', starting_value = 200000, fixed_cost = 0.00, floating_cost = 0.00):
    start_date = df_trades.index[0]
    end_date = df_trades.index[-1]

    df = get_data(str(start_date), str(end_date), [symbol], include_spy=False)

    portfolio = df.copy(deep=True)  # deep copy data
    portfolio[symbol] = 0.0  # initialize stock to 0
    portfolio["Cash"] = float(starting_value)  # initalize cash to starting value

    for index, row in df_trades.iterrows():
        amount = abs(int(row["Trades"]))
        if row["Trades"] > 0:
            purchase_prices = df.loc[index:,symbol] * amount
            portfolio[symbol] = update_column(purchase_prices, index, portfolio, symbol)
            cash_adjustment = calculate_cash_adjustment(purchase_prices)
            portfolio["Cash"] = update_column(cash_adjustment*-1, index, portfolio, "Cash", fixed_cost, floating_cost)
        elif row["Trades"] < 0:
            sale_prices = df.loc[index:,symbol] * amount
            portfolio[symbol] = update_column(sale_prices*-1, index, portfolio, symbol)
            cash_adjustment = calculate_cash_adjustment(sale_prices)
            portfolio["Cash"] = update_column(cash_adjustment, index, portfolio, "Cash", fixed_cost, floating_cost)

    portfolio["Portfolio"] = portfolio.sum(axis=1)
    daily_portfolio_values = portfolio["Portfolio"]

    print_statistics(start_date, end_date, portfolio, symbol, starting_value)
    benchmark_df = get_data(str(start_date), str(end_date), [symbol], include_spy=False)
    benchmark_starting_cash = starting_value - benchmark_df[symbol][0] * 1000
    benchmark_df[symbol] = benchmark_df[symbol] * 1000 + benchmark_starting_cash
    daily_benchmark_values = assess.get_cumulative_return(benchmark_df,symbol)
    

    return daily_portfolio_values, daily_benchmark_values


if __name__ == "__main__":
    # run test cases
    pass