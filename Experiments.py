import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_engineering import clean_yankee_data, get_yankee_and_stock_data
from Stock_Selection import stock_selection
from Model_Selection import model_selection
from BootstrapLearner import BootstrapLearner
from PERTLearner import PERTLearner
from YankeeStrategy import YankeeStrategy
from Backtester import assess_strategy
import backtest2
import backtestP7p3
import assess
from TechnicalStrategy import TechnicalStrategy
from stock_env import StockEnvironment
import yfinance as yf

# Data Prep
years = [2017, 2018, 2019, 2021, 2022]
for y in years:
    clean_yankee_data(y)

# Qualitative Stock Selection
stocks = ['BAC', 'DAL', 'F', 'HES', 'NATH', 'PEP', 'MA', 'TMUS', "BUD", "SONY", "T", "VWAGY"]
# Quantitative Stock Selection
selected_stocks = stock_selection("PERT", stocks, leaf_size=1)
year_sym_hypers = model_selection("PERT", selected_stocks, years)

starting_cash = 200000

# ## Experiment 1: Portfolio of Sponsors over different time periods 
# # Select the sponsors to use & train the corresponding models (w/ correct leaf size)
# # 2018-2022: Train model on first 60% of games for each year: Execute strategy on last 40% of games
# # Report vs. Broad Index over the same time strategy time

port_year_stats = pd.DataFrame(index=years, columns=['Cumulative Return', 'Avg. Daily Return', 'Std. Daily Return', 'Sharpe Ratio', 'Final Value'])
bench_year_stats = pd.DataFrame(index=years, columns=['Cumulative Return', 'Avg. Daily Return', 'Std. Daily Return', 'Sharpe Ratio', 'Final Value'])
num_trades_per_year = []
for year in years:
    sym_leaf_dict = year_sym_hypers[year]

    df = get_yankee_and_stock_data(f'{year}-01-01', f'{year}-12-31', selected_stocks)
    n = df.shape[0]
    s = int(n*0.5)

    train_start = df.index[0]
    train_end = df.index[s]
    train = df.loc[train_start:train_end]

    test_start = df.index[s+1]
    test_end = df.index[n-1]
    test = df.loc[test_start:test_end]

    train_x = train.iloc[:,len(selected_stocks):]
    test_x = test.iloc[:,len(selected_stocks):]

    models = []
    

    for i in range(len(selected_stocks)):
        sym = selected_stocks[i]
        leaf_size = sym_leaf_dict[sym][0]
        m = BootstrapLearner(PERTLearner, {'leaf_size': leaf_size}, bags=20)

        train_y = train.iloc[:,i]
        test_y = test.iloc[:,i]
        m.train(train_x, train_y)
        models.append(m)

    y = YankeeStrategy()
    trades = y.test(models, test_start.strftime('%Y-%m-%d'), test_end.strftime('%Y-%m-%d'), selected_stocks)
    num_trades_per_year.append((trades['Trade'] != 0.0).sum())
    portfolio = assess_strategy(trades, test_start.strftime('%Y-%m-%d'), test_end.strftime('%Y-%m-%d'))

    # Benchmarck
    bench_sharpe_ratio, bench_adr, bench_cr, bench_sd, bench_final = backtest2.calculate_benchmarks(test_start.strftime('%Y-%m-%d'), test_end.strftime('%Y-%m-%d'), 'SPY', starting_cash)

    # Stats & Plots
    port_cr, port_adr, port_sd, port_sr, port_final = assess.assess_portfolio(portfolio, bench_cr, 'E1', starting_cash)

    port_year_stats.loc[year, "Cumulative Return"] = port_cr
    port_year_stats.loc[year, "Avg. Daily Return"] = port_adr
    port_year_stats.loc[year, "Std. Daily Return"] = port_sd
    port_year_stats.loc[year, "Sharpe Ratio"] = port_sr
    port_year_stats.loc[year, "Final Value"] = port_final
    
    bench_year_stats.loc[year, "Cumulative Return"] = bench_cr[-1]
    bench_year_stats.loc[year, "Avg. Daily Return"] = bench_adr
    bench_year_stats.loc[year, "Std. Daily Return"] = bench_sd
    bench_year_stats.loc[year, "Sharpe Ratio"] = bench_sharpe_ratio
    bench_year_stats.loc[year, "Final Value"] = bench_final

# Number of Trades per year
plt.figure(figsize=(10,6))
plt.bar([str(num) for num in years], num_trades_per_year)
plt.xlabel("Year")
plt.ylabel("Number of Trades Made")
plt.title("Yankee Trader Analysis")
plt.savefig('./images/E1_trades', dpi=1200)
plt.show()

# Stats
print('----- Experiment: Portfolio of Sponsors Over Different Years -----')
print(port_year_stats)
print(bench_year_stats)
    

## Experiment 2a: Different Times of the Year
# Train on first 25%, test on second 25% (first half of season)

port_year_stats = pd.DataFrame(index=years, columns=['Cumulative Return', 'Avg. Daily Return', 'Std. Daily Return', 'Sharpe Ratio', 'Final Value'])
bench_year_stats = pd.DataFrame(index=years, columns=['Cumulative Return', 'Avg. Daily Return', 'Std. Daily Return', 'Sharpe Ratio', 'Final Value'])
num_trades_per_year = []
for year in years:
    sym_leaf_dict = year_sym_hypers[year]

    df = get_yankee_and_stock_data(f'{year}-01-01', f'{year}-12-31', selected_stocks)
    n = int(df.shape[0] / 2)
    s = int(n*0.5)

    train_start = df.index[0]
    train_end = df.index[s]
    train = df.loc[train_start:train_end]

    test_start = df.index[s+1]
    test_end = df.index[n-1]
    test = df.loc[test_start:test_end]

    train_x = train.iloc[:,len(selected_stocks):]
    test_x = test.iloc[:,len(selected_stocks):]

    models = []
    
    for i in range(len(selected_stocks)):
        sym = selected_stocks[i]
        leaf_size = sym_leaf_dict[sym][0]
        m = BootstrapLearner(PERTLearner, {'leaf_size': leaf_size}, bags=20)

        train_y = train.iloc[:,i]
        test_y = test.iloc[:,i]
        m.train(train_x, train_y)
        models.append(m)

    y = YankeeStrategy()
    trades = y.test(models, test_start.strftime('%Y-%m-%d'), test_end.strftime('%Y-%m-%d'), selected_stocks)
    num_trades_per_year.append((trades['Trade'] != 0.0).sum())
    portfolio = assess_strategy(trades, test_start.strftime('%Y-%m-%d'), test_end.strftime('%Y-%m-%d'))

    # Benchmarck
    bench_sharpe_ratio, bench_adr, bench_cr, bench_sd, bench_final = backtest2.calculate_benchmarks(test_start.strftime('%Y-%m-%d'), test_end.strftime('%Y-%m-%d'), 'SPY', starting_cash)

    # Stats & Plots
    port_cr, port_adr, port_sd, port_sr, port_final = assess.assess_portfolio(portfolio, bench_cr, 'E2', starting_cash)

    port_year_stats.loc[year, "Cumulative Return"] = port_cr
    port_year_stats.loc[year, "Avg. Daily Return"] = port_adr
    port_year_stats.loc[year, "Std. Daily Return"] = port_sd
    port_year_stats.loc[year, "Sharpe Ratio"] = port_sr
    port_year_stats.loc[year, "Final Value"] = port_final
    
    bench_year_stats.loc[year, "Cumulative Return"] = bench_cr[-1]
    bench_year_stats.loc[year, "Avg. Daily Return"] = bench_adr
    bench_year_stats.loc[year, "Std. Daily Return"] = bench_sd
    bench_year_stats.loc[year, "Sharpe Ratio"] = bench_sharpe_ratio
    bench_year_stats.loc[year, "Final Value"] = bench_final

# Number of Trades per year
plt.figure(figsize=(10,6))
plt.bar([str(num) for num in years], num_trades_per_year)
plt.xlabel("Year")
plt.ylabel("Number of Trades Made")
plt.title("Yankee Trader Analysis")
plt.savefig('./images/E2_trades', dpi=1200)
plt.show()

# Stats
print('----- Experiment: Portfolio of Sponsors Over First Half of Seasons -----')
print(port_year_stats)
print(bench_year_stats)

## Experiment 2b: Train on third 25%, test on last 25% (second half of season)

port_year_stats = pd.DataFrame(index=years, columns=['Cumulative Return', 'Avg. Daily Return', 'Std. Daily Return', 'Sharpe Ratio', 'Final Value'])
bench_year_stats = pd.DataFrame(index=years, columns=['Cumulative Return', 'Avg. Daily Return', 'Std. Daily Return', 'Sharpe Ratio', 'Final Value'])
num_trades_per_year = []
for year in years:
    sym_leaf_dict = year_sym_hypers[year]

    df = get_yankee_and_stock_data(f'{year}-01-01', f'{year}-12-31', selected_stocks)
    n = df.shape[0]
    s = int(n*0.75)

    train_start = df.index[n // 2]
    train_end = df.index[s]
    train = df.loc[train_start:train_end]

    test_start = df.index[s+1]
    test_end = df.index[n-1]
    test = df.loc[test_start:test_end]

    train_x = train.iloc[:,len(selected_stocks):]
    test_x = test.iloc[:,len(selected_stocks):]

    models = []
    

    for i in range(len(selected_stocks)):
        sym = selected_stocks[i]
        leaf_size = sym_leaf_dict[sym][0]
        m = BootstrapLearner(PERTLearner, {'leaf_size': leaf_size}, bags=20)

        train_y = train.iloc[:,i]
        test_y = test.iloc[:,i]
        m.train(train_x, train_y)
        models.append(m)

    y = YankeeStrategy()
    trades = y.test(models, test_start.strftime('%Y-%m-%d'), test_end.strftime('%Y-%m-%d'), selected_stocks)
    num_trades_per_year.append((trades['Trade'] != 0.0).sum())
    portfolio = assess_strategy(trades, test_start.strftime('%Y-%m-%d'), test_end.strftime('%Y-%m-%d'))

    # Benchmarck
    bench_sharpe_ratio, bench_adr, bench_cr, bench_sd, bench_final = backtest2.calculate_benchmarks(test_start.strftime('%Y-%m-%d'), test_end.strftime('%Y-%m-%d'), 'SPY', starting_cash)

    # Stats & Plots
    port_cr, port_adr, port_sd, port_sr, port_final = assess.assess_portfolio(portfolio, bench_cr, 'E3', starting_cash)

    port_year_stats.loc[year, "Cumulative Return"] = port_cr
    port_year_stats.loc[year, "Avg. Daily Return"] = port_adr
    port_year_stats.loc[year, "Std. Daily Return"] = port_sd
    port_year_stats.loc[year, "Sharpe Ratio"] = port_sr
    port_year_stats.loc[year, "Final Value"] = port_final
    
    bench_year_stats.loc[year, "Cumulative Return"] = bench_cr[-1]
    bench_year_stats.loc[year, "Avg. Daily Return"] = bench_adr
    bench_year_stats.loc[year, "Std. Daily Return"] = bench_sd
    bench_year_stats.loc[year, "Sharpe Ratio"] = bench_sharpe_ratio
    bench_year_stats.loc[year, "Final Value"] = bench_final

# Number of Trades per year
plt.figure(figsize=(10,6))
plt.bar([str(num) for num in years], num_trades_per_year)
plt.xlabel("Year")
plt.ylabel("Number of Trades Made")
plt.title("Yankee Trader Analysis")
plt.savefig('./images/E3_trades', dpi=1200)
plt.show()

# Stats
print('----- Experiment: Portfolio of Sponsors Over Second Half of Seasons -----')
print(port_year_stats)
print(bench_year_stats)

## Experiment 2c: Train model on all of last year. Test on this year. 
years = [2017, 2018, 2021]
starting_cash = 200000
port_year_stats = pd.DataFrame(index=years, columns=['Cumulative Return', 'Avg. Daily Return', 'Std. Daily Return', 'Sharpe Ratio', 'Final Value'])
bench_year_stats = pd.DataFrame(index=years, columns=['Cumulative Return', 'Avg. Daily Return', 'Std. Daily Return', 'Sharpe Ratio', 'Final Value'])
num_trades_per_year = []

for i in range(len(years)):
    year = years[i]
    sym_leaf_dict = year_sym_hypers[year]

    train = get_yankee_and_stock_data(f'{year}-01-01', f'{year}-12-31', selected_stocks)
    test = get_yankee_and_stock_data(f'{year + 1}-01-01', f'{year + 1}-12-31', selected_stocks)

    test_start = test.index[0]
    test_end = test.index[-1]

    train_x = train.iloc[:,len(selected_stocks):]
    test_x = test.iloc[:,len(selected_stocks):]
    
    models = []

    for i in range(len(selected_stocks)):
        sym = selected_stocks[i]
        leaf_size = sym_leaf_dict[sym][0]
        m = BootstrapLearner(PERTLearner, {'leaf_size': leaf_size}, bags=20)

        train_y = train.iloc[:,i]
        test_y = test.iloc[:,i]
        m.train(train_x, train_y)
        models.append(m)

    y = YankeeStrategy()
    trades = y.test(models, test_start.strftime('%Y-%m-%d'), test_end.strftime('%Y-%m-%d'), selected_stocks)
    num_trades_per_year.append((trades['Trade'] != 0.0).sum())
    portfolio = assess_strategy(trades, test_start.strftime('%Y-%m-%d'), test_end.strftime('%Y-%m-%d'))

    # Benchmarck
    bench_sharpe_ratio, bench_adr, bench_cr, bench_sd, bench_final = backtest2.calculate_benchmarks(test_start.strftime('%Y-%m-%d'), test_end.strftime('%Y-%m-%d'), 'SPY', starting_cash)

    # Stats & Plots
    port_cr, port_adr, port_sd, port_sr, port_final = assess.assess_portfolio(portfolio, bench_cr, 'E4', starting_cash)

    port_year_stats.loc[year, "Cumulative Return"] = port_cr
    port_year_stats.loc[year, "Avg. Daily Return"] = port_adr
    port_year_stats.loc[year, "Std. Daily Return"] = port_sd
    port_year_stats.loc[year, "Sharpe Ratio"] = port_sr
    port_year_stats.loc[year, "Final Value"] = port_final
    
    bench_year_stats.loc[year, "Cumulative Return"] = bench_cr[-1]
    bench_year_stats.loc[year, "Avg. Daily Return"] = bench_adr
    bench_year_stats.loc[year, "Std. Daily Return"] = bench_sd
    bench_year_stats.loc[year, "Sharpe Ratio"] = bench_sharpe_ratio
    bench_year_stats.loc[year, "Final Value"] = bench_final

# Number of Trades per year
plt.figure(figsize=(10,6))
plt.bar([str(num) for num in years], num_trades_per_year)
plt.xlabel("Year")
plt.ylabel("Number of Trades Made")
plt.title("Yankee Trader Analysis")
plt.savefig('./images/E4_trades', dpi=1200)
plt.show()

# Stats
print('----- Experiment: Portfolio of Sponsors (Training on Prior Year) -----')
print(port_year_stats)
print(bench_year_stats)


## Experiment 3: Comparing Single Stock Portfolios to Tech-Strategy & Q-Trader Portfolios for 2022
# Same test period

year = 2022

hypers = year_sym_hypers[year]

df = get_yankee_and_stock_data(f'{year}-01-01', f'{year}-12-31', selected_stocks)
n = df.shape[0]
s = int(n*0.6)

train_start = df.index[0]
train_end = df.index[s]
train = df.loc[train_start:train_end]

test_start = df.index[s+1]
test_end = df.index[n-1]
test = df.loc[test_start:test_end]

train_x = train.iloc[:,len(selected_stocks):]
test_x = test.iloc[:,len(selected_stocks):]

for i in range(len(selected_stocks)):
    sym = selected_stocks[i]

    data = yf.download([sym], start=f'{year}-01-01', end=f'{year}-12-31')
    data = data.reset_index()
    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    data.to_csv(f'./data/{sym}.csv', index=False)

    leaf_size = hypers[sym][0]
    m = BootstrapLearner(PERTLearner, {'leaf_size': leaf_size}, bags=20)
    train_y = train.iloc[:,i]
    test_y = test.iloc[:,i]
    m.train(train_x, train_y)

    y = YankeeStrategy()
    trades = y.test([m], test_start.strftime('%Y-%m-%d'), test_end.strftime('%Y-%m-%d'), [sym])
    y_portfolio = assess_strategy(trades, test_start.strftime('%Y-%m-%d'), test_end.strftime('%Y-%m-%d'))
    y_cr = (y_portfolio / y_portfolio.iloc[0]) - 1

    t = TechnicalStrategy()
    oos_trades = t.test(test_start.strftime('%Y-%m-%d'), test_end.strftime('%Y-%m-%d'), sym, starting_cash)
    t_trades = oos_trades.copy()
    t_portfolio = backtestP7p3.assess_strategy(test_start.strftime('%Y-%m-%d'), test_end.strftime('%Y-%m-%d'), t_trades, sym, starting_cash, fixed_cost=0, floating_cost=0)
    t_cr = (t_portfolio / t_portfolio.iloc[0]) - 1

    env = StockEnvironment(fixed = 0, floating = 0, starting_cash = starting_cash, share_limit = 1000)
    env.prepare_world(train_start, train_end, sym)
    env.train_learner(start =train_start, end =train_end, symbol = sym, trips = 500, dyna = 0, eps = 0.99, eps_decay = 0.99995 )
    q_trades, q_portfolio = env.test_learner(start = test_start, end = test_end, symbol = sym)
    q_cr = (q_portfolio / q_portfolio.iloc[0]) - 1

    plt.figure(figsize=(10,6))
    plt.plot(y_cr, label='Yankee Trader')
    plt.plot(t_cr, label='Technical Strategy')
    plt.plot(q_cr, label='Q Trader')
    plt.title(f"Strategy Comparison for {sym} in {year}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns (0-based)")
    plt.legend()
    plt.savefig(f'./images/E5_{sym}', dpi=1200)
    plt.show()

