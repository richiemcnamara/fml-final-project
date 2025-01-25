import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from get_data import get_data
import tech_ind
from TabularQLearner import TabularQLearner

class StockEnvironment:

  def __init__ (self, fixed = None, floating = None, starting_cash = None, share_limit = None):
    self.shares = share_limit
    self.fixed_cost = fixed
    self.floating_cost = floating
    self.starting_cash = starting_cash

    self.trained_learner = None
    self.df_price = None
    self.q = None
    

  def prepare_world (self, start_date, end_date, symbol):
    """
    Read the relevant price data and calculate some indicators.
    Return a DataFrame containing everything you need.
    """

    self.df_price = get_data(start_date, end_date, [symbol], "Adj Close", False)
    df_close = get_data(start_date, end_date, [symbol], "Close", False)
    df_high = get_data(start_date, end_date, [symbol], "High", False)
    df_low = get_data(start_date, end_date, [symbol], "Low", False)

    df_close["High"] = df_high[f"{symbol}"]
    df_close["Low"] = df_low[f"{symbol}"]

    df_ratio = tech_ind.price_SMA_ratio(self.df_price, 10, symbol)
    df_BBp = tech_ind.boll_bands_percent(self.df_price, 9, symbol)
    df_SO = tech_ind.stochastic_oscillator(df_close, symbol)

    df_inds = pd.DataFrame(index=self.df_price.index, columns=["Price/SMA Ratio", "BB %", "%K"])
    
    self.q = 3
    l = range(self.q)

    df_inds["Price/SMA Ratio"] = pd.qcut(df_ratio["Price to SMA-10 Ratio"], self.q, labels=l)
    df_inds["BB %"] = pd.qcut(df_BBp["BB %"], self.q, labels=l)
    df_inds["%K"] = pd.qcut(df_SO["%K"], self.q, labels=l)

    return df_inds


  def calc_state (self, df, day, holdings):
    """ Quantizes the state to a single number. """

    # Tech Ind + Holdings (internal state)

    # Quantization / Discretization
    # Can do table & then a base to get an int. (like pic from class)

    # Discretize into equal-sized bins
    # Can make indicators into table in prepare world & 
    # then calc state using holdings & index into it in calc_state: would be better: then only have to do this once

    pK = df.loc[day, "%K"]
    ratio = df.loc[day, "Price/SMA Ratio"]
    BBp = df.loc[day, "BB %"]

    if holdings > 0:
      h = 0
    elif holdings < 0:
      h = 1
    else:
      h = 2

    state = (h * self.q**3) + (ratio * self.q**2) + (BBp * self.q) + (pK * 1)

    if pd.isna(state):
      state = 0
    else:
      state += 1

    return state

  
  def train_learner(self, start = None, end = None, symbol = None, trips = 0, dyna = 0,
                     eps = 0.0, eps_decay = 0.0 ):
    """
    Construct a Q-Learning trader and train it through many iterations of a stock
    world.  Store the trained learner in an instance variable for testing.

    Print a summary result of what happened at the end of each trip.
    Feel free to include portfolio stats or other information, but AT LEAST:

    Trip 499 net result: $13600.00
    """

    # 2 Rewards Options: 
    # 1. Daily returns
    # 2. No rewards until you close your position. Get reward of cumul. ret at close (problem of sparse rewards)
    # 3. Some sort of mix

    # Terminal State: Last day

    df = self.prepare_world(start, end, symbol)

    if dyna > 0:
        Q = TabularQLearner(states = 82, actions = 3, epsilon = eps, epsilon_decay = eps_decay, dyna = dyna)
    else:
        Q = TabularQLearner(states = 82, actions = 3, epsilon = eps, epsilon_decay = eps_decay)

    for t in range(trips):

      curr_portfolio = pd.DataFrame(index=df.index, columns=["Dollar Value"])
      trades = pd.DataFrame(columns=["Trade"])

      stocks_in_portfolio = 0

      start_state = self.calc_state(df, df.index[0], 0)
      start_action = Q.test(start_state)

      # Update portfolio for first day
      price_per_share = self.df_price[symbol][df.index[0]]

      available_cash = self.starting_cash

      if start_action == 0:
          # Flat
          stocks_in_portfolio = 0
          
      elif start_action == 1:
          # Short
          trades.loc[df.index[0]] = [-1*self.shares]
          available_cash -= self.fixed_cost
          available_cash -= (price_per_share * self.shares * self.floating_cost)
          stocks_in_portfolio = (-1 * self.shares)
          available_cash += (price_per_share * self.shares)
          
      else:
          # Long
          trades.loc[df.index[0]] = [self.shares]
          available_cash -= self.fixed_cost
          available_cash -= (price_per_share * self.shares * self.floating_cost)
          stocks_in_portfolio = self.shares
          available_cash -= (price_per_share * self.shares)
          
      curr_portfolio["Dollar Value"][df.index[0]] = available_cash + (price_per_share * stocks_in_portfolio)

      # Keep track of when the current position was opened
      # Tuple of the position (0: Flat, 1: Short, 2: Long) and the date
      opened_position = (start_action, df.index[0])
      # Keep track of the date that it was closed
      closed_position = False

      for i in range(1, len(df.index)):

        ind = df.index[i]
        price_per_share = self.df_price[symbol][ind]

        curr_state = self.calc_state(df, ind, stocks_in_portfolio)
  
        # Previous Reward 
        if not closed_position:
          r = 0
        else:
          total_cumul_returns = (curr_portfolio["Dollar Value"][i-1] / curr_portfolio["Dollar Value"][0]) - 1
          total_cumul_returns += (curr_portfolio["Dollar Value"][i-2] / curr_portfolio["Dollar Value"][0]) - 1
          r = np.sign(total_cumul_returns) * np.square(5 * total_cumul_returns)

        # Next Action & Update Portfolio
        next_a = Q.train(curr_state, r)

        if next_a != opened_position[0]:
          # Position Change
          
          # Updating portfolio with new position & trade costs & trade
          if next_a == 0:

            if opened_position[0] == 1:
              # Short to Flat
              trades.loc[ind] = [self.shares]
              available_cash -= self.fixed_cost
              available_cash -= (price_per_share * self.shares * self.floating_cost)
              stocks_in_portfolio = 0
              available_cash -= price_per_share * self.shares
              
            else:
              # Long to Flat
              trades.loc[ind] = [-1 * self.shares]
              available_cash -= self.fixed_cost
              available_cash -= (price_per_share * self.shares * self.floating_cost)
              stocks_in_portfolio = 0
              available_cash += price_per_share * self.shares

          elif next_a == 1:

            if opened_position[0] == 0:
              # Flat to Short
              trades.loc[ind] = [-1 * self.shares]
              available_cash -= self.fixed_cost
              available_cash -= (price_per_share * self.shares * self.floating_cost)
              stocks_in_portfolio = (-1 * self.shares)
              available_cash += price_per_share * self.shares
              
            else:
              # Long to Short
              trades.loc[ind] = [-2 * self.shares]
              available_cash -= self.fixed_cost
              available_cash -= (price_per_share * 2 * self.shares * self.floating_cost)
              stocks_in_portfolio = (-1 * self.shares)
              available_cash += price_per_share * 2 * self.shares

          else:

            if opened_position[0] == 0:
              # Flat to Long
              trades.loc[ind] = [self.shares]
              available_cash -= self.fixed_cost
              available_cash -= (price_per_share * self.shares * self.floating_cost)
              stocks_in_portfolio = self.shares
              available_cash -= price_per_share * self.shares

            else:
              # Short to Long
              trades.loc[ind] = [2 * self.shares]
              available_cash -= self.fixed_cost
              available_cash -= (price_per_share * 2 * self.shares * self.floating_cost)
              stocks_in_portfolio = self.shares
              available_cash -= price_per_share * 2 * self.shares
          
          closed_position = True
          opened_position = (next_a, ind)
        
        else:
          closed_position = False

        curr_portfolio["Dollar Value"][ind] = available_cash + (price_per_share * stocks_in_portfolio)

      portfolio_amount = curr_portfolio["Dollar Value"][-1]
      print(f"Trip {t} net result: ${portfolio_amount}")

    self.trained_learner = Q

    return trades, curr_portfolio

  
  def test_learner(self, start = None, end = None, symbol = None):
    """
    Evaluate a trained Q-Learner on a particular stock trading task.

    Print a summary result of what happened during the test.
    Feel free to include portfolio stats or other information, but AT LEAST:

    Test trip, net result: $31710.00
    Benchmark result: $6690.0000
    """

    df = self.prepare_world(start, end, symbol)
    
    curr_portfolio = pd.DataFrame(index=df.index, columns=["Dollar Value"])
    trades = pd.DataFrame(columns=["Trade"])

    stocks_in_portfolio = 0

    prev_state = self.calc_state(df, df.index[0], 0)
    prev_action = self.trained_learner.test(prev_state)

    # Update portfolio for first day
    price_per_share = self.df_price[symbol][df.index[0]]

    available_cash = self.starting_cash

    if prev_action == 0:
        # Flat
        stocks_in_portfolio = 0

    elif prev_action == 1:
        # Short
        trades.loc[df.index[0]] = [-1*self.shares]
        available_cash -= self.fixed_cost
        available_cash -= (price_per_share * self.shares * self.floating_cost)
        stocks_in_portfolio = (-1 * self.shares)
        available_cash += (price_per_share * self.shares) 

    else:
        # Long
        trades.loc[df.index[0]] = [self.shares]
        available_cash -= self.fixed_cost
        available_cash -= (price_per_share * self.shares * self.floating_cost)
        stocks_in_portfolio = self.shares
        available_cash -= (price_per_share * self.shares) 

    curr_portfolio["Dollar Value"][df.index[0]] = available_cash + (price_per_share * stocks_in_portfolio)

    for i in range(1, len(df.index)):

      ind = df.index[i]
      prev_ind = df.index[i-1]
      price_per_share = self.df_price[symbol][ind]

      curr_state = self.calc_state(df, ind, stocks_in_portfolio)
      curr_action = self.trained_learner.test(curr_state)

      if curr_action != prev_action:
          # Position Change

          # Updating portfolio with new position & trade costs & trade
          if curr_action == 0:

            if prev_action == 1:
              # Short to Flat
              trades.loc[ind] = [self.shares]
              available_cash -= self.fixed_cost
              available_cash -= (price_per_share * self.shares * self.floating_cost)
              stocks_in_portfolio = 0
              available_cash -= price_per_share * self.shares

            else:
              # Long to Flat
              trades.loc[ind] = [-1 * self.shares]
              available_cash -= self.fixed_cost
              available_cash -= (price_per_share * self.shares * self.floating_cost)
              stocks_in_portfolio = 0
              available_cash += price_per_share * self.shares

          elif curr_action == 1:

            if prev_action == 0:
              # Flat to Short
              trades.loc[ind] = [-1 * self.shares]
              available_cash -= self.fixed_cost
              available_cash -= (price_per_share * self.shares * self.floating_cost)
              stocks_in_portfolio = (-1 * self.shares)
              available_cash += price_per_share * self.shares

            else:
              # Long to Short
              trades.loc[ind] = [-2 * self.shares]
              available_cash -= self.fixed_cost
              available_cash -= (price_per_share * 2 * self.shares * self.floating_cost)
              stocks_in_portfolio = (-1 * self.shares)
              available_cash += price_per_share * 2 * self.shares

          else:

            if prev_action == 0:
              # Flat to Long
              trades.loc[ind] = [self.shares]
              available_cash -= self.fixed_cost
              available_cash -= (price_per_share * self.shares * self.floating_cost)
              stocks_in_portfolio = self.shares
              available_cash -= price_per_share * self.shares

            else:
              # Short to Long
              trades.loc[ind] = [2 * self.shares]
              available_cash -= self.fixed_cost
              available_cash -= (price_per_share * 2 * self.shares * self.floating_cost)
              stocks_in_portfolio = self.shares
              available_cash -= price_per_share * 2 * self.shares

      
      curr_portfolio["Dollar Value"][ind] = available_cash + (price_per_share * stocks_in_portfolio)

      prev_state = curr_state
      prev_action = curr_action
    
    benchmark = get_data(start, end, [symbol], include_spy=False)
    benchmark["Stock Value"] = benchmark[symbol] * 1000
    benchmark["Cash"] = self.starting_cash - benchmark.iloc[0,1]
    benchmark["Benchmark"] = benchmark["Stock Value"] + benchmark["Cash"]

    portfolio_amount = curr_portfolio["Dollar Value"][-1]
    benchmark_amount = benchmark["Benchmark"][-1]

    print(f"Test trip, net result: ${portfolio_amount}")
    print(f"Benchmark result: ${benchmark_amount}")

    return trades, curr_portfolio
  

if __name__ == '__main__':
  # Load the requested stock for the requested dates, instantiate a Q-Learning agent,
  # and let it start trading.

  parser = argparse.ArgumentParser(description='Stock environment for Q-Learning.')

  date_args = parser.add_argument_group('date arguments')
  date_args.add_argument('--train_start', default='2018-01-01', metavar='DATE', help='Start of training period.')
  date_args.add_argument('--train_end', default='2019-12-31', metavar='DATE', help='End of training period.')
  date_args.add_argument('--test_start', default='2020-01-01', metavar='DATE', help='Start of testing period.')
  date_args.add_argument('--test_end', default='2021-12-31', metavar='DATE', help='End of testing period.')

  learn_args = parser.add_argument_group('learning arguments')
  learn_args.add_argument('--dyna', default=0, type=int, help='Dyna iterations per experience.')
  learn_args.add_argument('--eps', default=0.99, type=float, metavar='EPSILON', help='Starting epsilon for epsilon-greedy.')
  learn_args.add_argument('--eps_decay', default=0.99995, type=float, metavar='DECAY', help='Decay rate for epsilon-greedy.')

  sim_args = parser.add_argument_group('simulation arguments')
  sim_args.add_argument('--cash', default=200000, type=float, help='Starting cash for the agent.')
  sim_args.add_argument('--fixed', default=0.00, type=float, help='Fixed transaction cost.')
  sim_args.add_argument('--floating', default='0.00', type=float, help='Floating transaction cost.')
  sim_args.add_argument('--shares', default=1000, type=int, help='Number of shares to trade (also position limit).')
  sim_args.add_argument('--symbol', default='DIS', help='Stock symbol to trade.')
  sim_args.add_argument('--trips', default=500, type=int, help='Round trips through training data.')

  args = parser.parse_args()


  # Create an instance of the environment class.
  env = StockEnvironment( fixed = args.fixed, floating = args.floating, starting_cash = args.cash,
                          share_limit = args.shares )
  
  env.prepare_world(args.train_start, args.train_end, args.symbol)

  # Construct, train, and store a Q-learning trader.
  training_trades, training_port = env.train_learner( start = args.train_start, end = args.train_end,
                     symbol = args.symbol, trips = args.trips, dyna = args.dyna,
                     eps = args.eps, eps_decay = args.eps_decay )

  # Test the learned policy and see how it does.
  # In sample.
  print("")
  print("In Sample Testing:")
  insamp_test_trades, insamp_test_port = env.test_learner(start = args.train_start, end = args.train_end, symbol = args.symbol)
  

  # Out of sample.  Only do this once you are fully satisfied with the in sample performance!
  print("")
  print("Out of Sample Testing:")
  outsamp_test_trades, outsamp_test_port = env.test_learner( start = args.test_start, end = args.test_end, symbol = args.symbol )

