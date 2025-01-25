import pandas as pd

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