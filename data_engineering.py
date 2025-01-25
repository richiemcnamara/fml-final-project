import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


### Data Engineering ###

# Create Excel workbook for the year wanted and correct columns
# When importing to Excel: 
# 1) paste all data into first column 
# 2) Data --> Text to Columns 
# 3) Make Date a date type, W-L text type
# 4) Name the "Home" column


def clean_yankee_data(year, data_folder="./data"):
    """Cleans the data for a certain year and exports it as a CSV named year in the same data_folder folder"""
    
    df = pd.read_excel(f'{data_folder}/{year}.xlsx', index_col=1)

    # Clean Dates
    df.index = df.index.str.replace(r'\s*\(\d+\)$', '', regex=True)
    df.index = pd.to_datetime(df.index + f' {year}', format='%A %b %d %Y')
    df.index = df.index.strftime('%Y-%m-%d')

    df = df[~df.index.duplicated(keep='last')]

    df = df.drop('Tm', axis=1)

    # Clean Opp
    team_abbr_dict = {"ARI": 1, "ATL": 2, "BAL": 3, "BOS": 4, "CHC": 5, "CIN": 6, "CLE": 7, "COL": 8, "CHW": 9, "DET": 10,
        "HOU": 11, "KCR": 12, "LAA": 13, "LAD": 14, "MIA": 15, "MIL": 16, "MIN": 17, "NYM": 18, "NYY": 19, "OAK": 20,
        "PHI": 21, "PIT": 22, "SDP": 23, "SEA": 24, "SFG": 25, "STL": 26, "TBR": 27, "TEX": 28, "TOR": 29, "WSN": 30
    }
    # Map the abbreviations to integers
    df["Opp"] = df["Opp"].map(team_abbr_dict)

    # Clean Home
    df['Home'] = df['Home'].fillna(1).replace('@', 0)

    # Clean W/L
    df['W/L'] = df['W/L'].str[0].replace('W', 1).replace('L', 0)
    df = df.rename({'W/L': 'Win'}, axis=1)

    # Clean Inn
    df['Inn'] = df["Inn"].fillna(9)

    # Clean W-L
    df[['Wins', 'Losses']] = df['W-L'].str.split('-', expand=True)
    df['Wins'] = pd.to_numeric(df['Wins'])
    df['Losses'] = pd.to_numeric(df['Losses'])
    df['W-L'] = df['Wins'] - df['Losses']
    df = df.drop(['Wins', 'Losses'], axis=1)

    # Clean GB
    df['GB'] = df['GB'].replace('Tied', 0)
    def process_gb_value(value):
        if isinstance(value, str):
            if 'up' in value:
                number = value.split('up')[-1].strip()
                if '.' in number:
                    return -1 * float(number)
                else:
                    return -1 * float(number + '.0')
            else:
                return value
        elif isinstance(value, (int, float)):
            return value
        else:
            return value
    df['GB'] = df['GB'].apply(process_gb_value)

    # Clean Time
    def time_to_minutes(time):
        return time.hour * 60 + time.minute

    # Apply the function to the 'Time' column and create a new column 'Minutes'
    df['Time'] = df['Time'].apply(time_to_minutes)

    # Clean D/N
    df['D/N'] = df['D/N'].replace('D', 1).replace('N', 0)
    df = df.rename({'D/N': 'Day Game'}, axis=1)

    # Clean Attendance
    df['Attendance'] = df['Attendance'].fillna(0)

    # Clean Streak
    def process_streak_value(value):
        plus_count = value.count('+')
        minus_count = value.count('-')

        if plus_count > 0:
            return plus_count
        elif minus_count > 0:
            return -1 * minus_count
        else:
            return 0
    df['Streak'] = df['Streak'].apply(process_streak_value)

    df.to_csv(f'{data_folder}/{year}.csv')


def __get_yankee_data__(start_date, end_date, stats='All', data_folder="./data"):
    """Given a start date, end date, and stats, return a dataframe with all the cleaned yankee data
    
    If only certain stats are wanted, pass in a list of the column names"""

    start_year = int(start_date[0:4])
    end_year = int(end_date[0:4])

    dfs = []

    for year in range(start_year, end_year + 1):

        dfs.append(pd.read_csv(f"{data_folder}/{year}.csv", index_col=0))

    yankee_df = pd.concat(dfs)

    if stats == 'All':
        return yankee_df
    else:
        return yankee_df[stats]
        

def get_yankee_and_stock_data(start, end, symbols, stats='All'):
    """Creates a dataframe with the daily yankee stats and if next trading day's stock prices increase 

    Only includes Mon-Thurs where the Yankees Played (trying to predict tomorrow's price, so need to trade today)

    Includes a column which indicates if the next trading day's stock price goes up or down from today's

    1 if it goes up, 0 if it goes down
    
    """
    
    yankee_df = __get_yankee_data__(start, end, stats)
    yankee_df.index = pd.to_datetime(yankee_df.index)

    # Create an empty dataframe to store the data
    stock_df = pd.DataFrame()

    # Loop through each ticker symbol
    for symbol in symbols:
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

    # Compare today's price with tomorrow's price using .shift()
    for sym in symbols:
        stock_df[sym] = (stock_df[sym].shift(-1) > stock_df[sym]).astype(int)

    stock_df = stock_df.iloc[:-1]

    all_days_df = pd.DataFrame(index=pd.date_range(start, end))

    new_stock_df = all_days_df.join(stock_df, how='left').shift(-1).dropna()

    df = new_stock_df.join(yankee_df, how='inner')
    
    for sym in symbols:
        df = df.rename(columns={sym: f"Tomorrow's {sym} Increase"})

    return df[df.index.weekday <= 3]
