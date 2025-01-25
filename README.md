# fml-final-project
**Final Project for Financial Machine Learning (CSCI 3465)**

This project explores the possibility of using game results from the New York Yankees to predict the stock price movement of their corporate sponsors. The goal is to identify if there are unusual indicators, similar to the "Super Bowl Indicator," that could be leveraged for stock market predictions. The project seeks to answer whether there is a valid relationship between the game performance of the New York Yankees and the stock market performance of their corporate sponsors. The hypothesis is that positive or negative game outcomes may serve as predictors for the stock price movement of these companies.

---

## Data Sources
1. **Yankees Game Data**: Baseball Reference was used for gathering historical game results. 
   - The data includes player statistics and game outcomes for the 2017, 2018, 2019, 2021, and 2022 seasons (excluding the shortened 2020 season).
   - Games from Monday to Thursday were considered, as the effects of games played on weekends were assumed to be irrelevant for the stock market the following Monday.

2. **Stock Data**: Yahoo Finance provided the stock data for corporate sponsors.
   - Adjusted closing prices were used for analysis.
   - A new label was created for whether the stock price increased or decreased the next day.

---

### Methodology

**Data Cleaning & Preprocessing**
- Data from Baseball Reference was cleaned by removing insignificant features and standardizing the format.
- Doubleheader games were handled by omitting the first game of the day.
- Stock data was adjusted to create a binary classification indicating whether the stock price increased or decreased the next day.

**Machine Learning Model**
Machine learning was used to predict whether the stock price of each corporate sponsor would rise or fall the day after a Yankees game:
- Decision Trees: Chosen for their explainability and common usage in finance.
- Ensemble Learning: An ensemble of PERT learners was created for each stock, utilizing the best leaf size for maximum accuracy.

**Trading Strategy**
A stock trading strategy was developed based on the predictions from the machine learning models:
- For each trading time period, we built a portfolio of stocks based on their predicted price movements.
- If a model predicts an increase in price, we would "long" 1000 shares; if it predicts a decrease, we would "short" 1000 shares.
- Backtesting was performed to assess performance

---

## Experiments & Results

1. **Initial Experiment**:
   - The strategy was tested by training models using data from the first 60% of each season and evaluating them on the remaining 40%.
   - Results showed that the SPY outperformed our portfolio, but the volatility was lower than anticipated.

2. **Adjusted Time Frames**:
   - We adjusted the training and testing dates to assess if season timing had an impact on the portfolio's performance.
   - Results indicated that time of season did not significantly affect portfolio returns, and the performance did not carry over between seasons.

3. **2022 Season Focus**:
   - For the 2022 season, the strategy was compared to technical strategies (mean-reversion) and Q-trader.
   - Surprisingly, the Yankees-based strategy outperformed the other strategies in 4 out of 8 trials.

Plots are located in `/images`

---

## Running the Code

1. Run `Experiments.py`


## Conclusions and Future Work

The full research poster is located at `poster.pdf`

- The strategy demonstrated potential but was not able to consistently outperform the SPY.
- Our hypothesis that the SPY would outperform our portfolio was confirmed in most cases, but in 2022, the strategy showed a competitive performance.

- Expansion to include all MLB teams and explore the impact of on-field uniform sponsors, which were introduced in the 2023 season.
- Further research into other sports and leagues, considering the potential for more robust game result data.