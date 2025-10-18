from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import itertools
import pandas as pd
import pandas_ta as ta
import os
import glob
import yfinance as yf

from strategies.oversold_strat import EMA_RSI_TrendFollowing # import your strategy from here. DEFINE IN A SEPARATE FILE USING MY FORMAT

basket = [
    "ADI", "DASH", "CB", "WELL", "KKR", "BTI", "IBN", "CMCSA", "COP", "MO", "SO", "MELI", "BUD",
    "PLD", "BBVA", "SE", "VRTX", "SCCO", "ENB", "MMC", "BN", "CVS", "SMFG", "DELL", "NKE", "NEM",
    "DUK", "HCA", "MCK", "CME", "TT", "PH", "RACE", "BAM", "NTES", "RBLX", "SBUX", "BMO", "ICE",
    "IBIT", "GD", "GSK", "NOC", "BMY", "CDNS", "WM", "COIN", "ORLY", "AMT", "BP", "MCO", "AEM",
    "RIO", "MSTR", "RCL", "SHW", "RELX", "SNPS", "SNOW", "MMM", "CI", "CRH", "MDLZ", "EQIX",
    "BNS", "ELV", "MFG", "AJG", "HWM", "AON", "ECL", "ING", "MSI", "ABNB", "WMB",
    "PBR", "NET", "CTAS", "BK", "CM", "PNC", "CVNA", "MRVL", "TDG", "NGG", "USB", "EMR", "APO",
    "MAR", "GLW", "ITW", "NU", "BCS", "UPS", "CP", "JCI", "RSG", "TRI", "VST", "DB", "INFY",
    "EQNR", "AZO", "FI", "CSX", "ITUB", "MNST", "LYG", "VRT", "EPD", "PYPL", "CRWV", "ADSK",
    "NSC", "TEL", "CNQ", "URI", "PWR", "FTNT", "ZTS", "AMX", "CL", "WDAY", "AEP", "HLT"
]

output_dir = f"/Users/zmahatab/Desktop/algo-trading/optimization_correlation/data/"
os.makedirs(output_dir, exist_ok=True)

start_date = "2010-01-01"
end_date = "2025-10-17" # PUT THE DATE OF THE NEXT DAY HERE INSTEAD OF PRESENT

csv_files = glob.glob(os.path.join(output_dir, "*.csv"))

# code for downloading data
# for ticker in basket:
#     try:
#         print(f"Downloading {ticker}...")
#         data = yf.download(ticker, start=start_date, end=end_date, multi_level_index=False)
#         if data.empty:
#             print(f"No data found for {ticker}")
#             continue

#         file_path = os.path.join(output_dir, f"{ticker}.csv")
#         data.to_csv(file_path)
#         print(f"Saved {ticker} data to {file_path}")
#     except Exception as e:
#         print(f"Failed for {ticker}: {e}")

# adds indicator data to price data
def organize_price_data_and_indicators(file, start_date, end_date):
    df = pd.read_csv(file)

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    df = df.sort_index()
    df = df.loc[start_date : end_date]

    df = df[['Open', 'High', 'Low', 'Close']].copy()

    if df.empty:
        raise ValueError(f"No data found in {file} for range {start_date} - {end_date}")
    
    df['EMA20'] = ta.ema(df['Close'], length=20)
    df['EMA50'] = ta.ema(df['Close'], length=50)
    df['RSI14'] = ta.rsi(df['Close'], length=14)
    df['ROC3'] = ta.roc(df['Close'], length=3)

    df.to_csv(file) # overwrite the same file

    return df

# exhaustive search for one stock
def exhaustive_parameter_search(df):

    results = []

    # use a range of values you want for your indicators here
    # MUST USE INDICATORS ACCORDING TO YOUR STRATEGY
    ema_short_values = [3, 8, 12, 20] 
    ema_long_values = [10, 30, 50, 100]

    rsi_values = [10, 14, 20]
    roc_values = [3, 5, 7]

    for ema_s, ema_l, rsi, roc in itertools.product(ema_short_values, ema_long_values, rsi_values, roc_values):
        if ema_s >= ema_l:
            continue

        bt = Backtest(
            df,
            EMA_RSI_TrendFollowing,
            cash=1_000_000,
            commission=0.0002
        )

        stats = bt.run(
            ema_short_len=ema_s,
            ema_long_len=ema_l,
            rsi_len=rsi,
            roc_len=roc
        )

        trades = stats["_trades"]
        total_trades = len(trades)
        win_rate = float(stats["Win Rate [%]"])
        total_return = float(stats["Return [%]"])
        avg_trade_duration = pd.Timedelta(stats["Avg. Trade Duration"]).days
        # avg_profit = stats["Avg. Trade"]


        if (win_rate > 50 and total_trades >= 190 and total_return > 0 and avg_trade_duration < 5):
            results.append({
                "EMA_Short": ema_s,
                "EMA_Long": ema_l,
                "RSI_Len": rsi,
                "ROC_Len": roc,
                "Win Rate": win_rate,
                "Total Trades": total_trades,
                "Total Return": total_return,
                # "Avg Profit": avg_profit,
                "Avg Duration (days)": avg_trade_duration
            })

    results_df = pd.DataFrame(results).sort_values(by="Total Return", ascending=False)   
    return results_df  

# complete exhaustive search
def run_exhaustive_search_on_basket():
    all_results = [] 

    for file in csv_files:
        try:
            df = organize_price_data_and_indicators(file, start_date, end_date)
            results = exhaustive_parameter_search(df)
            ticker = os.path.basename(file).replace(".csv", "")
            if not results.empty:
                best = results.iloc[0].to_dict()
                best["Ticker"] = ticker
                all_results.append(best)
                print(f"{ticker}: Found {len(results)} valid parameter sets, best: {best}")
            else:
                print(f"{ticker}: No valid parameter sets found")
        except Exception as e:
            print(f"Error with {file}: {e}") # this error is triggered if (win_rate > 50 and total_trades >= 190 and total_return > 0 and avg_trade_duration < 5) are not fulfilled for a stock

    final_df = pd.DataFrame(all_results)
    final_df.to_csv("optimized_strategies.csv", index=False)

def main():
    run_exhaustive_search_on_basket()



if __name__ == "__main__":
    main()
