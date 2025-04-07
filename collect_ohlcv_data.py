from datetime import datetime
import MetaTrader5 as mt5
import pandas as pd

# Define your MT5 account credentials
ACCOUNT = '' # Your MetaTrader 5 account number      
PASSWORD = '' # Your MetaTrader 5 account password
SERVER = '' # Your MetaTrader 5 server (e.g., 'LiteFinance-MT5-Demo') 

#  Initialize MT5 connection
if not mt5.initialize():
    print("MT5 Initialization failed")
    quit()

# Login to the account
if mt5.login(ACCOUNT, password=PASSWORD, server=SERVER):
    print(f"Logged in successfully to {SERVER} with account {ACCOUNT}")
    
    symbol = "EURUSD_cl"  # Ensure this symbol is correct!
    timeframe = mt5.TIMEFRAME_H1  # 1-hour timeframe
    start_date = datetime(2024, 8, 23)  # 23 August 2024
    end_date = datetime(2025, 1, 23)    # 23 January 2025

    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    
    if rates is None or len(rates) == 0:
        print("Error: No data retrieved. Check symbol name or connection.")
        mt5.shutdown()
        quit()
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    print(f"Retrieved {len(df)} bars")
    df.rename(columns=[
        {'time':'DateTime'},
        {'open':'Open'},
        {'high':'High'}, 
        {'low':'Low'}, 
        {'close':'Close'},
        {'tick_volume':'Volume'}], inplace=True)
    df.drop(columns=['spread', 'real_volume'], inplace=True)
    print(df.info())
    print()
    print(df.head())
    print()
    print(df.tail())
    df.to_csv("Cleaned_EURUSD60.csv", index=False)
else:
    print("Login failed, check credentials!")
    mt5.shutdown()
    quit()

# Shutdown MT5 connection
mt5.shutdown()