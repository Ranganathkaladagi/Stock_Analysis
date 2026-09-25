import yfinance as yf
import pandas as pd

def get_data(stock, start, end):

    data = yf.download(
        stock,
        start=start,
        end=end,
        progress=False
    )

    # Fix MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data