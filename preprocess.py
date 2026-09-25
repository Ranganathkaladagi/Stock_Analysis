def preprocess(data):

    # 20-Day Moving Average
    data['MA20'] = data['Close'].rolling(window=20).mean()

    # 50-Day Moving Average
    data['MA50'] = data['Close'].rolling(window=50).mean()

    # Daily Returns
    data['Returns'] = data['Close'].pct_change()

    # Target Variable
    # 1 = Price goes up tomorrow
    # 0 = Price goes down tomorrow
    data['Target'] = (
        data['Close'].shift(-1) > data['Close']
    ).astype(int)

    # Remove missing values
    data.dropna(inplace=True)

    return data