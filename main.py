import yfinance as yf

sp500 = yf.Ticker("^GSPC")

sp500 = sp500.history(period="max")

print(sp500)

sp500.index

sp500.plot.line(y="Close", use_index=True)

del sp500["Dividends"]
del sp500["Stock Splits"]

sp500["Tomorrow"] = sp500["Close"].shift(-1)

sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)

sp500 = sp500.loc["1990-01-01":].copy()

sp500

from sklearn.ensemble import RandomForestClassifier 

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state = 1)

train = 