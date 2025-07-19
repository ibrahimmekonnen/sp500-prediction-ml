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

from sklearn.ensemble import RandomForestClassifier 

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state = 1)

train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])

from sklearn.metrics import precision_score
preds = model.predict(test[predictors])

import pandas as pd
preds = pd.Series(preds, index = test.index)
precision_score(test["Target"], preds)

combined = pd.concat([test["Target"], preds], axis=1)
combined.plot()

import matplotlib.pyplot as plt
plt.show()
