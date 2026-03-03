import yfinance as yf
import pandas as pd
spx = yf.Ticker("^SPX")
# See all available expiry dates
print(spx.options)

# Pull the options chain for a specific expiry

all_calls = []
all_puts = []
for exp in spx.options:
    chain = spx.option_chain(exp)
    chain.calls["expiration"] = exp
    chain.puts["expiration"] = exp
    all_calls.append(chain.calls)
    all_puts.append(chain.puts)
pd.concat(all_calls).to_csv("spy_calls.csv", index=False)
pd.concat(all_puts).to_csv("spy_puts.csv", index=False)

print("Saved all expirations!")