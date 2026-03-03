import yfinance as yf
import pandas as pd
spx = "^SPX"
# See all available expiry dates

# Pull the options chain for a specific expiry
def get_clean_iv_surface(ticker_symbol: str) -> pd.DataFrame:
    ticker = yf.Ticker(ticker_symbol)
    records = []
    for exp in ticker.options:
        chain = ticker.option_chain(exp)
        
        for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
            clean = df[["strike", "impliedVolatility", "bid", "ask", "volume", "openInterest"]].copy()
            
            # Filter out illiquid/bad data
            clean = clean[
                (clean["impliedVolatility"] > 0) &      # remove zero IV
                (clean["volume"] > 0) &                  # must have traded
                (clean["bid"] > 0) &                     # must have valid bid
                (clean["ask"] > clean["bid"])            # ask must exceed bid
            ]
            
            # Use mid-price IV (average bid/ask implied vol is approximated via mid price)
            clean["mid"] = (clean["bid"] + clean["ask"]) / 2
            
            clean["expiration"] = pd.to_datetime(exp)
            clean["type"] = opt_type
            records.append(clean)

    if not records:
        raise ValueError(f"No options data found for {ticker_symbol}")

    df_all = pd.concat(records, ignore_index=True)

    # Calculate time to maturity in years (trading days / 252)
    today = pd.Timestamp.today().normalize()
    df_all["days_to_expiry"] = (df_all["expiration"] - today).dt.days
    df_all["maturity_years"] = df_all["days_to_expiry"] / 365.0

    # Filter out expired or near-expired options
    df_all = df_all[df_all["days_to_expiry"] > 1]

    # Final clean surface
    surface = df_all[[
        "type",
        "strike",
        "expiration",
        "days_to_expiry",
        "maturity_years",
        "impliedVolatility",
        "mid",
        "volume",
        "openInterest"
    ]].sort_values(["expiration", "strike"]).reset_index(drop=True)

    return surface

surface = get_clean_iv_surface(spx)
output_file = f"{spx}_iv_surface.csv"
surface.to_csv(output_file, index=False)
print(f"Saved {len(surface)} rows to {output_file}")
print(surface.head(10))
