from datetime import datetime
import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_FX_METALS_XAU_USD

# Define the start and end dates
# Note: Dukascopy data is in UTC
start_time = datetime(1990, 2, 16)
end_time = datetime(2025, 11, 30)

# Define the instrument (Gold/USD), interval (4 hour), and offer side (BID price)
instrument = INSTRUMENT_FX_METALS_XAU_USD
interval = dukascopy_python.INTERVAL_HOUR_4
offer_side = dukascopy_python.OFFER_SIDE_BID

# Fetch the data
df = dukascopy_python.fetch(
    instrument,
    interval,
    offer_side,
    start_time,
    end_time,
)

# Print the first few rows of the data (a pandas DataFrame)
print(df.head(10))

# You can save the results to a file, for example, a CSV file
df.to_csv("XAUUSD_H4.csv")
