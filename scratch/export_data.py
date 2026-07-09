import os
from dotenv import load_dotenv
import simfin as sf
from simfin.names import *

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
sf.set_api_key(os.getenv('SIMFIN_API_KEY'))
sf.set_data_dir('~/simfin_data/')

# Annual income — all companies
df_annual = sf.load_income(variant='annual', market='us').reset_index()
out_annual = os.path.join(os.path.dirname(__file__), 'income_annual_all.csv')
df_annual.to_csv(out_annual, index=False)
print(f"Saved {len(df_annual):,} rows → {out_annual}")
print(f"Columns: {list(df_annual.columns)}\n")

# Quarterly income — all companies
df_quarterly = sf.load_income(variant='quarterly', market='us').reset_index()
out_quarterly = os.path.join(os.path.dirname(__file__), 'income_quarterly_all.csv')
df_quarterly.to_csv(out_quarterly, index=False)
print(f"Saved {len(df_quarterly):,} rows → {out_quarterly}")
