import os
from dotenv import load_dotenv
import simfin as sf
from simfin.names import *

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
sf.set_api_key(os.getenv('SIMFIN_API_KEY'))
sf.set_data_dir('~/simfin_data/')

# Annual balance sheet — all companies (needed for book equity -> ROE)
df_balance_annual = sf.load_balance(variant='annual', market='us').reset_index()
out_balance_annual = os.path.join(os.path.dirname(__file__), 'balance_annual_all.csv')
df_balance_annual.to_csv(out_balance_annual, index=False)
print(f"Saved {len(df_balance_annual):,} rows → {out_balance_annual}")
print(f"Columns: {list(df_balance_annual.columns)}\n")

# Quarterly balance sheet — all companies
df_balance_quarterly = sf.load_balance(variant='quarterly', market='us').reset_index()
out_balance_quarterly = os.path.join(os.path.dirname(__file__), 'balance_quarterly_all.csv')
df_balance_quarterly.to_csv(out_balance_quarterly, index=False)
print(f"Saved {len(df_balance_quarterly):,} rows → {out_balance_quarterly}")
print(f"Columns: {list(df_balance_quarterly.columns)}\n")

# Company reference data (ticker -> IndustryId), for sector neutralisation
df_companies = sf.load_companies(market='us').reset_index()
out_companies = os.path.join(os.path.dirname(__file__), 'companies_us.csv')
df_companies.to_csv(out_companies, index=False)
print(f"Saved {len(df_companies):,} rows → {out_companies}")
print(f"Columns: {list(df_companies.columns)}\n")

# Industry -> Sector lookup (IndustryId -> Industry, Sector)
df_industries = sf.load_industries().reset_index()
out_industries = os.path.join(os.path.dirname(__file__), 'industries.csv')
df_industries.to_csv(out_industries, index=False)
print(f"Saved {len(df_industries):,} rows → {out_industries}")
print(f"Columns: {list(df_industries.columns)}\n")
