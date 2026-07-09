"""
Extended SimFin data verification:
1. Quarterly income data for MSFT
2. Dataset date range and company count across all firms
"""

import os
from dotenv import load_dotenv
import simfin as sf
from simfin.names import *

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

api_key = os.getenv('SIMFIN_API_KEY')
if not api_key:
    raise ValueError("SIMFIN_API_KEY not found in .env file.")

sf.set_api_key(api_key)
sf.set_data_dir('~/simfin_data/')

# --- Check 1: Quarterly income for MSFT ---
print("=== Check 1: MSFT Quarterly Revenue & Net Income ===")
df_q = sf.load_income(variant='quarterly', market='us')
print(df_q.loc['MSFT', [REVENUE, NET_INCOME]])

# --- Check 2: Dataset date range and company count ---
print("\n=== Check 2: Annual Dataset Coverage ===")
df_all = sf.load_income(variant='annual', market='us')

# Reset index so Report Date and Ticker are accessible as columns
df_reset = df_all.reset_index()

print("Earliest report date:", df_reset['Report Date'].min())
print("Latest report date:  ", df_reset['Report Date'].max())
print("Number of unique companies:", df_reset['Ticker'].nunique())
