"""
Test script to verify SimFin data access.
API key is read from a local .env file (never committed to git).
"""

import os
from dotenv import load_dotenv
import simfin as sf
from simfin.names import *

# Load environment variables from .env file in the project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

api_key = os.getenv('SIMFIN_API_KEY')
if not api_key:
    raise ValueError(
        "SIMFIN_API_KEY not found. "
        "Please create a .env file in the project root with:\n"
        "  SIMFIN_API_KEY=your_key_here"
    )

sf.set_api_key(api_key)
sf.set_data_dir('~/simfin_data/')

print("Loading SimFin annual income data for US market...")
df = sf.load_income(variant='annual', market='us')

print("\n--- MSFT: Revenue & Net Income (Annual) ---")
print(df.loc['MSFT', [REVENUE, NET_INCOME]])
