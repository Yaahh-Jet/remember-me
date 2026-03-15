import sys
import os
"""Quick integration test for interaction store and summary generator."""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interaction_store import InteractionStore
from summary_generator import SummaryGenerator
from dotenv import load_dotenv
load_dotenv()

print("=== Testing DynamoDB ===")
try:
    store = InteractionStore('interactions')
    store.add_interaction('yajat', 'test interaction from debug script')
    records = store.get_interactions('yajat')
    print(f"SUCCESS: {len(records)} records found")
    for r in records:
        print(f"  - {r.timestamp[:10]}: {r.description[:60]}")
except Exception as e:
    print(f"FAILED: {e}")

print("\n=== Testing Claude Summary ===")
try:
    gen = SummaryGenerator()
    records = InteractionStore('interactions').get_interactions('yajat')
    summary = gen.generate('yajat', records)
    print(f"SUCCESS: {summary}")
except Exception as e:
    print(f"FAILED: {e}")
