#!/usr/bin/env python3
"""
Test Flexible Price Range Extraction
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from core.vehicle_rag_langchain_auto import FastVehicleRAG
import time

def test_flexible_price_scenarios():
    """Test various price range formats."""
    print("🚗 Testing Flexible Price Range Extraction")
    print("=" * 60)
    
    # Initialize the system
    rag = FastVehicleRAG(csv_path="src/data/stocked_cars.csv")
    
    # Test scenarios with different price formats
    test_scenarios = [
        {
            "name": "Scenario 1: Between 10k and 30k",
            "queries": [
                "I want a BMW",
                "between 10k and 30k"
            ]
        },
        {
            "name": "Scenario 2: 15k and 25k",
            "queries": [
                "Show me electric vehicles",
                "15k and 25k"
            ]
        },
        {
            "name": "Scenario 3: Between 5 and 20k",
            "queries": [
                "I need a family car",
                "between 5 and 20k"
            ]
        },
        {
            "name": "Scenario 4: 8k and 15k",
            "queries": [
                "SUV for my family",
                "8k and 15k"
            ]
        }
    ]
    
    for scenario_idx, scenario in enumerate(test_scenarios, 1):
        print(f"\n🔍 {scenario['name']}")
        print("-" * 50)
        
        # Reset conversation state for each scenario
        rag.conversation_state = {
            "passenger_count": None,
            "body_type": None,
            "price_range": None,
            "make_pref": None,
            "answered_questions": set(),
            "conversation_history": []
        }
        
        for query_idx, query in enumerate(scenario['queries'], 1):
            print(f"  Query {query_idx}: {query}")
            
            start_time = time.time()
            result = rag.process_with_langchain(query)
            response_time = time.time() - start_time
            
            print(f"    ⏱️ Response time: {response_time:.3f}s")
            print(f"    💬 Summary: {result['conversation_summary']}")
            
            # Show current conversation state
            print("    📊 Current State:")
            for key, value in rag.conversation_state.items():
                if key != 'conversation_history':
                    print(f"      - {key}: {value}")
            
            # Check if price range was extracted correctly
            if rag.conversation_state.get("price_range"):
                print(f"    ✅ Price range extracted: {rag.conversation_state['price_range']}")
            else:
                print("    ❌ Price range not extracted")
            
            print()
        
        print("    " + "="*50)
    
    print("\n🎯 Summary:")
    print("All flexible price range formats should now be supported:")
    print("- 'between 10k and 30k'")
    print("- '15k and 25k'") 
    print("- 'between 5 and 20k'")
    print("- '8k and 15k'")
    print("- 'between 10 and 25k'")
    print("- '10k and 30k'")

if __name__ == "__main__":
    test_flexible_price_scenarios() 