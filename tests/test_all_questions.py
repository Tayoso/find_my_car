#!/usr/bin/env python3
"""
Test All Questions from README
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from core.vehicle_rag_langchain_auto import FastVehicleRAG
import time

def test_all_questions():
    """Test all three questions from README."""
    print("🚗 Testing All Questions from README")
    print("=" * 70)
    
    # Initialize the system
    rag = FastVehicleRAG(csv_path="src/data/stocked_cars.csv")
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Question 1: BMW Scenario",
            "queries": [
                "I want a BMW below £20000",
                "6 to 8 passengers",
                "SUV"
            ],
            "expected_final_state": {
                "make_pref": "BMW",
                "price_range": "£0 - £20,000",
                "passenger_count": "6-8",
                "body_type": "Suv"
            }
        },
        {
            "name": "Question 2: Electric Vehicle Scenario",
            "queries": [
                "Show me electric vehicles for my family",
                "3 to 5 passenger",
                "Price range between 10 and 25k",
                "SUV"
            ],
            "expected_final_state": {
                "fuel_pref": "electric",
                "passenger_count": "3-5",
                "price_range": "£10 - £25,000",
                "body_type": "Suv"
            }
        },
        {
            "name": "Question 3: Diesel Hatchback Scenario",
            "queries": [
                "i need a new diesel hatchback less than 17000",
                "3 to 5 passengers"
            ],
            "expected_final_state": {
                "fuel_pref": "diesel",
                "body_type": "Hatchback",
                "price_range": "£0 - £17,000",
                "passenger_count": "3-5"
            }
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
            "fuel_pref": None,
            "answered_questions": set(),
            "conversation_history": []
        }
        
        # Store the final result
        final_result = None
        
        for query_idx, query in enumerate(scenario['queries'], 1):
            print(f"  Query {query_idx}: {query}")
            
            start_time = time.time()
            result = rag.process_with_langchain(query)
            response_time = time.time() - start_time
            
            print(f"    ⏱️ Response time: {response_time:.3f}s")
            print(f"    💬 Summary: {result['conversation_summary']}")
            
            # Store the final result
            final_result = result
            
            # Show current conversation state
            print("    📊 Current State:")
            for key, value in rag.conversation_state.items():
                if key != 'conversation_history':
                    print(f"      - {key}: {value}")
            
            # Show recommendations
            print("\n    🚗 Recommendations:")
            print(result['recommendations'])
            
            # Check specific criteria based on scenario
            if scenario_idx == 1:  # BMW scenario
                if "BMW" in result['recommendations']:
                    print("    ✅ BMW vehicles found")
                else:
                    print("    ❌ No BMW vehicles found")
                    
            elif scenario_idx == 2:  # Electric scenario
                if any(word in result['recommendations'].lower() for word in ['electric', 'ev']):
                    print("    ✅ Electric vehicles found")
                else:
                    print("    ❌ No electric vehicles found")
                    
            elif scenario_idx == 3:  # Diesel scenario
                if any(word in result['recommendations'].lower() for word in ['diesel']):
                    print("    ✅ Diesel vehicles found")
                else:
                    print("    ❌ No diesel vehicles found")
            
            print()
        
        # Final verification
        print("    🎯 Final Verification:")
        final_state = rag.conversation_state
        expected_state = scenario['expected_final_state']
        
        success = True
        for key, expected_value in expected_state.items():
            actual_value = final_state.get(key)
            if actual_value == expected_value:
                print(f"      ✅ {key}: {actual_value}")
            else:
                print(f"      ❌ {key}: {actual_value} (expected: {expected_value})")
                success = False
        
        if success:
            print("    🎉 SCENARIO PASSED!")
        else:
            print("    ❌ SCENARIO FAILED!")
        
        print("    " + "="*50)
    
    # Print final answers in requested format
    print("\n" + "="*70)
    print("📋 FINAL ANSWERS FOR ALL QUESTIONS")
    print("="*70)
    
    # Re-run each scenario to get final answers
    for scenario_idx, scenario in enumerate(test_scenarios, 1):
        print(f"\n🎯 {scenario['name']}")
        print("-" * 40)
        
        # Reset conversation state
        rag.conversation_state = {
            "passenger_count": None,
            "body_type": None,
            "price_range": None,
            "make_pref": None,
            "fuel_pref": None,
            "answered_questions": set(),
            "conversation_history": []
        }
        
        # Run all queries for this scenario
        for query in scenario['queries']:
            result = rag.process_with_langchain(query)
        
        # Print the final answer
        print("Question:")
        for i, query in enumerate(scenario['queries'], 1):
            print(f"  {i}. {query}")
        
        print("\nFinal Answer:")
        print(result['recommendations'])
        
        print(f"\nFinal State: {result['conversation_summary']}")
        print("-" * 40)
    
    print("\n🎯 Summary:")
    print("All three questions should work correctly with memory retention:")
    print("1. BMW + £0-£20k + 6-8 passengers + SUV")
    print("2. Electric + 3-5 passengers + £10-£25k + SUV")
    print("3. Diesel + Hatchback + £0-£17k + 3-5 passengers")

if __name__ == "__main__":
    test_all_questions() 