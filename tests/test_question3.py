#!/usr/bin/env python3
"""
Test Question 3: Diesel Hatchback Scenario
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from core.vehicle_rag_langchain_auto import FastVehicleRAG
import time

def test_question3():
    """Test Question 3: Diesel Hatchback Scenario."""
    print("🚗 Testing Question 3: Diesel Hatchback Scenario")
    print("=" * 60)
    
    # Initialize the system
    rag = FastVehicleRAG(csv_path="src/data/stocked_cars.csv")
    
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
    
    # Test Scenario: Diesel hatchback with memory retention
    test_queries = [
        "i need a new diesel hatchback less than 17000",
        "3 to 5 passengers"
    ]
    
    print("📋 Test Scenario:")
    print("1. i need a new diesel hatchback less than 17000")
    print("2. 3 to 5 passengers")
    print()
    
    for i, query in enumerate(test_queries, 1):
        print(f"🔍 Query {i}: {query}")
        print("-" * 40)
        
        start_time = time.time()
        result = rag.process_with_langchain(query)
        response_time = time.time() - start_time
        
        print(f"⏱️ Response time: {response_time:.3f}s")
        print(f"💬 Summary: {result['conversation_summary']}")
        
        # Show current conversation state
        print("📊 Current State:")
        for key, value in rag.conversation_state.items():
            if key != 'conversation_history':
                print(f"  - {key}: {value}")
        
        # Show recommendations
        print("\n🚗 Recommendations:")
        print(result['recommendations'])
        
        # Check if recommendations contain diesel vehicles
        if any(word in result['recommendations'].lower() for word in ['diesel']):
            print("✅ Diesel vehicles found in recommendations")
        else:
            print("❌ No diesel vehicles in recommendations")
        
        # Check if price filter is working
        if rag.conversation_state.get("price_range"):
            print(f"✅ Price range maintained: {rag.conversation_state.get('price_range')}")
        else:
            print("⚠️ Price range not set")
        
        # Check if passenger count is maintained
        if rag.conversation_state.get("passenger_count") == "3-5":
            print("✅ 3-5 passenger count maintained")
        else:
            print(f"⚠️ Passenger count: {rag.conversation_state.get('passenger_count')}")
        
        # Check if body type is maintained
        if rag.conversation_state.get("body_type") == "Hatchback":
            print("✅ Hatchback body type maintained")
        else:
            print(f"⚠️ Body type: {rag.conversation_state.get('body_type')}")
        
        print("\n" + "="*60 + "\n")
    
    # Final verification
    print("🎯 Final Verification:")
    final_state = rag.conversation_state
    print(f"Fuel preference: {final_state.get('fuel_pref')}")
    print(f"Body type: {final_state.get('body_type')}")
    print(f"Price range: {final_state.get('price_range')}")
    print(f"Passenger count: {final_state.get('passenger_count')}")
    
    # Test if all criteria are met
    success = (
        final_state.get('fuel_pref') == 'diesel' and
        final_state.get('body_type') == 'Hatchback' and
        final_state.get('price_range') == '£0 - £17,000' and
        final_state.get('passenger_count') == '3-5'
    )
    
    if success:
        print("🎉 SUCCESS: All criteria maintained across conversation!")
    else:
        print("❌ FAILURE: Some criteria not maintained")
    
    return success

if __name__ == "__main__":
    test_question3() 