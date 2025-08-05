#!/usr/bin/env python3
"""
Test Electric Vehicle scenario with memory retention
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from core.vehicle_rag_langchain_auto import FastVehicleRAG
import time

def test_electric_scenario():
    """Test the electric vehicle scenario with memory retention."""
    print("🚗 Testing Electric Vehicle Scenario with Memory Retention")
    print("=" * 70)
    
    # Initialize the system
    rag = FastVehicleRAG(csv_path="src/data/stocked_cars.csv")
    
    # Reset conversation state
    rag.conversation_state = {
        "passenger_count": None,
        "body_type": None,
        "price_range": None,
        "make_pref": None,
        "answered_questions": set(),
        "conversation_history": []
    }
    
    # Test Scenario: Electric vehicles with memory retention
    test_queries = [
        "Show me electric vehicles for my family",
        "3 to 5 passenger",
        "Price range between 10 and 25k",
        "SUV"
    ]
    
    print("📋 Test Scenario:")
    print("1. Show me electric vehicles for my family")
    print("2. 3 to 5 passenger")
    print("3. Price range between 10 and 25k")
    print("4. SUV")
    print()
    
    for i, query in enumerate(test_queries, 1):
        print(f"🔍 Query {i}: {query}")
        print("-" * 50)
        
        start_time = time.time()
        result = rag.process_with_langchain(query)
        response_time = time.time() - start_time
        
        print(f"⏱️ Response time: {response_time:.3f}s")
        print(f"💬 Summary: {result['conversation_summary']}")
        print(f"⏭️ Skipped: {result['skipped_questions']}")
        
        # Show current conversation state
        print("📊 Current State:")
        for key, value in rag.conversation_state.items():
            if key != 'conversation_history':
                print(f"  - {key}: {value}")
        
        # Show recommendations
        print("\n🚗 Recommendations:")
        print(result['recommendations'])
        
        # Check if recommendations contain electric vehicles
        if any(word in result['recommendations'].lower() for word in ['electric', 'ev', 'hybrid']):
            print("✅ Electric vehicles found in recommendations")
        else:
            print("❌ No electric vehicles in recommendations")
        
        # Check if passenger count is maintained
        if rag.conversation_state.get("passenger_count") == "3-5":
            print("✅ 3-5 passenger count maintained")
        else:
            print(f"⚠️ Passenger count: {rag.conversation_state.get('passenger_count')}")
        
        # Check if price filter is working
        if rag.conversation_state.get("price_range"):
            print(f"✅ Price range maintained: {rag.conversation_state.get('price_range')}")
        else:
            print("⚠️ Price range not set")
        
        # Check if body type is maintained
        if rag.conversation_state.get("body_type") == "Suv":
            print("✅ SUV body type maintained")
        else:
            print(f"⚠️ Body type: {rag.conversation_state.get('body_type')}")
        
        print("\n" + "="*70 + "\n")
    
    # Final verification
    print("🎯 Final Verification:")
    final_state = rag.conversation_state
    print(f"Passenger count: {final_state.get('passenger_count')}")
    print(f"Price range: {final_state.get('price_range')}")
    print(f"Body type: {final_state.get('body_type')}")
    print(f"Make preference: {final_state.get('make_pref')}")
    
    # Test if all criteria are met
    success = (
        final_state.get('passenger_count') == '3-5' and
        final_state.get('price_range') and
        final_state.get('body_type') == 'Suv'
    )
    
    if success:
        print("🎉 SUCCESS: All criteria maintained across conversation!")
    else:
        print("❌ FAILURE: Some criteria not maintained")
    
    return success

if __name__ == "__main__":
    test_electric_scenario() 