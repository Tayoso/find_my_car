#!/usr/bin/env python3
"""
Test all questions using the exact requirements from README
"""

import os
import sys
import time
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

def test_scenario_1():
    """Test Scenario 1: BMW below £20000 + 6-8 passengers"""
    print("🧪 Test Scenario 1: BMW below £20000 + 6-8 passengers")
    print("=" * 60)
    
    try:
        from src.core.vehicle_rag_enhanced import Llama3SemanticRAG
        
        rag = Llama3SemanticRAG("src/data/stocked_cars.csv")
        
        # Reset conversation state
        rag.conversation_state = {
            "passenger_count": None,
            "body_type": None,
            "price_range": None,
            "make_pref": None,
            "fuel_pref": None,
            "answered_questions": set(),
            "conversation_history": [],
            "reasoning_context": {},
            "semantic_matches": []
        }
        
        print("🔍 Question 1: I want a BMW below £20000")
        result1 = rag.process_with_llama3_semantic("I want a BMW below £20000")
        print(f"✅ Semantic matches: {result1['semantic_matches']}")
        print(f"✅ Summary: {result1['conversation_summary']}")
        print(f"✅ Next question: {result1['next_question']['question'] if result1['next_question'] else 'None'}")
        
        print("\n🔍 Prompt 2: 6 to 8 passenger")
        result2 = rag.process_with_llama3_semantic("6 to 8 passenger")
        print(f"✅ Semantic matches: {result2['semantic_matches']}")
        print(f"✅ Summary: {result2['conversation_summary']}")
        print(f"✅ Recommendations: {result2['recommendations'][:300]}...")
        
        # Verify the results match requirements
        expected_make = "BMW" in result2['conversation_summary']
        expected_passengers = "6-8" in result2['conversation_summary'] or "6 to 8" in result2['conversation_summary']
        expected_price = "£0 - £20000" in result2['conversation_summary'] or "20000" in result2['conversation_summary']
        expected_body = "SUV" in result2['conversation_summary'] or "suv" in result2['conversation_summary'].lower()
        
        print(f"\n✅ Expected BMW: {expected_make}")
        print(f"✅ Expected 6-8 passengers: {expected_passengers}")
        print(f"✅ Expected price under £20000: {expected_price}")
        print(f"✅ Expected SUV body type: {expected_body}")
        
        return expected_make and expected_passengers and expected_price and expected_body
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_scenario_2():
    """Test Scenario 2: Electric vehicles for family with multiple prompts"""
    print("\n🧪 Test Scenario 2: Electric vehicles for family")
    print("=" * 60)
    
    try:
        from src.core.vehicle_rag_enhanced import Llama3SemanticRAG
        
        rag = Llama3SemanticRAG("src/data/stocked_cars.csv")
        
        # Reset conversation state
        rag.conversation_state = {
            "passenger_count": None,
            "body_type": None,
            "price_range": None,
            "make_pref": None,
            "fuel_pref": None,
            "answered_questions": set(),
            "conversation_history": [],
            "reasoning_context": {},
            "semantic_matches": []
        }
        
        print("🔍 Question 1: Show me electric vehicles for my family")
        result1 = rag.process_with_llama3_semantic("Show me electric vehicles for my family")
        print(f"✅ Semantic matches: {result1['semantic_matches']}")
        print(f"✅ Summary: {result1['conversation_summary']}")
        
        print("\n🔍 Prompt 2: 3 to 5 passenger")
        result2 = rag.process_with_llama3_semantic("3 to 5 passenger")
        print(f"✅ Semantic matches: {result2['semantic_matches']}")
        print(f"✅ Summary: {result2['conversation_summary']}")
        print(f"✅ Answer: Should be all electrics with 3-5 passengers and family friendly based on memory")
        
        print("\n🔍 Prompt 3: Price range between 10 and 25k")
        result3 = rag.process_with_llama3_semantic("Price range between 10 and 25k")
        print(f"✅ Semantic matches: {result3['semantic_matches']}")
        print(f"✅ Summary: {result3['conversation_summary']}")
        print(f"✅ Answer: Should be all electrics with 3-5 passengers and family friendly and within 10-25k based on memory")
        
        print("\n🔍 Prompt 4: Body type")
        result4 = rag.process_with_llama3_semantic("SUV")
        print(f"✅ Semantic matches: {result4['semantic_matches']}")
        print(f"✅ Summary: {result4['conversation_summary']}")
        print(f"✅ Answer: Should be all electrics with 3-5 passengers and family friendly and within 10-25k and the body type selected based on memory")
        print(f"✅ Final recommendations: {result4['recommendations'][:300]}...")
        
        # Verify the results
        expected_fuel = "electric" in result4['conversation_summary'].lower()
        expected_passengers = "3-5" in result4['conversation_summary'] or "3 to 5" in result4['conversation_summary']
        expected_price = "£10000 - £25000" in result4['conversation_summary'] or "10" in result4['conversation_summary']
        expected_body = "SUV" in result4['conversation_summary']
        
        print(f"\n✅ Expected electric fuel: {expected_fuel}")
        print(f"✅ Expected 3-5 passengers: {expected_passengers}")
        print(f"✅ Expected price 10-25k: {expected_price}")
        print(f"✅ Expected body type selected: {expected_body}")
        
        return expected_fuel and expected_passengers and expected_price and expected_body
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_scenario_3():
    """Test Scenario 3: Diesel hatchback with passenger count"""
    print("\n🧪 Test Scenario 3: Diesel hatchback")
    print("=" * 60)
    
    try:
        from src.core.vehicle_rag_enhanced import Llama3SemanticRAG
        
        rag = Llama3SemanticRAG("src/data/stocked_cars.csv")
        
        # Reset conversation state
        rag.conversation_state = {
            "passenger_count": None,
            "body_type": None,
            "price_range": None,
            "make_pref": None,
            "fuel_pref": None,
            "answered_questions": set(),
            "conversation_history": [],
            "reasoning_context": {},
            "semantic_matches": []
        }
        
        print("🔍 Question 1: i need a new diesel hatchback less than 17000")
        result1 = rag.process_with_llama3_semantic("i need a new diesel hatchback less than 17000")
        print(f"✅ Semantic matches: {result1['semantic_matches']}")
        print(f"✅ Summary: {result1['conversation_summary']}")
        
        print("\n🔍 Prompt 2: 3 to 5 passengers")
        result2 = rag.process_with_llama3_semantic("3 to 5 passengers")
        print(f"✅ Semantic matches: {result2['semantic_matches']}")
        print(f"✅ Summary: {result2['conversation_summary']}")
        print(f"✅ Answer: Should be all diesel fuel type and hatchbacks below 17k based on memory. This is final answer given body type, fuel type and price has been given.")
        print(f"✅ Final recommendations: {result2['recommendations'][:300]}...")
        
        # Verify the results
        expected_fuel = "diesel" in result2['conversation_summary'].lower()
        expected_body = "hatchback" in result2['conversation_summary'].lower()
        expected_price = "£0 - £17000" in result2['conversation_summary'] or "17000" in result2['conversation_summary']
        expected_passengers = "3-5" in result2['conversation_summary'] or "3 to 5" in result2['conversation_summary']
        
        print(f"\n✅ Expected diesel fuel: {expected_fuel}")
        print(f"✅ Expected hatchback body: {expected_body}")
        print(f"✅ Expected price under £17000: {expected_price}")
        print(f"✅ Expected 3-5 passengers: {expected_passengers}")
        
        return expected_fuel and expected_body and expected_price and expected_passengers
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Run all tests using the exact requirements."""
    print("🚀 Testing All Questions Using README Requirements")
    print("=" * 70)
    
    start_time = time.time()
    
    # Test all scenarios
    scenario1_passed = test_scenario_1()
    scenario2_passed = test_scenario_2()
    scenario3_passed = test_scenario_3()
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS")
    print("=" * 70)
    print(f"✅ Scenario 1 (BMW + 6-8 passengers): {'PASS' if scenario1_passed else 'FAIL'}")
    print(f"✅ Scenario 2 (Electric family + multiple prompts): {'PASS' if scenario2_passed else 'FAIL'}")
    print(f"✅ Scenario 3 (Diesel hatchback + passengers): {'PASS' if scenario3_passed else 'FAIL'}")
    print(f"⏱️ Total time: {total_time:.2f}s")
    
    all_passed = all([scenario1_passed, scenario2_passed, scenario3_passed])
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! System matches README requirements.")
    else:
        print("\n⚠️ Some tests failed. Check the results above.")
    
    return all_passed

if __name__ == "__main__":
    main() 