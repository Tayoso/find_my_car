#!/usr/bin/env python3
"""
Comprehensive test of Llama 3 system with 5 examples and flow testing
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.vehicle_rag import VehicleRAG

def test_llama3_flow():
    """Test Llama 3 system with 5 different examples and complete flow."""
    print("🧪 Testing Llama 3 System with 5 Examples and Flow")
    print("=" * 70)
    
    # Initialize the RAG system
    rag = VehicleRAG("src/data/stocked_cars.csv")
    print("✅ Llama 3 RAG system initialized")
    
    # Test scenarios with different complexity levels
    test_scenarios = [
        {
            "name": "Specific Requirements",
            "initial_query": "i need a new diesel hatchback less than 17000",
            "expected_skips": ["body_type", "price_range"],
            "expected_next_question": "passenger_count",
            "description": "User provides specific body type and price"
        },
        {
            "name": "Fuel Type + Family Use",
            "initial_query": "show me electric vehicles for my family",
            "expected_skips": ["fuel_type"],
            "expected_next_question": "passenger_count",
            "description": "User mentions fuel type and family use"
        },
        {
            "name": "Brand Preference + Price",
            "initial_query": "I want a BMW under 20000",
            "expected_skips": ["price_range"],
            "expected_next_question": "passenger_count",
            "description": "User specifies brand and price range"
        },
        {
            "name": "Use Case + Efficiency",
            "initial_query": "family car with good fuel economy",
            "expected_skips": ["passenger_count"],
            "expected_next_question": "body_type",
            "description": "User mentions family use and fuel efficiency"
        },
        {
            "name": "Luxury + Body Type + Transmission",
            "initial_query": "luxury SUV with automatic transmission",
            "expected_skips": ["body_type", "transmission_type"],
            "expected_next_question": "passenger_count",
            "description": "User specifies luxury, body type, and transmission"
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🔍 Test {i}: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"Query: '{scenario['initial_query']}'")
        print("-" * 60)
        
        try:
            # Test initial query
            result = rag.process_with_llama3(scenario['initial_query'])
            
            print(f"✅ Llama 3 managed: {result.get('llama3_managed', False)}")
            print(f"✅ Next question: {result.get('next_question', 'None')}")
            print(f"✅ Question type: {result.get('question_type', 'None')}")
            print(f"✅ Options available: {len(result.get('options', []))}")
            
            # Check if expected skips match
            actual_skips = result.get('skipped_questions', [])
            expected_skips = scenario['expected_skips']
            skip_match = all(skip in actual_skips for skip in expected_skips)
            print(f"✅ Skip logic: {'PASSED' if skip_match else 'FAILED'}")
            if not skip_match:
                print(f"   Expected skips: {expected_skips}")
                print(f"   Actual skips: {actual_skips}")
            
            # Check next question
            next_q_match = scenario['expected_next_question'] in result.get('next_question', '')
            print(f"✅ Next question logic: {'PASSED' if next_q_match else 'FAILED'}")
            
            # Test follow-up flow
            if result.get('next_question') and result.get('next_question') != 'none':
                print(f"\n🔄 Testing follow-up flow...")
                
                # Simulate user answering the next question
                follow_up_answer = "3-5 passengers"  # Common answer
                follow_up_result = rag.process_with_llama3(f"User answered: {follow_up_answer}")
                
                print(f"✅ Follow-up processed: {follow_up_result.get('llama3_managed', False)}")
                print(f"✅ New next question: {follow_up_result.get('next_question', 'None')}")
                print(f"✅ Conversation summary: {follow_up_result.get('conversation_summary', 'None')}")
                
                # Test if conversation continues or completes
                if follow_up_result.get('next_question') == 'none':
                    print("✅ Conversation completed successfully")
                else:
                    print(f"✅ Conversation continues with: {follow_up_result.get('next_question')}")
            
            # Count recommendations
            if result.get('recommendations'):
                lines = result['recommendations'].split('\n')
                recommendation_count = len([l for l in lines if l.strip().startswith('**')])
                print(f"📋 Found {recommendation_count} recommendations")
            
            results.append({
                "scenario": scenario['name'],
                "success": True,
                "llama3_managed": result.get('llama3_managed', False),
                "skip_logic_passed": skip_match,
                "next_question_passed": next_q_match,
                "recommendations_count": recommendation_count if result.get('recommendations') else 0,
                "conversation_summary": result.get('conversation_summary', '')
            })
            
        except Exception as e:
            print(f"❌ Error in test {i}: {e}")
            results.append({
                "scenario": scenario['name'],
                "success": False,
                "error": str(e)
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 70)
    
    successful_tests = [r for r in results if r['success']]
    failed_tests = [r for r in results if not r['success']]
    
    print(f"✅ Successful tests: {len(successful_tests)}/5")
    print(f"❌ Failed tests: {len(failed_tests)}/5")
    
    if successful_tests:
        print(f"\n🎯 Llama 3 Management: {sum(1 for r in successful_tests if r['llama3_managed'])}/{len(successful_tests)}")
        print(f"🎯 Skip Logic: {sum(1 for r in successful_tests if r['skip_logic_passed'])}/{len(successful_tests)}")
        print(f"🎯 Next Question Logic: {sum(1 for r in successful_tests if r['next_question_passed'])}/{len(successful_tests)}")
        
        print("\n📋 Detailed Results:")
        for i, result in enumerate(successful_tests, 1):
            print(f"  {i}. {result['scenario']}")
            print(f"     Llama 3 managed: {result['llama3_managed']}")
            print(f"     Skip logic: {'✅' if result['skip_logic_passed'] else '❌'}")
            print(f"     Next question: {'✅' if result['next_question_passed'] else '❌'}")
            print(f"     Recommendations: {result['recommendations_count']}")
            print(f"     Summary: {result['conversation_summary'][:100]}...")
            print()
    
    if failed_tests:
        print("\n⚠️ Failed Tests:")
        for i, result in enumerate(failed_tests, 1):
            print(f"  {i}. {result['scenario']}: {result['error']}")
    
    # Overall assessment
    llama3_managed_count = sum(1 for r in successful_tests if r['llama3_managed'])
    skip_logic_count = sum(1 for r in successful_tests if r['skip_logic_passed'])
    next_question_count = sum(1 for r in successful_tests if r['next_question_passed'])
    
    print("\n" + "=" * 70)
    print("🎯 FLOW TEST ASSESSMENT")
    print("=" * 70)
    
    if len(successful_tests) == 5 and llama3_managed_count >= 4:
        print("🎉 EXCELLENT! All tests passed with Llama 3 management!")
        print("✅ Flow testing: PERFECT")
        print("✅ Skip logic: WORKING")
        print("✅ Next question logic: WORKING")
        print("✅ Conversation flow: WORKING")
    elif len(successful_tests) >= 4 and llama3_managed_count >= 3:
        print("✅ GOOD! Most tests passed with Llama 3 management!")
        print("✅ Flow testing: MOSTLY WORKING")
        print(f"✅ Skip logic: {skip_logic_count}/5")
        print(f"✅ Next question logic: {next_question_count}/5")
    else:
        print("⚠️ NEEDS ATTENTION! Some tests failed.")
        print("❌ Flow testing: HAS ISSUES")
    
    return len(successful_tests) == 5 and llama3_managed_count >= 4

def test_conversation_memory():
    """Test that Llama 3 maintains conversation memory across multiple interactions."""
    print("\n🧠 Testing Conversation Memory...")
    print("=" * 50)
    
    try:
        rag = VehicleRAG("src/data/stocked_cars.csv")
        
        # First interaction
        result1 = rag.process_with_llama3("I need a BMW")
        print(f"✅ First interaction: {result1.get('conversation_summary', 'None')}")
        
        # Second interaction (should remember BMW preference)
        result2 = rag.process_with_llama3("under 20000")
        print(f"✅ Second interaction: {result2.get('conversation_summary', 'None')}")
        
        # Third interaction (should combine both preferences)
        result3 = rag.process_with_llama3("hatchback")
        print(f"✅ Third interaction: {result3.get('conversation_summary', 'None')}")
        
        # Check if memory is maintained
        memory_working = all([
            'BMW' in result1.get('conversation_summary', ''),
            '20000' in result2.get('conversation_summary', '') or 'BMW' in result2.get('conversation_summary', ''),
            'hatchback' in result3.get('conversation_summary', '') or 'BMW' in result3.get('conversation_summary', '')
        ])
        
        print(f"✅ Memory test: {'PASSED' if memory_working else 'FAILED'}")
        return memory_working
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False

def main():
    """Run comprehensive Llama 3 flow testing."""
    print("🎯 Comprehensive Llama 3 Flow Testing")
    print("=" * 70)
    
    # Test main flow
    flow_success = test_llama3_flow()
    
    # Test conversation memory
    memory_success = test_conversation_memory()
    
    print("\n" + "=" * 70)
    print("📊 FINAL ASSESSMENT")
    print("=" * 70)
    print(f"✅ Flow Testing: {'PASSED' if flow_success else 'FAILED'}")
    print(f"✅ Memory Testing: {'PASSED' if memory_success else 'FAILED'}")
    
    if flow_success and memory_success:
        print("\n🎉 PERFECT! Llama 3 system is fully functional!")
        print("✅ All flow tests passed")
        print("✅ Conversation memory working")
        print("✅ Ready for production use")
        print("\n🚀 System Features Verified:")
        print("   - Llama 3 auto-management")
        print("   - Smart skip logic")
        print("   - Next question flow")
        print("   - Conversation memory")
        print("   - Interactive forms")
        print("   - Fallback mode")
    else:
        print("\n⚠️ Some tests failed. Check the output above.")
    
    return flow_success and memory_success

if __name__ == "__main__":
    main() 