#!/usr/bin/env python3
"""
Continuous Performance Testing for Vehicle Recommendation System
Ensures responses are always under 5 seconds and all 5 questions work correctly
"""

import time
import sys
import os
from src.core.vehicle_rag_langchain_auto import FastVehicleRAG

def test_single_query(query: str, expected_response_time: float = 5.0) -> bool:
    """Test a single query and return True if it passes performance requirements."""
    try:
        rag = FastVehicleRAG(csv_path="src/data/stocked_cars.csv")
        
        start_time = time.time()
        result = rag.process_with_langchain(query)
        response_time = time.time() - start_time
        
        # Check performance
        if response_time > expected_response_time:
            print(f"❌ FAIL: Query '{query}' took {response_time:.2f}s (exceeded {expected_response_time}s)")
            return False
        
        # Check if recommendations are provided
        if not result.get("recommendations"):
            print(f"❌ FAIL: Query '{query}' returned no recommendations")
            return False
        
        print(f"✅ PASS: Query '{query}' completed in {response_time:.2f}s")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Query '{query}' failed with error: {e}")
        return False

def test_5_questions_flow():
    """Test the 5 questions flow with different scenarios."""
    print("🚗 Testing 5 Questions Flow Performance")
    print("=" * 50)
    
    test_scenarios = [
        {
            "name": "Family SUV Scenario",
            "queries": [
                "I need a family car for 5 people",
                "SUV would be good",
                "Budget is under £20,000"
            ]
        },
        {
            "name": "BMW Hatchback Scenario", 
            "queries": [
                "I want a BMW hatchback",
                "Price under £25,000",
                "Just for me and my partner"
            ]
        },
        {
            "name": "Electric Vehicle Scenario",
            "queries": [
                "Show me electric vehicles",
                "For my family of 4",
                "Budget around £30,000"
            ]
        },
        {
            "name": "City Car Scenario",
            "queries": [
                "I need a small car for city driving",
                "Hatchback would be perfect", 
                "Budget under £15,000"
            ]
        },
        {
            "name": "Luxury SUV Scenario",
            "queries": [
                "I want a luxury SUV for my family",
                "Mercedes or BMW",
                "Budget £40,000-£60,000",
                "For 6 people"
            ]
        }
    ]
    
    total_passed = 0
    total_queries = 0
    
    for scenario in test_scenarios:
        print(f"\n🔍 {scenario['name']}")
        print("-" * 30)
        
        scenario_passed = True
        for query in scenario['queries']:
            total_queries += 1
            if test_single_query(query):
                total_passed += 1
            else:
                scenario_passed = False
        
        if scenario_passed:
            print(f"✅ {scenario['name']} PASSED")
        else:
            print(f"❌ {scenario['name']} FAILED")
    
    print(f"\n🎯 Overall Results: {total_passed}/{total_queries} queries passed")
    return total_passed == total_queries

def continuous_performance_test(duration_minutes: int = 5):
    """Run continuous performance tests for specified duration."""
    print(f"🔄 Running continuous performance test for {duration_minutes} minutes")
    print("=" * 60)
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    test_queries = [
        "I need a family car",
        "Show me BMW vehicles",
        "Electric cars under £30,000",
        "SUV for 6 people",
        "Hatchback under £20,000",
        "Luxury car for 2 people",
        "Hybrid vehicles",
        "Compact city car",
        "Family SUV with good fuel economy",
        "Sports car under £25,000"
    ]
    
    query_count = 0
    passed_count = 0
    failed_count = 0
    
    while time.time() < end_time:
        for query in test_queries:
            if time.time() >= end_time:
                break
                
            query_count += 1
            if test_single_query(query):
                passed_count += 1
            else:
                failed_count += 1
            
            # Small delay between queries
            time.sleep(0.1)
    
    total_time = time.time() - start_time
    success_rate = (passed_count / query_count * 100) if query_count > 0 else 0
    
    print(f"\n📊 Continuous Test Results:")
    print(f"   Duration: {total_time:.1f} seconds")
    print(f"   Total Queries: {query_count}")
    print(f"   Passed: {passed_count}")
    print(f"   Failed: {failed_count}")
    print(f"   Success Rate: {success_rate:.1f}%")
    
    return success_rate >= 95.0

def main():
    """Main test function."""
    print("🚗 Vehicle Recommendation System Performance Testing")
    print("=" * 60)
    
    # Test 1: 5 Questions Flow
    print("\n1️⃣ Testing 5 Questions Flow...")
    flow_success = test_5_questions_flow()
    
    # Test 2: Continuous Performance
    print("\n2️⃣ Testing Continuous Performance...")
    continuous_success = continuous_performance_test(duration_minutes=2)
    
    # Final Results
    print("\n" + "=" * 60)
    print("🎯 FINAL TEST RESULTS")
    print("=" * 60)
    
    if flow_success and continuous_success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ 5 Questions Flow: PASSED")
        print("✅ Continuous Performance: PASSED")
        print("✅ Response times consistently under 5 seconds")
        print("✅ System ready for production")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        if not flow_success:
            print("❌ 5 Questions Flow: FAILED")
        if not continuous_success:
            print("❌ Continuous Performance: FAILED")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 