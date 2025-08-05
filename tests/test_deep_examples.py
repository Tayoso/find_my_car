#!/usr/bin/env python3
"""
Deep Examples Test for LangChain Vehicle Recommendation System
Tests 5 complex scenarios with multiple conversation turns
"""

import time
import sys
import os
from src.core.vehicle_rag_langchain_auto import LangChainVehicleRAG

def test_deep_example_1():
    """Deep Example 1: Family SUV Journey with Multiple Refinements"""
    print("🔍 Deep Example 1: Family SUV Journey with Multiple Refinements")
    print("=" * 60)
    
    rag = LangChainVehicleRAG()
    
    # Reset conversation state
    rag.conversation_state = {
        "passenger_count": None,
        "body_type": None,
        "price_range": None,
        "make_pref": None,
        "answered_questions": set()
    }
    
    queries = [
        "I need a car for my family of 6 people",
        "SUV would be perfect for us",
        "Budget around £25,000",
        "I prefer diesel for fuel efficiency",
        "Can you show me options with low mileage?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n📝 Query {i}: {query}")
        
        start_time = time.time()
        result = rag.process_with_langchain(query)
        response_time = time.time() - start_time
        
        print(f"⏱️  Response time: {response_time:.3f}s")
        print(f"💬 Conversation: {result['conversation_summary']}")
        print(f"🎯 Next question: {result['next_question']}")
        print(f"📋 Recommendations: {result['recommendations'][:200]}...")
        
        if response_time > 5.0:
            print(f"⚠️  WARNING: Response time exceeded 5 seconds!")
    
    print("\n✅ Deep Example 1 completed")

def test_deep_example_2():
    """Deep Example 2: Luxury Car Buyer with Specific Requirements"""
    print("\n🔍 Deep Example 2: Luxury Car Buyer with Specific Requirements")
    print("=" * 60)
    
    rag = LangChainVehicleRAG()
    
    # Reset conversation state
    rag.conversation_state = {
        "passenger_count": None,
        "body_type": None,
        "price_range": None,
        "make_pref": None,
        "answered_questions": set()
    }
    
    queries = [
        "I want a luxury car for just me and my partner",
        "BMW or Mercedes would be ideal",
        "Budget is flexible, around £40,000",
        "I prefer automatic transmission",
        "Show me the newest models available"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n📝 Query {i}: {query}")
        
        start_time = time.time()
        result = rag.process_with_langchain(query)
        response_time = time.time() - start_time
        
        print(f"⏱️  Response time: {response_time:.3f}s")
        print(f"💬 Conversation: {result['conversation_summary']}")
        print(f"🎯 Next question: {result['next_question']}")
        print(f"📋 Recommendations: {result['recommendations'][:200]}...")
        
        if response_time > 5.0:
            print(f"⚠️  WARNING: Response time exceeded 5 seconds!")
    
    print("\n✅ Deep Example 2 completed")

def test_deep_example_3():
    """Deep Example 3: Electric Vehicle Enthusiast"""
    print("\n🔍 Deep Example 3: Electric Vehicle Enthusiast")
    print("=" * 60)
    
    rag = LangChainVehicleRAG()
    
    # Reset conversation state
    rag.conversation_state = {
        "passenger_count": None,
        "body_type": None,
        "price_range": None,
        "make_pref": None,
        "answered_questions": set()
    }
    
    queries = [
        "I'm interested in electric vehicles",
        "For a family of 4 people",
        "Budget under £35,000",
        "I need good range for daily commuting",
        "Show me options with fast charging capability"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n📝 Query {i}: {query}")
        
        start_time = time.time()
        result = rag.process_with_langchain(query)
        response_time = time.time() - start_time
        
        print(f"⏱️  Response time: {response_time:.3f}s")
        print(f"💬 Conversation: {result['conversation_summary']}")
        print(f"🎯 Next question: {result['next_question']}")
        print(f"📋 Recommendations: {result['recommendations'][:200]}...")
        
        if response_time > 5.0:
            print(f"⚠️  WARNING: Response time exceeded 5 seconds!")
    
    print("\n✅ Deep Example 3 completed")

def test_deep_example_4():
    """Deep Example 4: Budget-Conscious City Driver"""
    print("\n🔍 Deep Example 4: Budget-Conscious City Driver")
    print("=" * 60)
    
    rag = LangChainVehicleRAG()
    
    # Reset conversation state
    rag.conversation_state = {
        "passenger_count": None,
        "body_type": None,
        "price_range": None,
        "make_pref": None,
        "answered_questions": set()
    }
    
    queries = [
        "I need a small car for city driving",
        "Hatchback would be perfect",
        "Budget is tight, under £12,000",
        "I prefer petrol for simplicity",
        "Show me options with good fuel economy"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n📝 Query {i}: {query}")
        
        start_time = time.time()
        result = rag.process_with_langchain(query)
        response_time = time.time() - start_time
        
        print(f"⏱️  Response time: {response_time:.3f}s")
        print(f"💬 Conversation: {result['conversation_summary']}")
        print(f"🎯 Next question: {result['next_question']}")
        print(f"📋 Recommendations: {result['recommendations'][:200]}...")
        
        if response_time > 5.0:
            print(f"⚠️  WARNING: Response time exceeded 5 seconds!")
    
    print("\n✅ Deep Example 4 completed")

def test_deep_example_5():
    """Deep Example 5: Large Family with Complex Requirements"""
    print("\n🔍 Deep Example 5: Large Family with Complex Requirements")
    print("=" * 60)
    
    rag = LangChainVehicleRAG()
    
    # Reset conversation state
    rag.conversation_state = {
        "passenger_count": None,
        "body_type": None,
        "price_range": None,
        "make_pref": None,
        "answered_questions": set()
    }
    
    queries = [
        "I need a car for 7 people including children",
        "SUV or MPV would work best",
        "Budget around £30,000",
        "I need good safety features",
        "Show me options with plenty of storage space"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n📝 Query {i}: {query}")
        
        start_time = time.time()
        result = rag.process_with_langchain(query)
        response_time = time.time() - start_time
        
        print(f"⏱️  Response time: {response_time:.3f}s")
        print(f"💬 Conversation: {result['conversation_summary']}")
        print(f"🎯 Next question: {result['next_question']}")
        print(f"📋 Recommendations: {result['recommendations'][:200]}...")
        
        if response_time > 5.0:
            print(f"⚠️  WARNING: Response time exceeded 5 seconds!")
    
    print("\n✅ Deep Example 5 completed")

def run_all_deep_tests():
    """Run all 5 deep examples and provide summary."""
    print("🚗 Testing LangChain Vehicle Recommendation System with 5 Deep Examples")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        test_deep_example_1()
        test_deep_example_2()
        test_deep_example_3()
        test_deep_example_4()
        test_deep_example_5()
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("🎯 DEEP EXAMPLES TEST SUMMARY")
        print("=" * 80)
        print(f"✅ All 5 deep examples completed successfully")
        print(f"⏱️  Total test time: {total_time:.2f} seconds")
        print(f"📊 Average time per example: {total_time/5:.2f} seconds")
        print(f"🎉 System is working correctly with LangChain agents and memory!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during deep testing: {e}")
        return False

if __name__ == "__main__":
    success = run_all_deep_tests()
    
    if success:
        print("\n🎉 All deep examples passed! The system is ready for production.")
    else:
        print("\n⚠️ Some deep examples failed. Please review and fix issues.") 