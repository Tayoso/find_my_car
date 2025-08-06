#!/usr/bin/env python3
"""
Test that Llama 3 calls are working properly
"""

import os
import sys
import time
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

def test_llama3_connection():
    """Test if Llama 3 is available and working."""
    print("🧪 Testing Llama 3 Connection")
    print("=" * 50)
    
    try:
        from langchain_community.llms import Ollama
        llm = Ollama(model="llama3")
        
        # Simple test prompt
        test_prompt = "Hello, can you respond with 'Llama 3 is working'?"
        print(f"Testing with prompt: {test_prompt}")
        
        start_time = time.time()
        response = llm.invoke(test_prompt)
        response_time = time.time() - start_time
        
        print(f"✅ Llama 3 Response: {response}")
        print(f"✅ Response time: {response_time:.3f}s")
        print("✅ Llama 3 is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Llama 3 connection failed: {e}")
        print("Please make sure Ollama is running with llama3 model installed")
        return False

def test_enhanced_rag_llama3():
    """Test the enhanced RAG system with Llama 3."""
    print("\n🧪 Testing Enhanced RAG with Llama 3")
    print("=" * 50)
    
    try:
        from src.core.vehicle_rag_enhanced import Llama3SemanticRAG
        
        # Initialize RAG system
        rag = Llama3SemanticRAG("src/data/stocked_cars.csv")
        
        if not rag.semantic_ready:
            print("❌ Llama 3 not available in RAG system")
            return False
        
        # Test a simple query
        test_query = "I want a BMW"
        print(f"Testing query: {test_query}")
        
        start_time = time.time()
        result = rag.process_with_llama3_semantic(test_query)
        total_time = time.time() - start_time
        
        print(f"✅ Total processing time: {total_time:.3f}s")
        print(f"✅ Semantic matches: {result['semantic_matches']}")
        print(f"✅ Pattern matches: {result['pattern_matches']} (should be 0)")
        print(f"✅ Reasoning used: {result['reasoning_used']}")
        print(f"✅ Semantic used: {result['semantic_used']}")
        print(f"✅ Summary: {result['conversation_summary']}")
        print(f"✅ Recommendations: {result['recommendations'][:200]}...")
        
        if result['next_question']:
            print(f"✅ Next question: {result['next_question']['question']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced RAG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_preference_extraction():
    """Test Llama 3 preference extraction."""
    print("\n🧪 Testing Llama 3 Preference Extraction")
    print("=" * 50)
    
    try:
        from src.core.vehicle_rag_enhanced import Llama3SemanticRAG
        
        rag = Llama3SemanticRAG("src/data/stocked_cars.csv")
        
        test_inputs = [
            "I want a BMW below £20000",
            "Show me electric vehicles for my family",
            "i need a new diesel hatchback less than 17000"
        ]
        
        for i, test_input in enumerate(test_inputs, 1):
            print(f"Test {i}: {test_input}")
            try:
                preferences = rag.extract_preferences_with_llama3(test_input)
                print(f"  ✅ Preferences: {preferences}")
            except Exception as e:
                print(f"  ❌ Failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Preference extraction test failed: {e}")
        return False

def test_semantic_search():
    """Test Llama 3 semantic search."""
    print("\n🧪 Testing Llama 3 Semantic Search")
    print("=" * 50)
    
    try:
        from src.core.vehicle_rag_enhanced import Llama3SemanticRAG
        
        rag = Llama3SemanticRAG("src/data/stocked_cars.csv")
        
        test_queries = [
            "I want a BMW",
            "Show me electric cars",
            "I need a family car"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"Test {i}: {query}")
            try:
                results = rag.semantic_search_with_llama3(query, k=3)
                print(f"  ✅ Found {len(results)} matches")
                for j, result in enumerate(results[:2], 1):
                    print(f"    Match {j}: {result['metadata']['MAKE']} {result['metadata']['MODEL']}")
            except Exception as e:
                print(f"  ❌ Failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Semantic search test failed: {e}")
        return False

def main():
    """Run all Llama 3 tests."""
    print("🚀 Testing Llama 3 Connection and Functionality")
    print("=" * 70)
    
    start_time = time.time()
    
    # Test all components
    connection_passed = test_llama3_connection()
    rag_passed = test_enhanced_rag_llama3()
    extraction_passed = test_preference_extraction()
    search_passed = test_semantic_search()
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 LLAMA 3 TEST RESULTS")
    print("=" * 70)
    print(f"✅ Llama 3 connection: {'PASS' if connection_passed else 'FAIL'}")
    print(f"✅ Enhanced RAG with Llama 3: {'PASS' if rag_passed else 'FAIL'}")
    print(f"✅ Preference extraction: {'PASS' if extraction_passed else 'FAIL'}")
    print(f"✅ Semantic search: {'PASS' if search_passed else 'FAIL'}")
    print(f"⏱️ Total time: {total_time:.2f}s")
    
    all_passed = all([connection_passed, rag_passed, extraction_passed, search_passed])
    
    if all_passed:
        print("\n🎉 ALL LLAMA 3 TESTS PASSED! Llama 3 is working correctly.")
    else:
        print("\n⚠️ Some Llama 3 tests failed. Check the results above.")
    
    return all_passed

if __name__ == "__main__":
    main() 