#!/usr/bin/env python3
"""
Test script for the Car Sales Assistant app
"""

import pandas as pd
from anthropic import Anthropic
import os

def test_data_loading():
    """Test if car data loads correctly"""
    print("🧪 Testing data loading...")
    try:
        df = pd.read_csv('data/stocked_cars.csv')
        print(f"✅ Data loaded successfully: {len(df)} cars")
        print(f"📊 Available makes: {', '.join(df['MAKE'].unique()[:5])}...")
        print(f"🚗 Body types: {', '.join(df['BODY_TYPE'].unique())}")
        print(f"💰 Price range: £{df['PRICE'].min():,} - £{df['PRICE'].max():,}")
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_anthropic_connection():
    """Test if Anthropic API can be imported"""
    print("\n🧪 Testing Anthropic library...")
    try:
        from anthropic import Anthropic
        print("✅ Anthropic library imported successfully")
        return True
    except Exception as e:
        print(f"❌ Anthropic import failed: {e}")
        return False

def test_car_filtering():
    """Test the car filtering logic"""
    print("\n🧪 Testing car filtering logic...")
    try:
        df = pd.read_csv('data/stocked_cars.csv')
        
        # Test BMW filtering
        bmw_cars = df[df['MAKE'] == 'BMW']
        print(f"✅ BMW cars found: {len(bmw_cars)}")
        
        # Test SUV filtering
        suv_cars = df[df['BODY_TYPE'] == 'suv']
        print(f"✅ SUV cars found: {len(suv_cars)}")
        
        # Test price filtering
        affordable_cars = df[df['PRICE'] <= 15000]
        print(f"✅ Cars under £15k: {len(affordable_cars)}")
        
        return True
    except Exception as e:
        print(f"❌ Car filtering failed: {e}")
        return False

def test_streamlit_imports():
    """Test if Streamlit can be imported"""
    print("\n🧪 Testing Streamlit imports...")
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
        return True
    except Exception as e:
        print(f"❌ Streamlit import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚗 Car Sales Assistant - API Test Suite")
    print("=" * 50)
    
    tests = [
        test_data_loading,
        test_anthropic_connection,
        test_car_filtering,
        test_streamlit_imports,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The app should work correctly.")
        print("\n🌐 To run the app:")
        print("   uv run streamlit run claude_chatbot_app.py")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    main()
