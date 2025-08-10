#!/usr/bin/env python3
"""
Test script to check Streamlit and Anthropic imports
"""

import sys
import os

def test_imports():
    """Test all imports"""
    print("🧪 Testing imports...")
    print(f"Python executable: {sys.executable}")
    print(f"Python path: {sys.path[:3]}...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except Exception as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except Exception as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        from anthropic import Anthropic
        print("✅ Anthropic imported successfully")
        print(f"   Anthropic version: {Anthropic.__version__}")
    except Exception as e:
        print(f"❌ Anthropic import failed: {e}")
        return False
    
    try:
        from dotenv import load_dotenv
        print("✅ python-dotenv imported successfully")
    except Exception as e:
        print(f"❌ python-dotenv import failed: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    print("🚗 Streamlit Import Test")
    print("=" * 40)
    
    success = test_imports()
    
    if success:
        print("\n🎉 All imports successful!")
        print("   The app should work correctly.")
    else:
        print("\n⚠️  Some imports failed.")
        print("   Please check the package installation.")

if __name__ == "__main__":
    main()
