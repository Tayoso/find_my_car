#!/usr/bin/env python3
"""
Test Claude API connection
"""

import os
from dotenv import load_dotenv
from anthropic import Anthropic

def test_claude_connection():
    """Test Claude API connection"""
    print("🧪 Testing Claude API connection...")
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("❌ ANTHROPIC_API_KEY not set in .env file")
        print("   Please add your API key to .env file:")
        print("   ANTHROPIC_API_KEY=sk-ant-api03-...")
        return False
    
    try:
        # Test API connection
        client = Anthropic(api_key=api_key)
        
        # Simple test message
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=50,
            messages=[{"role": "user", "content": "Say 'Hello, Claude is working!'"}]
        )
        
        print("✅ Claude API connection successful!")
        print(f"📝 Response: {response.content[0].text}")
        return True
        
    except Exception as e:
        print(f"❌ Claude API connection failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚗 Claude API Connection Test")
    print("=" * 40)
    
    success = test_claude_connection()
    
    if success:
        print("\n🎉 All tests passed! Your API key is working.")
        print("   You can now run the Streamlit app.")
    else:
        print("\n⚠️  Please fix the API key issue before running the app.")

if __name__ == "__main__":
    main()
