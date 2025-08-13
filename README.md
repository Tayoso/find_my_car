# 🚗 Car Sales Assistant

A smart car sales chatbot powered by Claude 3.5 Sonnet with access to car inventory data.

**🔍 Python Logic**: Handles make, body type, price range filtering.
**🤖 LLM Intelligence**: Claude 3.5 Sonnet handles everything else, makes intelligent selections and rankings from filtered car list


## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd basic_chatbot
   ```

2. **Install dependencies with uv**
   ```bash
   uv sync
   ```

3. **Set up your API key**
   - Create a `.env` file in the project root
   - Add your Anthropic API key:
     ```
     ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
     ```

4. **Choose your chatbot version**

   **Original (Recommended):**
   ```bash
   uv run streamlit run claude_chatbot_app.py
   ```

   **LangChain Enhanced (Advanced Features):**
   ```bash
   uv run streamlit run langchain_enhanced_chatbot.py
   ```

   **LangChain Simple (Overkill for Claude 3.5):**
   ```bash
   uv run streamlit run langchain_simple_chatbot.py
   ```

5. **Open your browser**
   - Navigate to `http://localhost:8501`
   - Start asking about cars!

## 📊 Data

The app uses a comprehensive car inventory dataset (`data/stocked_cars.csv`) containing:
- **5,341 cars** with detailed information
- **Price range**: £5,500 - £51,850
- **Makes**: BMW, Mercedes-Benz, Audi, Ford, Volkswagen, and more
- **Body types**: SUV, Saloon, Hatchback, Estate, Convertible, Coupe, MPV
- **Fuel types**: Petrol, Diesel, Electric, Hybrid

## 🔧 Technical Details

- **Framework**: Streamlit
- **AI Model**: Claude 3.5 Sonnet (Anthropic)
- **Package Manager**: uv
- **Data Processing**: Pandas
- **Environment**: python-dotenv
- **LangChain**: Memory, conversation history, advanced prompting

## 🚀 LangChain Features

### **Simple Version** (`langchain_simple_chatbot.py`) - *Overkill for Claude 3.5*
- ⚠️ **Note**: Claude 3.5 Sonnet already has excellent context retention
- ⚠️ **Note**: LangChain memory is redundant for basic conversation
- ✅ **Advanced Prompting**: Dynamic prompt templates
- ✅ **Conversation Chains**: Structured conversation flow
- ✅ **Extensibility**: Easy to add LangChain features later

### **Enhanced Version** (`langchain_enhanced_chatbot.py`)
- ✅ All Simple features
- ✅ **Vector Database**: Semantic search capabilities
- ✅ **Knowledge Retrieval**: Advanced car matching
- ✅ **Full LangChain**: Complete framework integration

**💡 Recommendation**: Use the original `claude_chatbot_app.py` for most use cases. Claude 3.5's native context is sufficient for car sales conversations.**

**📖 See `LANGCHAIN_GUIDE.md` for detailed documentation**

## 📁 Project Structure

```
basic_chatbot/
├── claude_chatbot_app.py           # Original Streamlit application
├── langchain_simple_chatbot.py     # LangChain enhanced (recommended)
├── langchain_enhanced_chatbot.py   # Full LangChain features
├── data/
│   └── stocked_cars.csv            # Car inventory data
├── src/
│   ├── utils.py                    # Utility functions
│   └── prompts.py                  # System prompts
├── tests/                          # Test suite
├── pyproject.toml                  # Project dependencies
├── .env                           # Environment variables (API key)
├── README.md                      # This file
├── LANGCHAIN_GUIDE.md             # LangChain documentation
└── test_*.py                      # Test scripts
```

## 🧪 Testing

Run the test suite to verify everything is working:

```bash
# Quick test run
uv run python -m pytest tests/ -v

# Full test with coverage
uv run python -m pytest tests/ --cov=src --cov-report=term-missing
```

## 🔑 API Key Setup

1. Get your API key from [Anthropic Console](https://console.anthropic.com/)
2. Add it to your `.env` file:
   ```
   ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
   ```
3. The app will automatically load the key from the environment







