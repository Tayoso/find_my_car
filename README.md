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

4. **Run the app**
   ```bash
   uv run streamlit run claude_chatbot_app.py
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

## 📁 Project Structure

```
basic_chatbot/
├── claude_chatbot_app.py    # Main Streamlit application
├── data/
│   └── stocked_cars.csv     # Car inventory data
├── pyproject.toml           # Project dependencies
├── .env                     # Environment variables (API key)
├── README.md               # This file
└── test_*.py               # Test scripts
```

## 🧪 Testing

Run the test suite to verify everything is working:

```bash
uv run python test_app.py
```

## 🔑 API Key Setup

1. Get your API key from [Anthropic Console](https://console.anthropic.com/)
2. Add it to your `.env` file:
   ```
   ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
   ```
3. The app will automatically load the key from the environment



