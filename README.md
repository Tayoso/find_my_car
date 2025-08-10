# 🚗 Car Sales Assistant

A smart car sales chatbot powered by Claude 3.5 Sonnet with access to car inventory data.

## ✨ Features

- **AI-Powered Recommendations**: Uses Claude 3.5 Sonnet for intelligent car recommendations
- **Real Inventory Data**: Access to 5,000+ cars with detailed information
- **Smart Filtering**: Filter by make, model, body type, price range, and fuel type
- **VRM Tracking**: Includes Vehicle Registration Mark (VRM) for each car
- **Streamlit Interface**: Beautiful, interactive web interface

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- `uv` package manager
- Anthropic API key

### Installation

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

## 🎯 Example Queries

Try these queries in the app:

- "I want a Nissan saloon between 12k and 20k"
- "Show me BMW cars"
- "Electric cars under £20k"
- "SUV vehicles available"
- "Petrol cars under £15k"
- "What's the average price?"

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

## 🎉 Success!

The app is now running and ready to help customers find their perfect car! 🚗✨



# QUESTIONS

- Question 1
User: I want a BMW below £20000.
Prompt 2: 6 to 8 passenger
Answer: Should be all BMWs below 20k based on memory. 
Prompt 3: SUV
Answer: should be all BMW SUVs below 20k. Make sure you test this.

- Question 2
User: "Show me electric vehicles for my family"
Prompt 2: 3 to 5 passenger
Answer: Should be all electrics with 3-5 passengers and family friendly based on memory. 
Prompt 3: Price range between 10 and 25k
Answer: should be all electrics with 3-5 passengers and family friendly and within 10-25k based on memory. Make sure you test this.
Prompt 4: Body type
Answer: should be all electrics with 3-5 passengers and family friendly and within 10-25k and the body type selected based on memory. Make sure you test this.

- Question 3
User: "i need a new diesel hatchback less than 17000"
Prompt 2: 3 to 5 passengers
Answer: Should be all diesel fuel type and hatchbacks below 17k based on memory. This is final answer given body type, fuel, type and price has bene given. Test that this works.

- Question 4
User: "i need a an Audi A3"
Prompt 2: 3 to 5 passengers
Answer: Should be all audi a3s, 3 to 5 passsengers i.e. saloon, hatchbacks that exists in the data. 
Prompt 2: Price range
Answer: Since price range wasnt defined ask for this and then filter with reasoning
Prompt 2: Body type
Answer: Since body type wasnt defined ask for this and then filter with reasoning. Answer should be audi a3s, with 3 to 5 passengers with the selected price range and body type
Test that this works. 

- Question 5
I want a bmw between 15 and 20k
How many passengers will be in the car? (3-5 or 6-8)
Any
What body type do you prefer? (e.g., SUV, hatchback, saloon, estate)
Ask about vehicles (e.g., 'I want a BMW below £20000')
SUV
Sorry, no vehicles match your criteria. Please try adjusting your preferences.


- Question 6
I want a bmw under 20k
How many passengers will be in the car? (3-5 or 6-8)
5
What body type do you prefer? (e.g., SUV, hatchback, saloon, estate)
Ask about vehicles (e.g., 'I want a BMW below £20000')
suv
Sorry, no vehicles match your criteria. Please try adjusting your preferences.
No recommednations is wrong, there should be.

You are going to build me a robust vehicle recommendation chatbot.
Ensure that you use the deepseek in my local machine C:\Users\beani\.ollama\models\blobs
Let your prompt for any user question be trying to figure out how to use langchain memory and reasoning with this llm to answer in this format based on the data in data/stocked-cars.csv. 
- What body type of car are you after 
- How many seats do you want the car to have?
- How much do you want to spend
- Final answer should be filtered well enough by the llm reasoning to ensure that the top 3 selections align with the intial user prompt and follow up prompts afterwards.
Set up a streamlit app

Wow me with a robust app based on the stuff I attached. test before sharing the details on how to run with me.

- Question 7
I want a nissan saloon between 12 and 20k

The end...
