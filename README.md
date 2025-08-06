# 🚗 Vehicle Recommendation System

A sophisticated vehicle recommendation chatbot powered by **Llama 3** with automatic conversation management and interactive forms.

## 🌟 Features

### 🤖 **Llama 3 Auto-Management**
- **Zero Manual Work** - Llama 3 handles everything automatically
- **Conversation Memory** - Remembers user preferences across interactions
- **State Management** - Tracks conversation state and context
- **Question Flow** - Determines next questions intelligently
- **Skip Logic** - Automatically skips already-answered questions
- **Context Building** - Maintains conversation context throughout
- **Structured Output** - Parses responses automatically

### 🎯 **Smart Recommendations**
- **Immediate Results** - Get recommendations instantly
- **Guided Questions** - Interactive forms for follow-up questions
- **Context-Aware** - Understands complex queries
- **Personalized** - Adapts to user preferences
- **Fallback Mode** - Works even when Llama 3 is unavailable

### 💬 **Interactive Experience**
- **Form-Based UI** - Clean, intuitive interface
- **Real-Time Chat** - Natural conversation flow
- **Visual Feedback** - Clear status indicators
- **Responsive Design** - Works on all devices

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Virtual environment (recommended)
- Ollama with Llama 3 model installed

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd find_my_car
   ```

2. **Setup Python environment:**
   ```bash
   # Create virtual environment
   python -m venv .venv
   
   # Activate virtual environment
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Install and run Ollama with Llama 3:**
   ```bash
   # Install Ollama (if not already installed)
   # Visit: https://ollama.ai/
   
   # Pull Llama 3 model
   ollama pull llama3
   ```

4. **Run the tests:**
   ```bash
   # Test Llama 3 connection
   python tests/test_llama3_connection.py
   
   # Test all questions from README
   python tests/test_all_questions.py
   ```

5. **Run the Streamlit app:**
   ```bash
   # Start the web application
   streamlit run src/apps/app_langchain_auto.py
   ```

6. **Open your browser:**
   Navigate to `http://localhost:8501` (or the URL shown in terminal)

## 🧪 Testing

### **Run All Tests**
```bash
# Test Llama 3 connection and functionality
python tests/test_llama3_connection.py

# Test all three questions with memory retention
python tests/test_all_questions.py
```

### **Expected Test Results**
- ✅ **Question 1**: BMW + £0-£20k + 6-8 passengers + SUV
- ✅ **Question 2**: Electric + 3-5 passengers + £10-£25k + SUV  
- ✅ **Question 3**: Diesel + Hatchback + £0-£17k + 3-5 passengers

## 🌐 Web Application

### **Start the App**
```bash
streamlit run src/apps/app_langchain_auto.py
```

### **Features**
- **Interactive Chat**: Natural conversation with the vehicle recommendation system
- **Memory Retention**: System remembers your preferences across conversation turns
- **Smart Filtering**: Combines make, fuel type, price, passengers, and body type
- **Real-time Results**: Get recommendations instantly with detailed vehicle information

### **Usage Examples**
1. **"I want a BMW below £20000"** → **"6 to 8 passengers"** → **"SUV"**
2. **"Show me electric vehicles for my family"** → **"3 to 5 passenger"** → **"Price range between 10 and 25k"** → **"SUV"**
3. **"i need a new diesel hatchback less than 17000"** → **"3 to 5 passengers"**

## 📁 Repository Structure

```
find_my_car/
├── src/
│   ├── core/
│   │   ├── vehicle_rag_enhanced.py           # Main Llama 3 RAG engine
│   │   └── vehicle_rag_langchain_auto.py     # Alternative RAG implementation
│   ├── apps/
│   │   └── app_langchain_auto.py             # Streamlit web interface
│   └── data/
│       └── stocked_cars.csv                   # Vehicle database
├── tests/
│   ├── test_all_questions.py                 # Comprehensive test suite
│   └── test_llama3_connection.py             # Llama 3 connection tests
├── README.md                                  # This file
└── requirements.txt                           # Python dependencies
```

## 🏗️ Architecture

### **Llama 3 Integration**

The system uses Llama 3 through Ollama for all semantic operations:

```python
from src.core.vehicle_rag_enhanced import Llama3SemanticRAG

# Initialize with Llama 3
rag = Llama3SemanticRAG("src/data/stocked_cars.csv")
```

### **Core Components**
- **Semantic Search**: Llama 3 analyzes all vehicles for relevance
- **Preference Extraction**: Llama 3 extracts user preferences from natural language
- **Reasoning Agent**: Llama 3 provides intelligent recommendations
- **Question Flow**: Llama 3 determines the next most relevant question

## 🎯 Usage Examples

### **Basic Query**
```
User: "I need a BMW under £20,000"
Assistant: [Provides recommendations + asks follow-up questions]
```

### **Complex Query**
```
User: "Show me electric vehicles for my family"
Assistant: [Provides recommendations + guided questions about passenger count, etc.]
```

### **Specific Requirements**
```
User: "i need a new diesel hatchback less than 17000"
Assistant: [Matches exact criteria + asks for passenger count]
```

## 🔧 Configuration

### **Environment Variables**
Create a `.env` file (optional):
```env
# For custom Ollama settings
OLLAMA_HOST=http://localhost:11434
```

### **Data Format**
The system expects a CSV with these columns:
- `VRM` - Vehicle registration
- `MAKE` - Manufacturer
- `MODEL` - Model name
- `BODY_TYPE` - Hatchback, SUV, etc.
- `FUEL_TYPE` - Diesel, Petrol, Electric, etc.
- `PRICE` - Price in pounds
- `MILEAGE` - Mileage
- `TRANSMISSION_TYPE` - Manual, Automatic
- `COLOUR` - Vehicle color

## 🧪 Testing

### **Run System Tests**
```bash
# Test Llama 3 integration
python tests/test_llama3_connection.py

# Test all scenarios from README
python tests/test_all_questions.py
```

### **Test Different Queries**
```bash
# Start the web application
streamlit run src/apps/app_langchain_auto.py
```

## 🚀 Llama 3 Features

### **Enhanced Capabilities**
- **Better Reasoning** - Improved logical thinking
- **Context Understanding** - Better conversation flow
- **Structured Output** - Reliable JSON parsing
- **Memory Management** - Persistent conversation state
- **Question Skipping** - Intelligent flow control

### **Auto-Management Benefits**
- **Zero Configuration** - Works out of the box
- **Self-Optimizing** - Adapts to user patterns
- **Error Recovery** - Graceful fallback handling
- **Performance** - Faster than previous versions

## 🔄 Conversation Flow

The system follows this intelligent flow:

1. **User Input** → Llama 3 extracts preferences
2. **Semantic Search** → Llama 3 finds relevant vehicles
3. **Reasoning** → Llama 3 provides recommendations
4. **Next Question** → Llama 3 determines what to ask next
5. **Memory Update** → System remembers preferences

### **Example Flows**

#### **Question 1: BMW Scenario**
```
User: "I want a BMW below £20000"
System: [Extracts: make=BMW, price<£20000]
System: [Asks: "How many people will usually be in the car?"]

User: "6 to 8 passenger"
System: [Extracts: passengers=6-8, body_type=SUV (automatic)]
System: [Provides: BMW SUVs under £20000 for 6-8 passengers]
```

#### **Question 2: Electric Family Scenario**
```
User: "Show me electric vehicles for my family"
System: [Extracts: fuel_type=electric, family_friendly=true]
System: [Asks: "How many people will usually be in the car?"]

User: "3 to 5 passenger"
System: [Extracts: passengers=3-5]
System: [Asks: "What's your budget range?"]

User: "Price range between 10 and 25k"
System: [Extracts: price_range=£10000-£25000]
System: [Asks: "Which body types do you prefer?"]

User: "SUV"
System: [Extracts: body_type=SUV]
System: [Provides: Electric SUVs for 3-5 passengers, £10-25k, family-friendly]
```

#### **Question 3: Diesel Hatchback Scenario**
```
User: "i need a new diesel hatchback less than 17000"
System: [Extracts: fuel_type=diesel, body_type=hatchback, price<£17000]
System: [Asks: "How many people will usually be in the car?"]

User: "3 to 5 passengers"
System: [Extracts: passengers=3-5]
System: [Provides: Diesel hatchbacks under £17000 for 3-5 passengers]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Meta** for releasing Llama 3 as open source
- **Ollama** for making local LLM deployment easy
- **Streamlit** for the excellent web framework
- **LangChain** for the powerful LLM orchestration

---

**🎯 Ready to find your perfect vehicle with Llama 3!** 
