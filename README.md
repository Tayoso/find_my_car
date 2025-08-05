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

3. **Run the test script:**
   ```bash
   # Test all three questions from README
   python test_all_questions.py
   ```

4. **Run the Streamlit app:**
   ```bash
   # Start the web application
   python -m streamlit run src/apps/app_langchain_auto.py
   ```

5. **Open your browser:**
   Navigate to `http://localhost:8501` (or the URL shown in terminal)

## 🧪 Testing

### **Run All Tests**
```bash
# Test all three questions with memory retention
python tests/test_all_questions.py
```

### **Individual Test Scripts**
```bash
# Test BMW scenario (Question 1)
python tests/test_bmw_scenario.py

# Test Electric vehicle scenario (Question 2)
python tests/test_electric_scenario.py

# Test Diesel hatchback scenario (Question 3)
python tests/test_question3.py

# Test flexible price ranges
python tests/test_flexible_price.py
```

### **Expected Test Results**
- ✅ **Question 1**: BMW + £0-£20k + 6-8 passengers + SUV
- ✅ **Question 2**: Electric + 3-5 passengers + £10-£25k + SUV  
- ✅ **Question 3**: Diesel + Hatchback + £0-£17k + 3-5 passengers

## 🌐 Web Application

### **Start the App**
```bash
python -m streamlit run src/apps/app_langchain_auto.py
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
│   │   └── vehicle_rag_langchain_auto.py    # Main RAG engine
│   ├── apps/
│   │   └── app_langchain_auto.py            # Streamlit web interface
│   └── data/
│       └── stocked_cars.csv                  # Vehicle database
├── tests/
│   ├── test_all_questions.py                # Comprehensive test suite
│   ├── test_bmw_scenario.py                 # BMW scenario tests
│   ├── test_electric_scenario.py            # Electric vehicle tests
│   ├── test_question3.py                    # Diesel hatchback tests
│   ├── test_flexible_price.py               # Price range tests
│   ├── test_performance.py                  # Performance tests
│   ├── test_deep_examples.py                # Deep conversation tests
│   └── test_llama3_flow.py                  # Llama3 integration tests
├── docs/
│   ├── TECHNICAL_DOCUMENTATION.md           # Technical deep dive
│   ├── REFACTOR_SUMMARY.md                  # Refactoring history
│   └── PERFORMANCE_SUMMARY.md               # Performance analysis
├── README.md                                 # This file
├── requirements.txt                          # Python dependencies
└── main.py                                  # Entry point
```

## 🏗️ Architecture

### **Llama 3 Integration**

The system supports two Llama 3 deployment options:

#### **Option 1: Ollama (Recommended)**
```python
from core.vehicle_rag import VehicleRAG

# Initialize with Ollama
rag = VehicleRAG("src/data/stocked_cars.csv")
```

#### **Option 2: LlamaCpp (Local GGUF)**
```python
from core.vehicle_rag import VehicleRAG

# Initialize with LlamaCpp
rag = VehicleRAG("src/data/stocked_cars.csv", use_llamacpp=True)
```

**For LlamaCpp, set your model path:**
```bash
export LLAMA_MODEL_PATH="path/to/llama-3-70b-instruct.Q5_K_M.gguf"
```

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
Create a `.env` file:
```env
# For LlamaCpp (optional)
LLAMA_MODEL_PATH=path/to/your/llama-3-model.gguf
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
uv run python -c "
import sys; sys.path.append('src')
from core.vehicle_rag import VehicleRAG
rag = VehicleRAG('src/data/stocked_cars.csv')
result = rag.process_with_llama3('i need a diesel hatchback under 17000')
print('✅ Llama 3 working:', result['llama3_managed'])
"
```

### **Test Different Queries**
```bash
# Test various scenarios
uv run streamlit run main.py
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

## 🔄 Migration from Llama 2

The system has been upgraded from Llama 2 to Llama 3:

### **Key Changes**
- ✅ **Better Performance** - Llama 3 is more capable
- ✅ **Improved Reasoning** - Better understanding of complex queries
- ✅ **Enhanced Context** - Better conversation memory
- ✅ **Structured Output** - More reliable JSON parsing

### **Backward Compatibility**
- ✅ **Same Interface** - No code changes needed
- ✅ **Fallback Support** - Works when Llama 3 unavailable
- ✅ **Data Compatibility** - Same CSV format

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



The flow is as below: Number of People in the car >> price range >> body type ...

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
The end...