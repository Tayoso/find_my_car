#!/usr/bin/env python3
"""
Main Vehicle Recommendation System using Llama 3
This is the core RAG system that handles everything automatically
"""

import os
import pandas as pd
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json

# Load environment variables
load_dotenv()

# Define structured output for Llama 3 to handle automatically
class VehicleRecommendation(BaseModel):
    recommendations: str = Field(description="Vehicle recommendations in markdown format")
    next_question: str = Field(description="Next question to ask user, or 'none' if complete")
    question_type: str = Field(description="Type: radio, multiselect, text, or none")
    options: List[str] = Field(default=[], description="Options for radio/multiselect")
    conversation_summary: str = Field(description="Summary of what we know about user preferences")
    should_skip_questions: List[str] = Field(default=[], description="Questions to skip based on user input")

class VehicleRAG:
    def __init__(self, csv_path: str = "src/data/stocked_cars.csv", use_llamacpp: bool = False):
        """Initialize the Vehicle RAG system with Llama 3 auto-management."""
        self.csv_path = csv_path
        self.llm = None
        self.memory = ConversationBufferMemory(return_messages=True)
        self.output_parser = PydanticOutputParser(pydantic_object=VehicleRecommendation)
        self.use_llamacpp = use_llamacpp
        self.setup_llama3()
        
    def setup_llama3(self):
        """Setup Llama 3 via Ollama or LlamaCpp."""
        if self.use_llamacpp:
            self.setup_llamacpp()
        else:
            self.setup_ollama()
        
        self.load_vehicle_data()
        
    def setup_ollama(self):
        """Setup Llama 3 via Ollama."""
        try:
            from langchain_community.llms import Ollama
            
            self.llm = Ollama(
                model="llama3",  # Updated to Llama 3
                temperature=0.7,
                base_url="http://localhost:11434"
            )
            print("✅ Llama 3 (Ollama) setup complete")
            
        except Exception as e:
            print(f"❌ Llama 3 (Ollama) setup error: {e}")
            print("⚠️ Make sure Ollama is running with: ollama run llama3")
            self.llm = None
    
    def setup_llamacpp(self):
        """Setup Llama 3 via LlamaCpp for local GGUF models."""
        try:
            from langchain_community.llms import LlamaCpp
            from langchain_core.prompts import ChatPromptTemplate
            
            # You can specify the path to your GGUF model
            model_path = os.getenv("LLAMA_MODEL_PATH", "llama-3-70b-instruct.Q5_K_M.gguf")
            
            self.llm = LlamaCpp(
                model_path=model_path,
                temperature=0.7,
                max_tokens=2000,
                n_ctx=8192,
                verbose=False
            )
            print(f"✅ Llama 3 (LlamaCpp) setup complete with {model_path}")
            
        except Exception as e:
            print(f"❌ Llama 3 (LlamaCpp) setup error: {e}")
            print("⚠️ Make sure you have the GGUF model file available")
            self.llm = None
        
    def load_vehicle_data(self):
        """Load vehicle data for Llama 3 to use."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file {self.csv_path} not found")
        
        df = pd.read_csv(self.csv_path)
        print(f"✅ Loaded {len(df)} vehicles for Llama 3")
        
        # Create documents for Llama 3
        self.documents = []
        for _, row in df.iterrows():
            vehicle_text = f"""
            Make: {row.get('MAKE', 'Unknown')}
            Model: {row.get('MODEL', 'Unknown')}
            Price: £{row.get('PRICE', 0):,}
            Body Style: {row.get('BODY_TYPE', 'Unknown')}
            Fuel Type: {row.get('FUEL_TYPE', 'Unknown')}
            Transmission: {row.get('TRANSMISSION_TYPE', 'Unknown')}
            Mileage: {row.get('MILEAGE', 0):,} miles
            Color: {row.get('COLOUR', 'Unknown')}
            """
            
            self.documents.append(Document(
                page_content=vehicle_text.strip(),
                metadata=row.to_dict()
            ))
        
        print(f"✅ Llama 3 processed {len(self.documents)} vehicles")
        
    def create_llama3_prompt(self) -> PromptTemplate:
        """Create prompt for Llama 3 to handle the conversation flow automatically."""
        template = """You are an intelligent vehicle recommendation assistant powered by Llama 3. You help users find the perfect car.

Available vehicle data:
{vehicle_data}

User query: {user_input}

Conversation history: {history}

Available questions to ask:
1. passenger_count: "How many people will usually be in the car?" (radio: Up to 2, 3-5, 6-8)
2. body_type: "Which body types do you like?" (multiselect: Hatchback, Estate, SUV, Coupe, Convertible, Saloon, MPV)
3. price_range: "Price Range:" (text: e.g., £10,000 - £20,000)
4. make_pref: "Make preference?" (text: e.g., Toyota, BMW, no preference)

You will automatically:
- Analyze user input for mentioned preferences
- Skip questions already answered
- Maintain conversation context
- Provide vehicle recommendations
- Determine next question

Provide your response in this exact JSON format:
{{
    "recommendations": "## 🚗 **Immediate Recommendations**\\n\\n1. **Vehicle Name**\\n   - Price: £X,XXX\\n   - Key Features: [features]\\n   - Why it's recommended: [explanation]\\n\\n2. **Vehicle Name**\\n   - Price: £X,XXX\\n   - Key Features: [features]\\n   - Why it's recommended: [explanation]\\n\\n3. **Vehicle Name**\\n   - Price: £X,XXX\\n   - Key Features: [features]\\n   - Why it's recommended: [explanation]",
    "next_question": "The next question to ask, or 'none' if complete",
    "question_type": "radio|multiselect|text|none",
    "options": ["option1", "option2"],
    "conversation_summary": "Brief summary of what we know about user preferences",
    "should_skip_questions": ["question1", "question2"]
}}

{format_instructions}"""
        
        return PromptTemplate(
            template=template,
            input_variables=["vehicle_data", "user_input", "history"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
    
    def process_with_llama3(self, user_input: str) -> Dict[str, Any]:
        """Let Llama 3 handle everything automatically."""
        if self.llm is None:
            return self._fallback_response(user_input)
        
        try:
            # Get relevant vehicles for context
            relevant_vehicles = self._get_relevant_vehicles(user_input, k=5)
            vehicle_data = "\n\n".join([doc.page_content for doc in relevant_vehicles])
            
            # Get conversation history from memory
            history = self.memory.buffer if hasattr(self.memory, 'buffer') else ""
            
            # Create Llama 3 prompt
            prompt = self.create_llama3_prompt()
            
            # Process with Llama 3
            chain = prompt | self.llm | self.output_parser
            
            # Process with Llama 3
            result = chain.invoke({
                "vehicle_data": vehicle_data,
                "user_input": user_input,
                "history": history
            })
            
            # Handle memory automatically
            self.memory.save_context(
                {"input": user_input},
                {"output": result.recommendations}
            )
            
            return {
                "recommendations": result.recommendations,
                "next_question": result.next_question,
                "question_type": result.question_type,
                "options": result.options,
                "conversation_summary": result.conversation_summary,
                "skipped_questions": result.should_skip_questions,
                "llama3_managed": True
            }
            
        except Exception as e:
            print(f"❌ Llama 3 processing error: {e}")
            return self._fallback_response(user_input)
    
    def _get_relevant_vehicles(self, query: str, k: int = 5) -> List[Document]:
        """Simple vehicle matching for context."""
        query_lower = query.lower()
        relevant = []
        
        for doc in self.documents:
            content_lower = doc.page_content.lower()
            score = 0
            
            # Simple matching
            if any(word in query_lower for word in ['electric', 'ev']) and 'electric' in content_lower:
                score += 10
            if any(word in query_lower for word in ['hybrid']) and 'hybrid' in content_lower:
                score += 10
            if any(word in query_lower for word in ['diesel']) and 'diesel' in content_lower:
                score += 10
            if any(word in query_lower for word in ['petrol']) and 'petrol' in content_lower:
                score += 8
            if any(word in query_lower for word in ['suv']) and 'suv' in content_lower:
                score += 8
            if any(word in query_lower for word in ['hatchback']) and 'hatchback' in content_lower:
                score += 8
            if any(word in query_lower for word in ['bmw', 'mercedes', 'audi']) and any(brand in content_lower for brand in ['bmw', 'mercedes', 'audi']):
                score += 6
            if any(word in query_lower for word in ['family']) and any(brand in content_lower for brand in ['toyota', 'honda']):
                score += 6
            if any(word in query_lower for word in ['under', 'budget', 'cheap', 'less than']):
                for line in content_lower.split('\n'):
                    if 'price:' in line:
                        try:
                            price_str = line.split('price:')[1].strip().replace('£', '').replace(',', '')
                            price = int(price_str)
                            if price <= 20000:
                                score += 4
                        except:
                            pass
            
            if score > 0:
                relevant.append((score, doc))
        
        relevant.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in relevant[:k]]
    
    def _fallback_response(self, user_input: str) -> Dict[str, Any]:
        """Fallback when Llama 3 is unavailable."""
        relevant_vehicles = self._get_relevant_vehicles(user_input, k=3)
        
        recommendations = []
        for i, doc in enumerate(relevant_vehicles, 1):
            lines = doc.page_content.split('\n')
            make = ""
            model = ""
            price = ""
            
            for line in lines:
                if line.startswith("Make:"):
                    make = line.replace("Make:", "").strip()
                elif line.startswith("Model:"):
                    model = line.replace("Model:", "").strip()
                elif line.startswith("Price:"):
                    price = line.replace("Price:", "").strip()
            
            vehicle_name = f"{make} {model}" if make and model else "Vehicle"
            recommendations.append(f"{i}. **{vehicle_name}** - {price}")
        
        return {
            "recommendations": "## 🚗 **Recommendations**\n" + "\n".join(recommendations),
            "next_question": "How many people will usually be in the car?",
            "question_type": "radio",
            "options": ["Up to 2 passengers", "3 to 5 passengers", "6 to 8 passengers"],
            "conversation_summary": "Using fallback mode",
            "skipped_questions": [],
            "llama3_managed": False
        } 