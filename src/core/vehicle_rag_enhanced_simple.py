#!/usr/bin/env python3
"""
Enhanced Vehicle Recommendation System with Llama 3 Reasoning Agents
Simplified version without semantic search for initial testing
"""

import os
import pandas as pd
import time
import json
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class EnhancedVehicleRAGSimple:
    def __init__(self, csv_path: str = "src/data/stocked_cars.csv"):
        """Initialize enhanced system with Llama 3 reasoning."""
        self.csv_path = csv_path
        
        # Enhanced conversation state with reasoning context
        self.conversation_state = {
            "passenger_count": None,
            "body_type": None,
            "price_range": None,
            "make_pref": None,
            "fuel_pref": None,
            "answered_questions": set(),
            "conversation_history": [],
            "reasoning_context": {},
            "semantic_matches": []
        }
        
        # Body type to passenger category lookup
        self.body_type_passenger_category = {
            "SUV": "3-5 or 6-8",
            "MPV": "6-8",
            "Hatchback": "3-5",
            "Estate": "3-5",
            "Saloon": "3-5",
            "Coupe": "0-2 or 3-5",
            "Convertible": "0-2 or 3-5"
        }
        
        # Initialize Llama 3 reasoning
        self._setup_llama3_reasoning()
        
        # Load vehicle data
        self.load_vehicle_data()
    
    def _setup_llama3_reasoning(self):
        """Setup Llama 3 for reasoning agents."""
        try:
            from langchain_community.llms import Ollama
            self.llm = Ollama(model="llama3")
            self.reasoning_ready = True
            print("✅ Llama 3 reasoning ready")
        except Exception as e:
            print(f"⚠️ Llama 3 reasoning not available: {e}")
            self.reasoning_ready = False
    
    def load_vehicle_data(self):
        """Load vehicle data for processing."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file {self.csv_path} not found")
        
        df = pd.read_csv(self.csv_path)
        print(f"✅ Loaded {len(df)} vehicles for processing")
        
        # Create enhanced documents
        self.documents = []
        self.vehicles_df = df
        
        for _, row in df.iterrows():
            # Create rich vehicle description
            vehicle_text = f"""
            {row.get('MAKE', 'Unknown')} {row.get('MODEL', 'Unknown')} 
            {row.get('BODY_TYPE', 'Unknown')} {row.get('FUEL_TYPE', 'Unknown')} 
            £{row.get('PRICE', 0):,} {row.get('MILEAGE', 0):,} miles
            {row.get('TRANSMISSION_TYPE', 'Unknown')} transmission
            Color: {row.get('COLOUR', 'Unknown')}
            VRM: {row.get('VRM', 'Unknown')}
            """
            
            self.documents.append({
                'content': vehicle_text.strip(),
                'metadata': row.to_dict()
            })
        
        print(f"✅ Processed {len(self.documents)} vehicles")
    
    def reasoning_agent_analysis(self, user_input: str, vehicles: List[Dict]) -> str:
        """Use Llama 3 to analyze and reason about vehicle recommendations."""
        if not self.reasoning_ready or not vehicles:
            return self._fallback_recommendation(vehicles)
        
        try:
            # Create context for reasoning
            vehicle_summary = "\n".join([
                f"- {v['metadata']['MAKE']} {v['metadata']['MODEL']} ({v['metadata']['BODY_TYPE']}, {v['metadata']['FUEL_TYPE']}, £{v['metadata']['PRICE']:,})"
                for v in vehicles[:5]  # Top 5 vehicles
            ])
            
            reasoning_prompt = f"""
            Analyze these vehicle recommendations based on the user's request: "{user_input}"
            
            Available vehicles:
            {vehicle_summary}
            
            Current conversation state:
            - Passenger count: {self.conversation_state.get('passenger_count', 'Not specified')}
            - Body type: {self.conversation_state.get('body_type', 'Not specified')}
            - Price range: {self.conversation_state.get('price_range', 'Not specified')}
            - Make preference: {self.conversation_state.get('make_pref', 'Not specified')}
            - Fuel type: {self.conversation_state.get('fuel_pref', 'Not specified')}
            
            Please provide:
            1. A brief analysis of how well these vehicles match the user's requirements
            2. Any important considerations (safety, fuel efficiency, etc.)
            3. A recommendation summary
            
            Format your response as a natural, helpful recommendation.
            """
            
            response = self.llm.invoke(reasoning_prompt)
            return response.strip()
            
        except Exception as e:
            print(f"⚠️ Reasoning analysis failed: {e}")
            return self._fallback_recommendation(vehicles)
    
    def _fallback_recommendation(self, vehicles: List[Dict]) -> str:
        """Fallback recommendation when reasoning is not available."""
        if not vehicles:
            return "I couldn't find any vehicles matching your requirements. Please try adjusting your criteria."
        
        recommendations = []
        for vehicle in vehicles[:3]:
            v = vehicle['metadata']
            recommendations.append(
                f"• {v['MAKE']} {v['MODEL']} - {v['BODY_TYPE']} ({v['FUEL_TYPE']}) - £{v['PRICE']:,}"
            )
        
        return f"Here are some vehicles that might interest you:\n" + "\n".join(recommendations)
    
    def process_with_enhanced_rag(self, user_input: str) -> Dict[str, Any]:
        """Process user input with enhanced reasoning."""
        start_time = time.time()
        
        try:
            # Update conversation state
            self._update_conversation_state(user_input)
            
            # Perform pattern-based search
            pattern_results = self._get_relevant_vehicles_pattern(user_input, k=10)
            
            # Use reasoning agent for analysis
            reasoning_analysis = self.reasoning_agent_analysis(user_input, pattern_results)
            
            # Determine next question
            next_question = self._determine_next_question(user_input)
            
            # Generate conversation summary
            conversation_summary = self._generate_conversation_summary()
            
            response_time = time.time() - start_time
            
            return {
                "recommendations": reasoning_analysis,
                "next_question": next_question,
                "conversation_summary": conversation_summary,
                "semantic_matches": 0,  # Not available in simple version
                "pattern_matches": len(pattern_results),
                "combined_matches": len(pattern_results),
                "reasoning_used": self.reasoning_ready,
                "semantic_used": False,
                "response_time": response_time,
                "conversation_state": self.conversation_state.copy()
            }
            
        except Exception as e:
            print(f"⚠️ Enhanced RAG processing failed: {e}")
            return self._fallback_response(user_input)
    
    def _update_conversation_state(self, user_input: str):
        """Update conversation state with enhanced preference extraction."""
        user_input_lower = user_input.lower()
        
        # Extract preferences with enhanced logic
        self._extract_preferences_enhanced(user_input_lower)
        
        # Add to conversation history
        self.conversation_state["conversation_history"].append({
            "user_input": user_input,
            "timestamp": time.time()
        })
    
    def _extract_preferences_enhanced(self, user_input_lower: str):
        """Enhanced preference extraction with semantic understanding."""
        
        # Passenger count extraction
        if "passenger" in user_input_lower:
            if any(x in user_input_lower for x in ["6 to 8", "6-8", "six to eight", "6 or more"]):
                self.conversation_state["passenger_count"] = "6-8"
                self.conversation_state["answered_questions"].add("passenger_count")
            elif any(x in user_input_lower for x in ["3 to 5", "3-5", "three to five", "family"]):
                self.conversation_state["passenger_count"] = "3-5"
                self.conversation_state["answered_questions"].add("passenger_count")
            elif any(x in user_input_lower for x in ["up to 2", "0-2", "one or two", "couple"]):
                self.conversation_state["passenger_count"] = "0-2"
                self.conversation_state["answered_questions"].add("passenger_count")
        
        # Body type extraction
        body_types = ["suv", "hatchback", "estate", "saloon", "coupe", "convertible", "mpv"]
        for body_type in body_types:
            if body_type in user_input_lower:
                self.conversation_state["body_type"] = body_type.title()
                self.conversation_state["answered_questions"].add("body_type")
                break
        
        # Price range extraction
        price_patterns = [
            (r"below\s*£?(\d+)", lambda m: f"£0 - £{m.group(1)},000"),
            (r"under\s*£?(\d+)", lambda m: f"£0 - £{m.group(1)},000"),
            (r"less\s*than\s*£?(\d+)", lambda m: f"£0 - £{m.group(1)},000"),
            (r"between\s*£?(\d+)\s*and\s*£?(\d+)", lambda m: f"£{m.group(1)} - £{m.group(2)},000"),
            (r"£?(\d+)\s*to\s*£?(\d+)", lambda m: f"£{m.group(1)} - £{m.group(2)},000")
        ]
        
        for pattern, formatter in price_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                self.conversation_state["price_range"] = formatter(match)
                self.conversation_state["answered_questions"].add("price_range")
                break
        
        # Make preference extraction
        makes = ["bmw", "audi", "mercedes", "volkswagen", "ford", "toyota", "honda", "nissan", "kia", "hyundai", "volvo", "skoda", "seat", "peugeot", "renault", "citroen", "vauxhall", "mg", "tesla", "polestar", "land rover", "jaguar", "mini", "alfa romeo", "cupra", "ds", "fiat", "mazda", "mitsubishi", "suzuki", "dacia"]
        for make in makes:
            if make in user_input_lower:
                self.conversation_state["make_pref"] = make.upper()
                self.conversation_state["answered_questions"].add("make_pref")
                break
        
        # Fuel type extraction
        fuel_types = {
            "electric": "electric",
            "diesel": "diesel", 
            "petrol": "petrol",
            "hybrid": "hybrid",
            "plug-in": "plug-in hybrid"
        }
        
        for fuel_keyword, fuel_type in fuel_types.items():
            if fuel_keyword in user_input_lower:
                self.conversation_state["fuel_pref"] = fuel_type
                self.conversation_state["answered_questions"].add("fuel_pref")
                break
    
    def _get_relevant_vehicles_pattern(self, user_input: str, k: int = 5) -> List[Dict]:
        """Get relevant vehicles using pattern matching."""
        user_input_lower = user_input.lower()
        relevant_vehicles = []
        
        for _, row in self.vehicles_df.iterrows():
            score = 0
            vehicle_data = row.to_dict()
            
            # Check make preference
            if self.conversation_state.get("make_pref"):
                if self.conversation_state["make_pref"].lower() in str(vehicle_data.get("MAKE", "")).lower():
                    score += 10
            
            # Check fuel type preference
            if self.conversation_state.get("fuel_pref"):
                if self.conversation_state["fuel_pref"].lower() in str(vehicle_data.get("FUEL_TYPE", "")).lower():
                    score += 8
            
            # Check body type preference
            if self.conversation_state.get("body_type"):
                if self.conversation_state["body_type"].lower() in str(vehicle_data.get("BODY_TYPE", "")).lower():
                    score += 6
            
            # Check price range
            if self.conversation_state.get("price_range"):
                price_range = self.conversation_state["price_range"]
                vehicle_price = vehicle_data.get("PRICE", 0)
                
                if "£0 - £" in price_range:
                    max_price = int(re.search(r"£0 - £(\d+)", price_range).group(1)) * 1000
                    if vehicle_price <= max_price:
                        score += 5
                elif "£" in price_range and "to" in price_range:
                    min_price, max_price = map(int, re.findall(r"£(\d+)", price_range))
                    if min_price * 1000 <= vehicle_price <= max_price * 1000:
                        score += 5
            
            # Check passenger count compatibility
            if self.conversation_state.get("passenger_count"):
                passenger_count = self.conversation_state["passenger_count"]
                body_type = vehicle_data.get("BODY_TYPE", "").lower()
                
                if passenger_count == "6-8" and body_type in ["suv", "mpv"]:
                    score += 4
                elif passenger_count == "3-5" and body_type in ["hatchback", "estate", "saloon", "suv"]:
                    score += 4
                elif passenger_count == "0-2" and body_type in ["coupe", "convertible", "hatchback"]:
                    score += 4
            
            # Keyword matching
            keywords = user_input_lower.split()
            for keyword in keywords:
                if keyword in str(vehicle_data).lower():
                    score += 1
            
            if score > 0:
                relevant_vehicles.append({
                    'metadata': vehicle_data,
                    'score': score
                })
        
        # Sort by score and return top k
        relevant_vehicles.sort(key=lambda x: x['score'], reverse=True)
        return relevant_vehicles[:k]
    
    def _determine_next_question(self, user_input: str) -> Dict[str, Any]:
        """Determine the next question based on conversation state."""
        answered = self.conversation_state["answered_questions"]
        
        # Priority order for questions
        questions = [
            {"id": "passenger_count", "question": "How many people will usually be in the car?", "type": "radio", "options": ["Up to 2 passengers", "3 to 5 passengers", "6 to 8 passengers"]},
            {"id": "price_range", "question": "What's your budget range?", "type": "text", "options": []},
            {"id": "body_type", "question": "Which body types do you like?", "type": "multiselect", "options": ["Hatchback", "Estate", "SUV", "Coupe", "Convertible", "Saloon", "MPV"]},
            {"id": "make_pref", "question": "Any specific make preference?", "type": "text", "options": []}
        ]
        
        for question in questions:
            if question["id"] not in answered:
                return question
        
        # If all questions answered, return None
        return None
    
    def _generate_conversation_summary(self) -> str:
        """Generate a summary of the current conversation state."""
        state = self.conversation_state
        summary_parts = []
        
        if state.get("make_pref"):
            summary_parts.append(f"Make: {state['make_pref']}")
        if state.get("fuel_pref"):
            summary_parts.append(f"Fuel: {state['fuel_pref']}")
        if state.get("passenger_count"):
            summary_parts.append(f"Passengers: {state['passenger_count']}")
        if state.get("body_type"):
            summary_parts.append(f"Body: {state['body_type']}")
        if state.get("price_range"):
            summary_parts.append(f"Budget: {state['price_range']}")
        
        if summary_parts:
            return " | ".join(summary_parts)
        else:
            return "No preferences set yet"
    
    def _get_skipped_questions(self) -> List[str]:
        """Get list of questions that can be skipped."""
        return list(self.conversation_state["answered_questions"])
    
    def _fallback_response(self, user_input: str) -> Dict[str, Any]:
        """Fallback response when processing fails."""
        return {
            "recommendations": "I'm having trouble processing your request. Please try again with different criteria.",
            "next_question": None,
            "conversation_summary": "Error occurred",
            "semantic_matches": 0,
            "pattern_matches": 0,
            "combined_matches": 0,
            "reasoning_used": False,
            "semantic_used": False,
            "response_time": 0,
            "conversation_state": self.conversation_state.copy()
        }

def test_enhanced_rag_simple():
    """Test the simplified enhanced RAG system."""
    print("🧪 Testing Simplified Enhanced RAG System")
    print("=" * 50)
    
    rag = EnhancedVehicleRAGSimple("src/data/stocked_cars.csv")
    
    # Test scenarios
    test_queries = [
        "I want a BMW below £20000",
        "Show me electric vehicles for my family", 
        "i need a new diesel hatchback less than 17000"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing: {query}")
        result = rag.process_with_enhanced_rag(query)
        
        print(f"✅ Pattern matches: {result['pattern_matches']}")
        print(f"✅ Combined matches: {result['combined_matches']}")
        print(f"✅ Reasoning used: {result['reasoning_used']}")
        print(f"✅ Response time: {result['response_time']:.3f}s")
        print(f"✅ Summary: {result['conversation_summary']}")
        print(f"✅ Recommendations: {result['recommendations'][:200]}...")

if __name__ == "__main__":
    test_enhanced_rag_simple() 