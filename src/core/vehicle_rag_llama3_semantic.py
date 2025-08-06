#!/usr/bin/env python3
"""
Enhanced Vehicle Recommendation System with Llama 3 Semantic Search and Reasoning Agents
Uses Llama 3 for both semantic understanding and intelligent reasoning
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

class Llama3SemanticRAG:
    def __init__(self, csv_path: str = "src/data/stocked_cars.csv"):
        """Initialize enhanced system with Llama 3 semantic search and reasoning."""
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
        
        # Initialize Llama 3 for semantic search and reasoning
        self._setup_llama3_semantic()
        
        # Load vehicle data
        self.load_vehicle_data()
    
    def _setup_llama3_semantic(self):
        """Setup Llama 3 for semantic search and reasoning."""
        try:
            from langchain_community.llms import Ollama
            self.llm = Ollama(model="llama3")
            self.semantic_ready = True
            print("✅ Llama 3 semantic search and reasoning ready")
        except Exception as e:
            print(f"⚠️ Llama 3 not available: {e}")
            self.semantic_ready = False
    
    def load_vehicle_data(self):
        """Load vehicle data for semantic processing."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file {self.csv_path} not found")
        
        df = pd.read_csv(self.csv_path)
        print(f"✅ Loaded {len(df)} vehicles for processing")
        
        # Create enhanced documents for semantic search
        self.documents = []
        self.vehicles_df = df
        
        for _, row in df.iterrows():
            # Create rich vehicle description for semantic search
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
        
        print(f"✅ Processed {len(self.documents)} vehicles with semantic search")
    
    def semantic_search_with_llama3(self, query: str, k: int = 10) -> List[Dict]:
        """Perform semantic search using Llama 3."""
        if not self.semantic_ready:
            return []
        
        try:
            # Create semantic search prompt
            search_prompt = f"""
            You are a vehicle recommendation expert. Given the user query: "{query}"
            
            I have a database of vehicles. For each vehicle, I need you to:
            1. Analyze how well it matches the user's requirements
            2. Provide a relevance score from 0-10 (10 being perfect match)
            3. Explain why it's relevant or not
            
            Here are the vehicles to analyze:
            """
            
            # Add vehicle descriptions to the prompt
            vehicle_descriptions = []
            for i, doc in enumerate(self.documents[:50]):  # Limit to first 50 for performance
                vehicle_descriptions.append(f"Vehicle {i+1}: {doc['content']}")
            
            search_prompt += "\n".join(vehicle_descriptions)
            search_prompt += f"""
            
            Please analyze each vehicle and return a JSON response with this format:
            {{
                "vehicle_analysis": [
                    {{
                        "vehicle_index": 0,
                        "relevance_score": 8.5,
                        "reasoning": "This BMW SUV matches the user's requirements for a luxury family vehicle",
                        "metadata": {{
                            "MAKE": "BMW",
                            "MODEL": "X3",
                            "BODY_TYPE": "SUV",
                            "FUEL_TYPE": "Petrol",
                            "PRICE": 25000
                        }}
                    }}
                ]
            }}
            
            Focus on semantic understanding - match intent, not just keywords.
            """
            
            # Get Llama 3 analysis
            response = self.llm.invoke(search_prompt)
            
            # Parse the response
            try:
                # Try to extract JSON from the response
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    analysis = json.loads(json_str)
                    
                    # Convert to our format
                    results = []
                    for item in analysis.get('vehicle_analysis', []):
                        vehicle_idx = item.get('vehicle_index', 0)
                        if vehicle_idx < len(self.documents):
                            results.append({
                                'document': self.documents[vehicle_idx],
                                'similarity': item.get('relevance_score', 0) / 10.0,
                                'metadata': self.documents[vehicle_idx]['metadata'],
                                'reasoning': item.get('reasoning', '')
                            })
                    
                    # Sort by relevance score
                    results.sort(key=lambda x: x['similarity'], reverse=True)
                    return results[:k]
                else:
                    # Fallback: use simple keyword matching
                    return self._fallback_semantic_search(query, k)
                    
            except json.JSONDecodeError:
                # Fallback: use simple keyword matching
                return self._fallback_semantic_search(query, k)
                
        except Exception as e:
            print(f"⚠️ Llama 3 semantic search failed: {e}")
            return self._fallback_semantic_search(query, k)
    
    def _fallback_semantic_search(self, query: str, k: int = 10) -> List[Dict]:
        """Fallback semantic search using simple keyword matching."""
        query_lower = query.lower()
        results = []
        
        for doc in self.documents:
            score = 0
            content_lower = doc['content'].lower()
            
            # Simple keyword matching
            keywords = query_lower.split()
            for keyword in keywords:
                if keyword in content_lower:
                    score += 1
            
            if score > 0:
                results.append({
                    'document': doc,
                    'similarity': min(score / len(keywords), 1.0),
                    'metadata': doc['metadata'],
                    'reasoning': f"Matched {score} keywords from query"
                })
        
        # Sort by score and return top k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:k]
    
    def reasoning_agent_analysis(self, user_input: str, vehicles: List[Dict]) -> str:
        """Use Llama 3 to analyze and reason about vehicle recommendations."""
        if not self.semantic_ready or not vehicles:
            return self._fallback_recommendation(vehicles)
        
        try:
            # Create context for reasoning
            vehicle_summary = "\n".join([
                f"- {v['metadata']['MAKE']} {v['metadata']['MODEL']} ({v['metadata']['BODY_TYPE']}, {v['metadata']['FUEL_TYPE']}, £{v['metadata']['PRICE']:,})"
                for v in vehicles[:5]  # Top 5 vehicles
            ])
            
            reasoning_prompt = f"""
            You are an expert vehicle consultant. Analyze these vehicle recommendations based on the user's request: "{user_input}"
            
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
            2. Any important considerations (safety, fuel efficiency, reliability, etc.)
            3. A recommendation summary with specific vehicle suggestions
            4. Any additional questions you might have for the user
            
            Format your response as a natural, helpful recommendation that a vehicle consultant would give to a customer.
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
    
    def process_with_llama3_semantic(self, user_input: str) -> Dict[str, Any]:
        """Process user input with Llama 3 semantic search and reasoning."""
        start_time = time.time()
        
        try:
            # Update conversation state
            self._update_conversation_state(user_input)
            
            # Perform semantic search with Llama 3
            semantic_results = self.semantic_search_with_llama3(user_input, k=10)
            
            # Use reasoning agent for analysis
            reasoning_analysis = self.reasoning_agent_analysis(user_input, semantic_results)
            
            # Determine next question
            next_question = self._determine_next_question(user_input)
            
            # Generate conversation summary
            conversation_summary = self._generate_conversation_summary()
            
            response_time = time.time() - start_time
            
            return {
                "recommendations": reasoning_analysis,
                "next_question": next_question,
                "conversation_summary": conversation_summary,
                "semantic_matches": len(semantic_results),
                "pattern_matches": 0,  # Not using pattern matching
                "combined_matches": len(semantic_results),
                "reasoning_used": self.semantic_ready,
                "semantic_used": self.semantic_ready,
                "response_time": response_time,
                "conversation_state": self.conversation_state.copy()
            }
            
        except Exception as e:
            print(f"⚠️ Llama 3 semantic RAG processing failed: {e}")
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

def test_llama3_semantic_rag():
    """Test the Llama 3 semantic RAG system."""
    print("🧪 Testing Llama 3 Semantic RAG System")
    print("=" * 50)
    
    rag = Llama3SemanticRAG("src/data/stocked_cars.csv")
    
    # Test scenarios
    test_queries = [
        "I want a BMW below £20000",
        "Show me electric vehicles for my family", 
        "i need a new diesel hatchback less than 17000"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing: {query}")
        result = rag.process_with_llama3_semantic(query)
        
        print(f"✅ Semantic matches: {result['semantic_matches']}")
        print(f"✅ Combined matches: {result['combined_matches']}")
        print(f"✅ Reasoning used: {result['reasoning_used']}")
        print(f"✅ Response time: {result['response_time']:.3f}s")
        print(f"✅ Summary: {result['conversation_summary']}")
        print(f"✅ Recommendations: {result['recommendations'][:200]}...")

if __name__ == "__main__":
    test_llama3_semantic_rag() 