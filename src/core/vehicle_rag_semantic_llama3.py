#!/usr/bin/env python3
"""
Enhanced Vehicle Recommendation System with Llama3 Semantic Search and Reasoning Agents
Combines fast pattern matching with intelligent semantic understanding and reasoning
"""

import os
import pandas as pd
import time
import json
import re
import numpy as np
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

class EnhancedVehicleRAG:
    def __init__(self, csv_path: str = "src/data/stocked_cars.csv"):
        """Initialize enhanced system with Llama3 semantic search and reasoning."""
        self.csv_path = csv_path
        
        # Conversation state with reliable memory
        self.conversation_state = {
            "passenger_count": None,
            "body_type": None,
            "price_range": None,
            "make_pref": None,
            "fuel_pref": None,
            "answered_questions": set(),
            "conversation_history": []
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
        
        # Initialize semantic search
        self._setup_semantic_search()
        
        # Initialize Llama3 reasoning
        self._setup_llama3_reasoning()
        
        # Load vehicle data
        self.load_vehicle_data()
    
    def _setup_semantic_search(self):
        """Setup semantic search with sentence transformers."""
        try:
            print("🔄 Loading semantic search model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.semantic_search_ready = True
            print("✅ Semantic search ready")
        except Exception as e:
            print(f"⚠️ Semantic search not available: {e}")
            self.semantic_search_ready = False
    
    def _setup_llama3_reasoning(self):
        """Setup Llama3 for reasoning agents."""
        try:
            from langchain_community.llms import Ollama
            self.llm = Ollama(model="llama3")
            self.reasoning_ready = True
            print("✅ Llama3 reasoning ready")
        except Exception as e:
            print(f"⚠️ Llama3 reasoning not available: {e}")
            self.reasoning_ready = False
    
    def load_vehicle_data(self):
        """Load vehicle data and prepare for semantic search."""
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
        
        # Create embeddings for semantic search
        if self.semantic_search_ready:
            self._create_embeddings()
        
        print(f"✅ Processed {len(self.documents)} vehicles")
    
    def _create_embeddings(self):
        """Create embeddings for semantic search."""
        print("🔄 Creating semantic embeddings...")
        
        # Create rich descriptions for embedding
        vehicle_descriptions = []
        for doc in self.documents:
            metadata = doc['metadata']
            description = f"""
            {metadata.get('MAKE', '')} {metadata.get('MODEL', '')} 
            {metadata.get('BODY_TYPE', '')} {metadata.get('FUEL_TYPE', '')} 
            {metadata.get('PRICE', 0)} {metadata.get('MILEAGE', 0)} miles
            {metadata.get('TRANSMISSION_TYPE', '')} transmission
            """
            vehicle_descriptions.append(description.strip())
        
        # Create embeddings
        self.embeddings = self.embedding_model.encode(vehicle_descriptions)
        print(f"✅ Created embeddings for {len(self.embeddings)} vehicles")
    
    def semantic_search(self, query: str, k: int = 5) -> List[Dict]:
        """Perform semantic search for vehicles."""
        if not self.semantic_search_ready:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top k matches
        top_indices = similarities.argsort()[-k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.3:  # Minimum similarity threshold
                results.append({
                    'vehicle': self.documents[idx]['metadata'],
                    'similarity': similarities[idx],
                    'description': self.documents[idx]['content']
                })
        
        return results
    
    def reasoning_agent_analysis(self, user_input: str, vehicles: List[Dict]) -> str:
        """Use Llama3 for reasoning and analysis."""
        if not self.reasoning_ready or not vehicles:
            return self._fallback_recommendation(vehicles)
        
        try:
            # Create reasoning prompt
            vehicle_info = []
            for i, vehicle in enumerate(vehicles[:3], 1):
                v = vehicle['vehicle']
                vehicle_info.append(f"""
                {i}. {v.get('MAKE', 'Unknown')} {v.get('MODEL', 'Unknown')}
                   - Price: £{v.get('PRICE', 0):,}
                   - Body: {v.get('BODY_TYPE', 'Unknown')}
                   - Fuel: {v.get('FUEL_TYPE', 'Unknown')}
                   - Mileage: {v.get('MILEAGE', 0):,} miles
                """)
            
            prompt = f"""
            User Query: {user_input}
            
            Available Vehicles:
            {''.join(vehicle_info)}
            
            Current Preferences:
            - Passenger Count: {self.conversation_state.get('passenger_count', 'Not specified')}
            - Price Range: {self.conversation_state.get('price_range', 'Not specified')}
            - Body Type: {self.conversation_state.get('body_type', 'Not specified')}
            - Make Preference: {self.conversation_state.get('make_pref', 'Not specified')}
            - Fuel Preference: {self.conversation_state.get('fuel_pref', 'Not specified')}
            
            Please analyze these vehicles and provide:
            1. Which vehicle best matches the user's needs and why
            2. Key trade-offs between the options
            3. A clear recommendation with reasoning
            
            Format your response as a clear recommendation with reasoning.
            """
            
            # Get Llama3 reasoning
            response = self.llm.invoke(prompt)
            return response
            
        except Exception as e:
            print(f"⚠️ Llama3 reasoning failed: {e}")
            return self._fallback_recommendation(vehicles)
    
    def _fallback_recommendation(self, vehicles: List[Dict]) -> str:
        """Fallback recommendation when reasoning is not available."""
        if not vehicles:
            return "## 🚗 **Recommendations**\nNo vehicles found matching your criteria. Please try different preferences."
        
        recommendation = "## 🚗 **Recommendations**\n\n"
        
        for i, vehicle_data in enumerate(vehicles[:3], 1):
            vehicle = vehicle_data['vehicle']
            recommendation += f"""
**{i}. {vehicle.get('MAKE', 'Unknown')} {vehicle.get('MODEL', 'Unknown')}**
   - **Price:** £{vehicle.get('PRICE', 0):,}
   - **Mileage:** {vehicle.get('MILEAGE', 0):,} miles
   - **Details:** Body: {vehicle.get('BODY_TYPE', 'Unknown')} | Fuel: {vehicle.get('FUEL_TYPE', 'Unknown')} | VRM: {vehicle.get('VRM', 'Unknown')}
   - **Why recommended:** Good match for your requirements
"""
        
        return recommendation
    
    def process_with_enhanced_rag(self, user_input: str) -> Dict[str, Any]:
        """Process user input with enhanced semantic search and reasoning."""
        start_time = time.time()
        
        try:
            # Update conversation state with memory
            self._update_conversation_state(user_input)
            
            # Get relevant vehicles using both pattern matching and semantic search
            pattern_vehicles = self._get_relevant_vehicles_pattern(user_input, k=3)
            semantic_vehicles = self.semantic_search(user_input, k=3)
            
            # Combine and deduplicate results
            all_vehicles = self._combine_vehicle_results(pattern_vehicles, semantic_vehicles)
            
            # Use reasoning agent for analysis
            recommendations = self.reasoning_agent_analysis(user_input, all_vehicles)
            
            # Determine next question
            next_question = self._determine_next_question(user_input)
            
            # Generate conversation summary
            conversation_summary = self._generate_conversation_summary()
            
            # Get skipped questions
            skipped_questions = self._get_skipped_questions()
            
            response_time = time.time() - start_time
            
            return {
                "recommendations": recommendations,
                "next_question": next_question,
                "conversation_summary": conversation_summary,
                "skipped_questions": skipped_questions,
                "response_time": response_time,
                "enhanced_rag": True,
                "semantic_search_used": len(semantic_vehicles) > 0,
                "reasoning_used": self.reasoning_ready
            }
            
        except Exception as e:
            print(f"❌ Error in enhanced processing: {e}")
            return self._fallback_response(user_input)
    
    def _update_conversation_state(self, user_input: str):
        """Update conversation state with enhanced pattern recognition."""
        user_input_lower = user_input.lower()
        
        # Add to conversation history
        self.conversation_state["conversation_history"].append({
            "input": user_input,
            "timestamp": time.time()
        })
        
        # Enhanced pattern matching with semantic understanding
        self._extract_preferences_enhanced(user_input_lower)
    
    def _extract_preferences_enhanced(self, user_input_lower: str):
        """Enhanced preference extraction with semantic understanding."""
        
        # Price range extraction (enhanced)
        if re.search(r'below\s*£?(\d+(?:,\d+)*)', user_input_lower):
            below_match = re.search(r'below\s*£?(\d+(?:,\d+)*)', user_input_lower)
            price_limit = int(below_match.group(1).replace(',', ''))
            self.conversation_state["price_range"] = f"£0 - £{price_limit:,}"
            self.conversation_state["answered_questions"].add("price_range")
        elif re.search(r'between\s*(\d+(?:k)?)\s*and\s*(\d+(?:k)?)', user_input_lower):
            range_match = re.search(r'between\s*(\d+(?:k)?)\s*and\s*(\d+(?:k)?)', user_input_lower)
            min_price_str = range_match.group(1)
            max_price_str = range_match.group(2)
            min_price = int(min_price_str.replace('k', '')) * (1000 if 'k' in min_price_str else 1)
            max_price = int(max_price_str.replace('k', '')) * (1000 if 'k' in max_price_str else 1)
            self.conversation_state["price_range"] = f"£{min_price:,} - £{max_price:,}"
            self.conversation_state["answered_questions"].add("price_range")
        elif re.search(r'less\s*than\s*(\d+(?:,\d+)*)', user_input_lower):
            less_than_match = re.search(r'less\s*than\s*(\d+(?:,\d+)*)', user_input_lower)
            price_limit = int(less_than_match.group(1).replace(',', ''))
            self.conversation_state["price_range"] = f"£0 - £{price_limit:,}"
            self.conversation_state["answered_questions"].add("price_range")
        
        # Enhanced passenger count recognition
        if any(word in user_input_lower for word in ['6 people', '7 people', '8 people', 'six people', 'seven people', 'eight people', '6 to 8', '6-8', '6 to 8 passengers', '6-8 passengers']):
            self.conversation_state["passenger_count"] = "6-8"
            self.conversation_state["answered_questions"].add("passenger_count")
        elif any(word in user_input_lower for word in ['3 people', '4 people', '5 people', 'three people', 'four people', 'five people', '3 to 5', '3-5', '3 to 5 passengers', '3-5 passengers']):
            self.conversation_state["passenger_count"] = "3-5"
            self.conversation_state["answered_questions"].add("passenger_count")
        elif any(word in user_input_lower for word in ['1 person', '2 people', 'one person', 'two people', '1 to 2', '1-2', '1 to 2 passengers', '1-2 passengers']):
            self.conversation_state["passenger_count"] = "0-2"
            self.conversation_state["answered_questions"].add("passenger_count")
        
        # Enhanced body type recognition
        body_types = ['suv', 'hatchback', 'estate', 'saloon', 'coupe', 'convertible', 'mpv']
        for body_type in body_types:
            if body_type in user_input_lower:
                self.conversation_state["body_type"] = body_type.title()
                self.conversation_state["answered_questions"].add("body_type")
                break
        
        # Enhanced make preference recognition
        makes = ['bmw', 'audi', 'mercedes', 'volkswagen', 'ford', 'volvo', 'hyundai', 'kia', 'mg', 'tesla']
        for make in makes:
            if make in user_input_lower:
                self.conversation_state["make_pref"] = make.upper()
                self.conversation_state["answered_questions"].add("make_pref")
                break
        
        # Enhanced fuel type recognition
        if any(word in user_input_lower for word in ['electric', 'ev', 'electric vehicle']):
            self.conversation_state["fuel_pref"] = "electric"
            self.conversation_state["answered_questions"].add("fuel_pref")
        elif 'hybrid' in user_input_lower:
            self.conversation_state["fuel_pref"] = "hybrid"
            self.conversation_state["answered_questions"].add("fuel_pref")
        elif 'diesel' in user_input_lower:
            self.conversation_state["fuel_pref"] = "diesel"
            self.conversation_state["answered_questions"].add("fuel_pref")
        elif 'petrol' in user_input_lower:
            self.conversation_state["fuel_pref"] = "petrol"
            self.conversation_state["answered_questions"].add("fuel_pref")
    
    def _get_relevant_vehicles_pattern(self, user_input: str, k: int = 5) -> List[Dict]:
        """Get relevant vehicles using pattern-based filtering."""
        relevant_vehicles = []
        
        for _, vehicle in self.vehicles_df.iterrows():
            score = 0
            price = vehicle.get('PRICE', 0)
            fuel_type = vehicle.get('FUEL_TYPE', '').lower()
            make = vehicle.get('MAKE', '').lower()
            body_type = vehicle.get('BODY_TYPE', '').lower()
            
            # Make preference scoring
            if self.conversation_state.get("make_pref"):
                if self.conversation_state["make_pref"].lower() in make:
                    score += 20
            
            # Body type scoring
            if self.conversation_state.get("body_type"):
                if self.conversation_state["body_type"].lower() in body_type:
                    score += 15
            
            # Price range filtering
            if self.conversation_state.get("price_range"):
                price_range = self.conversation_state["price_range"]
                range_match = re.search(r'£(\d+(?:,\d+)*)\s*-\s*£(\d+(?:,\d+)*)', price_range)
                if range_match:
                    min_price = int(range_match.group(1).replace(',', ''))
                    max_price = int(range_match.group(2).replace(',', ''))
                    if min_price <= price <= max_price:
                        score += 15
            
            # Fuel type scoring
            if self.conversation_state.get("fuel_pref"):
                preferred_fuel = self.conversation_state["fuel_pref"].lower()
                if preferred_fuel == "electric" and 'electric' in fuel_type and 'hybrid' not in fuel_type:
                    score += 25
                elif preferred_fuel == "diesel" and 'diesel' in fuel_type:
                    score += 20
                elif preferred_fuel == "petrol" and 'petrol' in fuel_type:
                    score += 20
                elif preferred_fuel == "hybrid" and 'hybrid' in fuel_type:
                    score += 20
            
            if score > 0:
                relevant_vehicles.append({
                    'vehicle': vehicle.to_dict(),
                    'score': score,
                    'description': f"{vehicle.get('MAKE', 'Unknown')} {vehicle.get('MODEL', 'Unknown')}"
                })
        
        return sorted(relevant_vehicles, key=lambda x: x['score'], reverse=True)[:k]
    
    def _combine_vehicle_results(self, pattern_vehicles: List[Dict], semantic_vehicles: List[Dict]) -> List[Dict]:
        """Combine pattern and semantic search results."""
        combined = []
        
        # Add pattern results
        for vehicle_data in pattern_vehicles:
            combined.append({
                'vehicle': vehicle_data['vehicle'],
                'score': vehicle_data['score'],
                'source': 'pattern'
            })
        
        # Add semantic results
        for vehicle_data in semantic_vehicles:
            combined.append({
                'vehicle': vehicle_data['vehicle'],
                'score': vehicle_data['similarity'] * 100,  # Convert similarity to score
                'source': 'semantic'
            })
        
        # Remove duplicates and sort by score
        seen_vrms = set()
        unique_vehicles = []
        
        for vehicle_data in combined:
            vrm = vehicle_data['vehicle'].get('VRM', '')
            if vrm not in seen_vrms:
                seen_vrms.add(vrm)
                unique_vehicles.append(vehicle_data)
        
        return sorted(unique_vehicles, key=lambda x: x['score'], reverse=True)[:5]
    
    def _determine_next_question(self, user_input: str) -> Dict[str, Any]:
        """Determine the next question based on conversation state."""
        answered = self.conversation_state["answered_questions"]
        
        if "passenger_count" not in answered:
            return {
                "question": "How many passengers do you need to accommodate?",
                "type": "radio",
                "options": ["Up to 2 passengers", "3 to 5 passengers", "6 to 8 passengers"]
            }
        elif "price_range" not in answered:
            return {
                "question": "What's your budget range?",
                "type": "text",
                "placeholder": "e.g., below £20,000 or between £10k and £30k"
            }
        elif "body_type" not in answered:
            return {
                "question": "What body type do you prefer?",
                "type": "multiselect",
                "options": ["SUV", "Hatchback", "Estate", "Saloon", "Coupe", "Convertible"]
            }
        else:
            return None
    
    def _generate_conversation_summary(self) -> str:
        """Generate a summary of the current conversation state."""
        summary_parts = []
        
        if self.conversation_state.get("passenger_count"):
            summary_parts.append(f"Passenger count: {self.conversation_state['passenger_count']}")
        
        if self.conversation_state.get("body_type"):
            summary_parts.append(f"Body type: {self.conversation_state['body_type']}")
        
        if self.conversation_state.get("price_range"):
            summary_parts.append(f"Budget: {self.conversation_state['price_range']}")
        
        if self.conversation_state.get("make_pref"):
            summary_parts.append(f"Make preference: {self.conversation_state['make_pref']}")
        
        if self.conversation_state.get("fuel_pref"):
            summary_parts.append(f"Fuel preference: {self.conversation_state['fuel_pref']}")
        
        return " | ".join(summary_parts) if summary_parts else "No preferences set yet"
    
    def _get_skipped_questions(self) -> List[str]:
        """Get list of questions that were skipped."""
        answered = self.conversation_state["answered_questions"]
        all_questions = {"passenger_count", "price_range", "body_type", "make_pref", "fuel_pref"}
        return list(all_questions - answered)
    
    def _fallback_response(self, user_input: str) -> Dict[str, Any]:
        """Fallback response when processing fails."""
        return {
            "recommendations": "## 🚗 **Recommendations**\nSorry, I encountered an error. Please try again.",
            "next_question": None,
            "conversation_summary": "Error occurred",
            "skipped_questions": [],
            "response_time": 0,
            "enhanced_rag": False,
            "semantic_search_used": False,
            "reasoning_used": False
        }

# Alias for compatibility
EnhancedVehicleRAG = EnhancedVehicleRAG 