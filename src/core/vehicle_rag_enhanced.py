#!/usr/bin/env python3
"""
Enhanced Vehicle Recommendation System with Llama 3 Semantic Search and Reasoning Agents
Uses Llama 3 for both semantic understanding and intelligent reasoning (no pattern search)
"""

import os
import pandas as pd
import time
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Llama3SemanticRAG:
    def __init__(self, csv_path: str = "src/data/stocked_cars.csv"):
        self.csv_path = csv_path
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
        self._setup_llama3()
        self.load_vehicle_data()

    def _setup_llama3(self):
        try:
            from langchain_community.llms import Ollama
            self.llm = Ollama(model="llama3")
            self.semantic_ready = True
            print("✅ Llama 3 semantic search and reasoning ready")
        except Exception as e:
            print(f"⚠️ Llama 3 not available: {e}")
            self.semantic_ready = False

    def load_vehicle_data(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file {self.csv_path} not found")
        df = pd.read_csv(self.csv_path)
        print(f"✅ Loaded {len(df)} vehicles for processing")
        self.documents = []
        self.vehicles_df = df
        for _, row in df.iterrows():
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
        if not self.semantic_ready:
            return []
        try:
            # Create a comprehensive semantic search prompt with conversation context
            search_prompt = f"""
            You are a vehicle recommendation expert. Given the user query: "{query}"
            
            Current conversation context:
            - Passengers: {self.conversation_state.get('passenger_count', 'Not specified')}
            - Body type: {self.conversation_state.get('body_type', 'Not specified')}
            - Budget: {self.conversation_state.get('price_range', 'Not specified')}
            - Make: {self.conversation_state.get('make_pref', 'Not specified')}
            - Fuel: {self.conversation_state.get('fuel_pref', 'Not specified')}
            
            VEHICLE DOMAIN KNOWLEDGE:
            - SUVs accommodate 6-8 passengers (family vehicles, spacious)
            - MPVs accommodate 6-8 passengers (multi-purpose vehicles)
            - Hatchbacks accommodate 3-5 passengers (compact family cars)
            - Saloons accommodate 3-5 passengers (sedan style)
            - Estates accommodate 3-5 passengers (wagon style)
            - Coupes accommodate 2-4 passengers (sporty, limited space)
            - Convertibles accommodate 2-4 passengers (open-top, limited space)
            
            SEMANTIC UNDERSTANDING RULES:
            - If user mentions "SUV", they need 6-8 passenger capacity
            - If user mentions "family car", they need 3-5 passenger capacity
            - If user mentions "sports car", they need 2-4 passenger capacity
            - If user mentions "6-8 passengers", only SUVs and MPVs are suitable
            - If user mentions "3-5 passengers", hatchbacks, saloons, estates, and some SUVs are suitable
            - If user mentions "2-4 passengers", coupes, convertibles, and some hatchbacks are suitable
            
            I need you to analyze all vehicles in this database and find the most relevant matches.
            For each vehicle, consider:
            1. Semantic understanding of user intent (not just keywords)
            2. Context and implied requirements from conversation history
            3. Vehicle type and passenger capacity compatibility
            4. Relevance score from 0-10 (10 being perfect match)
            5. Detailed reasoning for the score
            
            Important: Consider ALL conversation context when scoring vehicles.
            If passenger count is specified, only include vehicles that can accommodate that many people.
            If body type is specified, only include vehicles of that body type.
            If price range is specified, only include vehicles within that budget.
            If make preference is specified, prioritize vehicles of that make.
            If fuel type is specified, only include vehicles of that fuel type.
            
            Here are all vehicles to analyze:
            """
            
            # Include all vehicles for full semantic analysis
            vehicle_descriptions = []
            for i, doc in enumerate(self.documents):
                vehicle_descriptions.append(f"Vehicle {i+1}: {doc['content']}")
            
            search_prompt += "\n".join(vehicle_descriptions)
            search_prompt += f"""
            
            Return ONLY a JSON response with this exact format:
            {{
                "vehicle_analysis": [
                    {{
                        "vehicle_index": 0,
                        "relevance_score": 8.5,
                        "reasoning": "BMW SUV matches luxury and family requirements with good safety features"
                    }}
                ]
            }}
            
            Focus on semantic understanding - match intent, context, and implied needs.
            Only include vehicles with relevance_score >= 5.
            Consider factors like: family needs, budget constraints, lifestyle, safety, efficiency.
            IMPORTANT: Apply ALL conversation context filters strictly.
            IMPORTANT: Use vehicle domain knowledge to understand passenger capacity requirements.
            """
            
            response = self.llm.invoke(search_prompt)
            
            try:
                # Try to extract JSON from the response
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    analysis = json.loads(json_str)
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
                    print("⚠️ Llama 3 semantic search returned invalid JSON format")
                    return []
                    
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON parsing error in semantic search: {e}")
                return []
                
        except Exception as e:
            print(f"⚠️ Llama 3 semantic search failed: {e}")
            return []

    def extract_preferences_with_llama3(self, user_input: str) -> Dict[str, Any]:
        """Use Llama 3 to extract user preferences semantically."""
        if not self.semantic_ready:
            return {}
        
        try:
            extraction_prompt = f"""
            You are a vehicle preference extraction expert. Analyze this user input: "{user_input}"
            
            Current conversation context:
            - Passengers: {self.conversation_state.get('passenger_count', 'Not specified')}
            - Body type: {self.conversation_state.get('body_type', 'Not specified')}
            - Budget: {self.conversation_state.get('price_range', 'Not specified')}
            - Make: {self.conversation_state.get('make_pref', 'Not specified')}
            - Fuel: {self.conversation_state.get('fuel_pref', 'Not specified')}
            
            VEHICLE DOMAIN KNOWLEDGE:
            - SUVs accommodate 6-8 passengers (family vehicles, spacious)
            - MPVs accommodate 6-8 passengers (multi-purpose vehicles)
            - Hatchbacks accommodate 3-5 passengers (compact family cars)
            - Saloons accommodate 3-5 passengers (sedan style)
            - Estates accommodate 3-5 passengers (wagon style)
            - Coupes accommodate 2-4 passengers (sporty, limited space)
            - Convertibles accommodate 2-4 passengers (open-top, limited space)
            
            SEMANTIC UNDERSTANDING RULES:
            - If user mentions "SUV", they need 6-8 passenger capacity
            - If user mentions "family car", they need 3-5 passenger capacity
            - If user mentions "sports car", they need 2-4 passenger capacity
            - If user mentions "6-8 passengers", only SUVs and MPVs are suitable
            - If user mentions "3-5 passengers", hatchbacks, saloons, estates, and some SUVs are suitable
            - If user mentions "2-4 passengers", coupes, convertibles, and some hatchbacks are suitable
            
            Extract the following preferences and return ONLY a JSON response:
            1. passenger_count: "0-2", "3-5", "6-8", or null (only update if new info provided)
            2. body_type: "SUV", "Hatchback", "Estate", "Saloon", "Coupe", "Convertible", "MPV", or null (only update if new info provided)
            3. price_range: budget range in format "£X - £Y" or null (only update if new info provided)
            4. make_pref: car make preference or null (only update if new info provided)
            5. fuel_pref: "electric", "diesel", "petrol", "hybrid", "plug-in hybrid", or null (only update if new info provided)
            
            Consider semantic understanding, not just keywords. For example:
            - "family car" implies 3-5 passengers
            - "luxury" might imply higher budget
            - "eco-friendly" might imply electric/hybrid
            - "6 to 8 passenger" means 6-8 passengers AND automatically implies SUV body type
            - "3 to 5 passenger" means 3-5 passengers
            - "0-2 passengers" or "couple" implies 0-2 passengers
            - "SUV" automatically implies 6-8 passenger capacity
            - "hatchback" or "saloon" implies 3-5 passenger capacity
            - "coupe" or "convertible" implies 2-4 passenger capacity
            
            IMPORTANT LOGIC:
            - If passenger count is "6-8", automatically set body_type to "SUV" (only SUVs can accommodate 6-8 passengers)
            - If body_type is "SUV", automatically set passenger_count to "6-8" (SUVs accommodate 6-8 passengers)
            - If passenger count is "0-2", body type can be any (coupe, convertible, hatchback, etc.)
            - If passenger count is "3-5", body type can be any (hatchback, estate, saloon, SUV, etc.)
            
            IMPORTANT: Only update preferences that are explicitly mentioned in the new input.
            If the input doesn't provide new information for a preference, keep the existing value.
            
            Return format:
            {{
                "passenger_count": "3-5",
                "body_type": "SUV",
                "price_range": "£15000 - £25000",
                "make_pref": "BMW",
                "fuel_pref": "diesel"
            }}
            """
            
            response = self.llm.invoke(extraction_prompt)
            
            try:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    preferences = json.loads(json_str)
                    
                    # Apply automatic inference logic
                    if preferences.get('passenger_count') == '6-8':
                        preferences['body_type'] = 'SUV'
                    elif preferences.get('body_type') == 'SUV':
                        preferences['passenger_count'] = '6-8'
                    elif preferences.get('passenger_count') == '0-2':
                        # For 0-2 passengers, don't automatically set body type
                        # Let user specify or system ask
                        pass
                    elif preferences.get('passenger_count') == '3-5':
                        # For 3-5 passengers, don't automatically set body type
                        # Let user specify or system ask
                        pass
                    
                    return preferences
                else:
                    print("⚠️ Llama 3 preference extraction returned invalid JSON")
                    return {}
                    
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON parsing error in preference extraction: {e}")
                return {}
                
        except Exception as e:
            print(f"⚠️ Llama 3 preference extraction failed: {e}")
            return {}

    def reasoning_agent_analysis(self, user_input: str, vehicles: List[Dict]) -> str:
        if not self.semantic_ready or not vehicles:
            return "I couldn't find any vehicles matching your requirements. Please try adjusting your criteria."
        
        try:
            # Create a comprehensive reasoning prompt with full context
            vehicle_summary = "\n".join([
                f"- {v['metadata']['MAKE']} {v['metadata']['MODEL']} ({v['metadata']['BODY_TYPE']}, {v['metadata']['FUEL_TYPE']}, £{v['metadata']['PRICE']:,}) - {v['reasoning']}"
                for v in vehicles[:5]  # Include top 5 for comprehensive analysis
            ])
            
            reasoning_prompt = f"""
            You are an expert vehicle consultant. The user asked: "{user_input}"
            
            Current conversation context:
            - Passengers: {self.conversation_state.get('passenger_count', 'Not specified')}
            - Body type: {self.conversation_state.get('body_type', 'Not specified')}
            - Budget: {self.conversation_state.get('price_range', 'Not specified')}
            - Make: {self.conversation_state.get('make_pref', 'Not specified')}
            - Fuel: {self.conversation_state.get('fuel_pref', 'Not specified')}
            
            VEHICLE DOMAIN KNOWLEDGE:
            - SUVs accommodate 6-8 passengers (family vehicles, spacious)
            - MPVs accommodate 6-8 passengers (multi-purpose vehicles)
            - Hatchbacks accommodate 3-5 passengers (compact family cars)
            - Saloons accommodate 3-5 passengers (sedan style)
            - Estates accommodate 3-5 passengers (wagon style)
            - Coupes accommodate 2-4 passengers (sporty, limited space)
            - Convertibles accommodate 2-4 passengers (open-top, limited space)
            
            SEMANTIC UNDERSTANDING RULES:
            - If user mentions "SUV", they need 6-8 passenger capacity
            - If user mentions "family car", they need 3-5 passenger capacity
            - If user mentions "sports car", they need 2-4 passenger capacity
            - If user mentions "6-8 passengers", only SUVs and MPVs are suitable
            - If user mentions "3-5 passengers", hatchbacks, saloons, estates, and some SUVs are suitable
            - If user mentions "2-4 passengers", coupes, convertibles, and some hatchbacks are suitable
            
            Top matching vehicles with reasoning:
            {vehicle_summary}
            
            Provide a comprehensive, helpful recommendation (3-4 sentences) focusing on:
            1. How well these vehicles match the user's needs and preferences
            2. Key benefits, features, and considerations for each option
            3. Any potential drawbacks or trade-offs
            4. Additional questions to help narrow down choices
            5. Lifestyle and practical considerations
            
            IMPORTANT: Reference the conversation context and explain how the vehicles match the accumulated preferences.
            IMPORTANT: Use vehicle domain knowledge to understand passenger capacity requirements.
            Keep it conversational, informative, and actionable.
            """
            
            response = self.llm.invoke(reasoning_prompt)
            return response.strip()
            
        except Exception as e:
            print(f"⚠️ Reasoning analysis failed: {e}")
            return "I'm having trouble analyzing the vehicles. Please try again."

    def determine_next_question_with_llama3(self, user_input: str) -> Dict[str, Any]:
        """Use Llama 3 to determine the next most relevant question."""
        if not self.semantic_ready:
            return None
        
        try:
            answered = self.conversation_state["answered_questions"]
            available_questions = [
                {"id": "passenger_count", "question": "How many people will usually be in the car?", "type": "radio", "options": ["Up to 2 passengers", "3 to 5 passengers", "6 to 8 passengers"]},
                {"id": "price_range", "question": "What's your budget range?", "type": "text", "options": []},
                {"id": "body_type", "question": "Which body types do you prefer?", "type": "multiselect", "options": ["Hatchback", "Estate", "SUV", "Coupe", "Convertible", "Saloon", "MPV"]},
                {"id": "make_pref", "question": "Any specific make preference?", "type": "text", "options": []},
                {"id": "fuel_pref", "question": "What fuel type do you prefer?", "type": "radio", "options": ["Petrol", "Diesel", "Electric", "Hybrid", "Plug-in Hybrid"]}
            ]
            
            # Filter out questions that are automatically answered
            filtered_questions = []
            for question in available_questions:
                if question["id"] not in answered:
                    # Skip body_type question if passenger_count is 6-8 (automatically SUV)
                    if question["id"] == "body_type" and self.conversation_state.get("passenger_count") == "6-8":
                        continue
                    # Skip passenger_count question if body_type is SUV (automatically 6-8 passengers)
                    elif question["id"] == "passenger_count" and self.conversation_state.get("body_type") == "SUV":
                        continue
                    # Skip body_type question if passenger_count is 0-2 and user hasn't specified
                    elif question["id"] == "body_type" and self.conversation_state.get("passenger_count") == "0-2":
                        continue
                    filtered_questions.append(question)
            
            if not filtered_questions:
                return None
            
            question_prompt = f"""
            You are a vehicle consultant determining the next most important question to ask.
            
            User's latest input: "{user_input}"
            Current conversation state: {self.conversation_state}
            Already answered questions: {list(answered)}
            
            VEHICLE DOMAIN KNOWLEDGE:
            - SUVs accommodate 6-8 passengers (family vehicles, spacious)
            - MPVs accommodate 6-8 passengers (multi-purpose vehicles)
            - Hatchbacks accommodate 3-5 passengers (compact family cars)
            - Saloons accommodate 3-5 passengers (sedan style)
            - Estates accommodate 3-5 passengers (wagon style)
            - Coupes accommodate 2-4 passengers (sporty, limited space)
            - Convertibles accommodate 2-4 passengers (open-top, limited space)
            
            SEMANTIC UNDERSTANDING RULES:
            - If user mentions "SUV", they need 6-8 passenger capacity
            - If user mentions "family car", they need 3-5 passenger capacity
            - If user mentions "sports car", they need 2-4 passenger capacity
            - If user mentions "6-8 passengers", only SUVs and MPVs are suitable
            - If user mentions "3-5 passengers", hatchbacks, saloons, estates, and some SUVs are suitable
            - If user mentions "2-4 passengers", coupes, convertibles, and some hatchbacks are suitable
            
            Available unanswered questions:
            {json.dumps(filtered_questions, indent=2)}
            
            Based on the user's input and current context, which question is most important to ask next?
            Consider:
            1. What information would be most valuable to narrow down choices?
            2. What would help provide the best recommendations?
            3. Natural conversation flow
            4. If the user has provided enough information for a complete recommendation, return "none"
            
            IMPORTANT LOGIC:
            - If passenger count is 6-8, body type is automatically SUV (don't ask for body type)
            - If body type is SUV, passenger count is automatically 6-8 (don't ask for passenger count)
            - If passenger count is 0-2, body type can be any (ask for body type if needed)
            - If passenger count is 3-5, body type can be any (ask for body type if needed)
            
            Return ONLY the question ID (e.g., "passenger_count") of the most important question to ask next, or "none" if enough information is available.
            """
            
            response = self.llm.invoke(question_prompt)
            next_question_id = response.strip().lower()
            
            if next_question_id == "none":
                return None
            
            # Find the question object
            for question in filtered_questions:
                if question["id"] == next_question_id:
                    return question
            
            # Fallback to first unanswered question
            return filtered_questions[0] if filtered_questions else None
            
        except Exception as e:
            print(f"⚠️ Llama 3 question determination failed: {e}")
            # Fallback logic
            answered = self.conversation_state["answered_questions"]
            questions = [
                {"id": "passenger_count", "question": "How many people will usually be in the car?", "type": "radio", "options": ["Up to 2 passengers", "3 to 5 passengers", "6 to 8 passengers"]},
                {"id": "price_range", "question": "What's your budget range?", "type": "text", "options": []},
                {"id": "body_type", "question": "Which body types do you like?", "type": "multiselect", "options": ["Hatchback", "Estate", "SUV", "Coupe", "Convertible", "Saloon", "MPV"]},
                {"id": "make_pref", "question": "Any specific make preference?", "type": "text", "options": []}
            ]
            
            # Apply the same filtering logic
            filtered_questions = []
            for question in questions:
                if question["id"] not in answered:
                    if question["id"] == "body_type" and self.conversation_state.get("passenger_count") == "6-8":
                        continue
                    elif question["id"] == "passenger_count" and self.conversation_state.get("body_type") == "SUV":
                        continue
                    filtered_questions.append(question)
            
            for question in filtered_questions:
                return question
            return None

    def process_with_llama3_semantic(self, user_input: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            # Extract preferences using Llama 3
            preferences = self.extract_preferences_with_llama3(user_input)
            self._update_conversation_state_with_preferences(preferences)
            
            # Perform semantic search using Llama 3
            semantic_results = self.semantic_search_with_llama3(user_input, k=10)
            
            # Generate reasoning analysis using Llama 3
            reasoning_analysis = self.reasoning_agent_analysis(user_input, semantic_results)
            
            # Determine next question using Llama 3
            next_question = self.determine_next_question_with_llama3(user_input)
            
            # Generate conversation summary
            conversation_summary = self._generate_conversation_summary()
            
            response_time = time.time() - start_time
            
            return {
                "recommendations": reasoning_analysis,
                "next_question": next_question,
                "conversation_summary": conversation_summary,
                "semantic_matches": len(semantic_results),
                "pattern_matches": 0,  # Always 0 since we don't use pattern matching
                "combined_matches": len(semantic_results),
                "reasoning_used": self.semantic_ready,
                "semantic_used": self.semantic_ready,
                "response_time": response_time,
                "conversation_state": self.conversation_state.copy()
            }
        except Exception as e:
            print(f"⚠️ Llama 3 semantic RAG processing failed: {e}")
            return self._fallback_response(user_input)

    def _update_conversation_state_with_preferences(self, preferences: Dict[str, Any]):
        """Update conversation state with Llama 3 extracted preferences."""
        for key, value in preferences.items():
            if value is not None and value != "":
                self.conversation_state[key] = value
                self.conversation_state["answered_questions"].add(key)
        
        self.conversation_state["conversation_history"].append({
            "user_input": "preferences_updated",
            "timestamp": time.time()
        })

    def _generate_conversation_summary(self) -> str:
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

    def _fallback_response(self, user_input: str) -> Dict[str, Any]:
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

def test_specific_scenarios():
    """Test the specific scenarios mentioned in the requirements."""
    print("🧪 Testing Specific Scenarios")
    print("=" * 60)
    
    rag = Llama3SemanticRAG("src/data/stocked_cars.csv")
    
    # Test Scenario 1: BMW below £20000 + 6-8 passengers
    print("\n🔍 Test Scenario 1: BMW below £20000 + 6-8 passengers")
    print("-" * 50)
    
    # Reset conversation state
    rag.conversation_state = {
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
    
    # First query
    result1 = rag.process_with_llama3_semantic("I want a BMW below £20000")
    print(f"✅ Query 1: BMW below £20000")
    print(f"✅ Semantic matches: {result1['semantic_matches']}")
    print(f"✅ Summary: {result1['conversation_summary']}")
    print(f"✅ Next question: {result1['next_question']['question'] if result1['next_question'] else 'None'}")
    
    # Second query
    result2 = rag.process_with_llama3_semantic("6 to 8 passenger")
    print(f"✅ Query 2: 6 to 8 passenger")
    print(f"✅ Semantic matches: {result2['semantic_matches']}")
    print(f"✅ Summary: {result2['conversation_summary']}")
    print(f"✅ Recommendations: {result2['recommendations'][:200]}...")
    
    # Test Scenario 2: Electric vehicles for family
    print("\n🔍 Test Scenario 2: Electric vehicles for family")
    print("-" * 50)
    
    # Reset conversation state
    rag.conversation_state = {
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
    
    # First query
    result1 = rag.process_with_llama3_semantic("Show me electric vehicles for my family")
    print(f"✅ Query 1: Show me electric vehicles for my family")
    print(f"✅ Semantic matches: {result1['semantic_matches']}")
    print(f"✅ Summary: {result1['conversation_summary']}")
    
    # Second query
    result2 = rag.process_with_llama3_semantic("3 to 5 passenger")
    print(f"✅ Query 2: 3 to 5 passenger")
    print(f"✅ Semantic matches: {result2['semantic_matches']}")
    print(f"✅ Summary: {result2['conversation_summary']}")
    
    # Third query
    result3 = rag.process_with_llama3_semantic("Price range between 10 and 25k")
    print(f"✅ Query 3: Price range between 10 and 25k")
    print(f"✅ Semantic matches: {result3['semantic_matches']}")
    print(f"✅ Summary: {result3['conversation_summary']}")
    
    # Fourth query
    result4 = rag.process_with_llama3_semantic("SUV")
    print(f"✅ Query 4: SUV")
    print(f"✅ Semantic matches: {result4['semantic_matches']}")
    print(f"✅ Summary: {result4['conversation_summary']}")
    print(f"✅ Final recommendations: {result4['recommendations'][:300]}...")
    
    # Test Scenario 3: Diesel hatchback
    print("\n🔍 Test Scenario 3: Diesel hatchback")
    print("-" * 50)
    
    # Reset conversation state
    rag.conversation_state = {
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
    
    # First query
    result1 = rag.process_with_llama3_semantic("i need a new diesel hatchback less than 17000")
    print(f"✅ Query 1: i need a new diesel hatchback less than 17000")
    print(f"✅ Semantic matches: {result1['semantic_matches']}")
    print(f"✅ Summary: {result1['conversation_summary']}")
    
    # Second query
    result2 = rag.process_with_llama3_semantic("3 to 5 passengers")
    print(f"✅ Query 2: 3 to 5 passengers")
    print(f"✅ Semantic matches: {result2['semantic_matches']}")
    print(f"✅ Summary: {result2['conversation_summary']}")
    print(f"✅ Final recommendations: {result2['recommendations'][:300]}...")

def test_enhanced_rag():
    """Test the enhanced RAG system."""
    print("🧪 Testing Enhanced RAG System")
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
        print(f"✅ Pattern matches: {result['pattern_matches']}")
        print(f"✅ Combined matches: {result['combined_matches']}")
        print(f"✅ Reasoning used: {result['reasoning_used']}")
        print(f"✅ Response time: {result['response_time']:.3f}s")
        print(f"✅ Summary: {result['conversation_summary']}")
        print(f"✅ Recommendations: {result['recommendations'][:200]}...")

if __name__ == "__main__":
    test_specific_scenarios() 