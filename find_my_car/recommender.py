"""Car recommendation system using transformer models."""

import os
from typing import List, Optional
import pandas as pd
from transformers import pipeline

def load_classifier():
    """Load the zero-shot classification model."""
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        token=os.getenv("HF_TOKEN")
    )

def generate_response(classifier, prompt: str) -> str:
    """Generate response using the text classification model."""
    try:
        # Define car categories/features to check against
        categories = [
            "family car", "long distance", "durable", "fuel efficient",
            "luxury", "sporty", "budget friendly", "compact"
        ]
        
        # Check user's requirements against each category
        results = []
        try:
            # Single classification for all categories
            result = classifier(
                sequences=prompt,
                candidate_labels=categories,
                multi_label=True
            )
            
            # Extract categories with high confidence
            for label, score in zip(result["labels"], result["scores"]):
                if score > 0.7:
                    results.append(label)
                    
        except Exception as e:
            return f"Classification error: {str(e)}"
        
        if not results:
            return "I couldn't clearly identify your car preferences. Could you please be more specific about what you're looking for in a car?"
        
        # Format the response
        response = "Based on your requirements, you're looking for:\n"
        for category in results:
            response += f"- {category.title()}\n"
        
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"

def filter_cars(df: pd.DataFrame, requirements: List[str]) -> pd.DataFrame:
    """Filter cars based on requirements."""
    filtered_df = df.copy()
    
    # Apply filters based on identified requirements
    if "family car" in requirements:
        filtered_df = filtered_df[filtered_df["body_type"].isin(["suv", "wagon"])]
    if "long distance" in requirements:
        filtered_df = filtered_df[filtered_df["fuel_type"].isin(["hybrid", "diesel"])]
    if "durable" in requirements:
        filtered_df = filtered_df[
            (filtered_df["age"] <= 5) & 
            (filtered_df["mileage"] <= 80000)
        ]
    if "fuel efficient" in requirements:
        filtered_df = filtered_df[filtered_df["fuel_type"] == "hybrid"]
    if "luxury" in requirements:
        filtered_df = filtered_df[filtered_df["cost"] >= 40000]
    if "budget friendly" in requirements:
        filtered_df = filtered_df[filtered_df["cost"] <= 30000]
    if "compact" in requirements:
        filtered_df = filtered_df[filtered_df["body_type"].isin(["sedan", "hatchback"])]
    
    return filtered_df

def get_car_recommendation(classifier, user_query: str, df: pd.DataFrame) -> str:
    """Generate car recommendations using the classification model."""
    try:
        # Get user requirements analysis
        analysis = generate_response(classifier, user_query)
        if "couldn't clearly identify" in analysis:
            return analysis
            
        # Extract requirements from analysis
        requirements = [
            line.strip("- ").lower() 
            for line in analysis.split("\n") 
            if line.startswith("-")
        ]
        
        # Filter cars based on requirements
        filtered_cars = filter_cars(df, requirements)
        
        if filtered_cars.empty:
            return f"{analysis}\n\nI couldn't find any cars matching all your requirements. Try broadening your search criteria."
        
        # Sort by relevant criteria (prioritize newer cars with lower mileage)
        filtered_cars = filtered_cars.sort_values(["age", "mileage"])
        top_cars = filtered_cars.head(3)
        
        # Format response
        response = f"{analysis}\n\nBased on these requirements, here are the best matches:\n\n"
        
        for i, (_, car) in enumerate(top_cars.iterrows(), 1):
            response += (
                f"{i}. {car['make'].title()} {car['model'].title()}\n"
                f"   • {car['age']} years old\n"
                f"   • {car['body_type'].title()}, {car['fuel_type'].title()} fuel\n"
                f"   • {car['transmission_type'].title()} transmission\n"
                f"   • {car['mileage']:,.0f} miles\n"
                f"   • ${car['cost']:,.2f}\n\n"
            )
        
        return response

    except Exception as e:
        return f"Error generating recommendations: {str(e)}" 