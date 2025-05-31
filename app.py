"""
Car Recommendation Assistant using a transformer model locally.

This application helps users find their ideal car based on their preferences and requirements.
It uses a CSV-based car database and natural language processing to provide intelligent
car recommendations through a chat interface.
"""

import os
from typing import Optional, List, Dict, Tuple
import json
import pandas as pd
import streamlit as st
import torch
from transformers import pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Streamlit page configuration
st.set_page_config(
    page_title="Find My Car Assistant",
    page_icon="🚗",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "classifier" not in st.session_state:
    with st.spinner("Loading model... this may take a few minutes"):
        try:
            # Use a zero-shot classification model
            st.session_state.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                token=os.getenv("HF_TOKEN")
            )
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.stop()

def load_csv(file) -> Optional[pd.DataFrame]:
    """Load and validate a CSV file containing car data."""
    try:
        df = pd.read_csv(file)
        required_columns = [
            'make', 'model', 'age', 'body_type', 'fuel_type',
            'transmission_type', 'mileage', 'cost'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None

def format_car_features(df: pd.DataFrame) -> str:
    """Format car features for the prompt."""
    features = {
        'body_types': df['body_type'].unique().tolist(),
        'fuel_types': df['fuel_type'].unique().tolist(),
        'transmission_types': df['transmission_type'].unique().tolist(),
        'price_range': (float(df['cost'].min()), float(df['cost'].max())),
        'age_range': (int(df['age'].min()), int(df['age'].max())),
        'mileage_range': (float(df['mileage'].min()), float(df['mileage'].max()))
    }
    
    return (
        f"- Body types: {', '.join(features['body_types'])}\n"
        f"- Fuel types: {', '.join(features['fuel_types'])}\n"
        f"- Transmission types: {', '.join(features['transmission_types'])}\n"
        f"- Price range: ${features['price_range'][0]:,.2f} - ${features['price_range'][1]:,.2f}\n"
        f"- Age range: {features['age_range'][0]} - {features['age_range'][1]} years\n"
        f"- Mileage range: {features['mileage_range'][0]:,.0f} - {features['mileage_range'][1]:,.0f} miles"
    )

def generate_response(prompt: str) -> str:
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
            result = st.session_state.classifier(
                sequences=prompt,
                candidate_labels=categories,
                multi_label=True
            )
            
            # Extract categories with high confidence
            for label, score in zip(result["labels"], result["scores"]):
                if score > 0.7:
                    results.append(label)
                    
        except Exception as e:
            st.error(f"Classification error: {str(e)}")
            return "I encountered an error while analyzing your requirements. Please try again."
        
        if not results:
            return "I couldn't clearly identify your car preferences. Could you please be more specific about what you're looking for in a car?"
        
        # Format the response
        response = "Based on your requirements, you're looking for:\n"
        for category in results:
            response += f"- {category.title()}\n"
        
        return response
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None

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

def get_car_recommendation(user_query: str, df: pd.DataFrame) -> str:
    """Generate car recommendations using the classification model."""
    try:
        # Get user requirements analysis
        analysis = generate_response(user_query)
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

# Main UI
st.title("🚗 Find My Car Assistant")
st.markdown("""
Welcome to the Car Recommendation Assistant! This system uses the Mistral language model
to provide intelligent car recommendations based on your requirements.

The model runs locally on your machine, ensuring privacy and requiring no API keys.
""")

# File upload section
st.subheader("Upload Car Database")
uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type="csv",
    help="Upload a CSV file containing car information"
)

if uploaded_file is not None:
    df = load_csv(uploaded_file)
    if df is not None:
        st.session_state.df = df
        st.success("CSV file loaded successfully!")
        
        # Display the dataframe
        st.subheader("Current Car Database")
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

# Chat interface
st.subheader("Chat with Car Assistant")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about your ideal car..."):
    if st.session_state.df is None:
        st.error("Please upload a car database first!")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Finding the best matches..."):
                response = get_car_recommendation(prompt, st.session_state.df)
                st.markdown(response)
            
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response}) 