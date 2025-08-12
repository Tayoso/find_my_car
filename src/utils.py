import os
import pandas as pd
import re
from anthropic import Anthropic

def load_car_data():
    """Load and process car data for the chatbot"""
    try:
        df = pd.read_csv('data/stocked_cars.csv')
        
        # Create a summary of the car data
        car_summary = f"""
        CAR INVENTORY DATA SUMMARY:
        - Total cars available: {len(df)}
        - Price range: £{df['PRICE'].min():,.0f} - £{df['PRICE'].max():,.0f}
        - Average price: £{df['PRICE'].mean():,.0f}
        - Makes available: {', '.join(df['MAKE'].unique())}
        - Body types: {', '.join(df['BODY_TYPE'].unique())}
        - Fuel types: {', '.join(df['FUEL_TYPE'].unique())}
        - Age groups: {', '.join(df['AGE_GROUP'].unique())}
        
        SAMPLE CARS WITH VRM:
        """
        
        # Add sample cars with VRM for context (sorted by desirability)
        sample_cars = df.sort_values('RETAIL_DESIRABILITY_SCORE', ascending=False).head(5).to_dict('records')
        for car in sample_cars:
            desirability = f"{car['RETAIL_DESIRABILITY_SCORE']:.2f}"
            car_summary += f"\n- VRM: {car['VRM']} | {car['MAKE']} {car['MODEL']} ({car['AGE_GROUP']}, {car['BODY_TYPE']}, {car['FUEL_TYPE']}, £{car['PRICE']:,}, {car['MILEAGE']:,} miles, Desirability: {desirability})"
        
        return car_summary, df
        
    except Exception as e:
        return f"Error loading car data: {str(e)}", None

def get_car_context(query, df):
    """Get relevant car data based on query for LLM context.
    
    Args:
        query (str): The user's query
        df (pd.DataFrame): The car inventory data
        
    Returns:
        str: The context for the LLM
    """

    if df is None:
        return ""
    
    query_lower = query.lower()
    context = ""
    
    # Extract criteria from query
    # Extract make
    makes = df['MAKE'].unique()
    requested_make = None
    for make in makes:
        if make.lower() in query_lower:
            requested_make = make
            break
    
    # Extract body type
    body_types = ['suv', 'saloon', 'hatchback', 'estate', 'convertible']
    requested_body_type = None
    for body_type in body_types:
        if body_type in query_lower:
            requested_body_type = body_type
            break
    
    # Extract price range
    min_price = None
    max_price = None
    if 'between' in query_lower and 'and' in query_lower:
        # Extract numbers for "between X and Y"
        numbers = re.findall(r'\d+', query)
        if len(numbers) >= 2:
            min_price = int(numbers[0]) * 1000
            max_price = int(numbers[1]) * 1000
    elif 'under' in query_lower:
        # Extract number for "under X"
        numbers = re.findall(r'\d+', query)
        if numbers:
            max_price = int(numbers[0]) * 1000
    
    # Filter cars based on exact criteria
    filtered_df = df.copy()
    
    if requested_make:
        filtered_df = filtered_df[filtered_df['MAKE'] == requested_make]
    
    if requested_body_type:
        filtered_df = filtered_df[filtered_df['BODY_TYPE'] == requested_body_type]
    
    if min_price is not None:
        filtered_df = filtered_df[filtered_df['PRICE'] >= min_price]
    
    if max_price is not None:
        filtered_df = filtered_df[filtered_df['PRICE'] <= max_price]
    
    # Get cars that match ALL criteria
    if not filtered_df.empty:
        car_count = len(filtered_df)
        context += f"\nFound {car_count} car(s) matching your exact criteria:\n"
        
        # Intelligent diversity selection for recommendations
        if car_count >= 3:
            # Try to select diverse makes/models when possible
            diverse_selection = select_diverse_cars(filtered_df, max_cars=10)
            context += "(Showing diverse selection for better variety):\n"
            for car in diverse_selection:
                desirability = f"{car['RETAIL_DESIRABILITY_SCORE']:.2f}"
                context += f"- VRM: {car['VRM']} | {car['MAKE']} {car['MODEL']} ({car['AGE_GROUP']}, {car['BODY_TYPE']}, {car['FUEL_TYPE']}, £{car['PRICE']:,}, {car['MILEAGE']:,} miles, Desirability: {desirability})\n"
        else:
            # If fewer cars, just show what we have
            for _, car in filtered_df.head(10).iterrows():
                desirability = f"{car['RETAIL_DESIRABILITY_SCORE']:.2f}"
                context += f"- VRM: {car['VRM']} | {car['MAKE']} {car['MODEL']} ({car['AGE_GROUP']}, {car['BODY_TYPE']}, {car['FUEL_TYPE']}, £{car['PRICE']:,}, {car['MILEAGE']:,} miles, Desirability: {desirability})\n"
    else:
        context += f"\nNO CARS FOUND matching your exact criteria.\n"
        context += f"DO NOT invent or suggest cars that don't exist in our inventory.\n"
        # Show what criteria were requested for debugging
        debug_info = []
        if requested_make:
            debug_info.append(f"Make: {requested_make}")
        if requested_body_type:
            debug_info.append(f"Body type: {requested_body_type}")
        if min_price is not None:
            debug_info.append(f"Min price: £{min_price:,}")
        if max_price is not None:
            debug_info.append(f"Max price: £{max_price:,}")
        if debug_info:
            context += f"Requested criteria: {', '.join(debug_info)}\n"
        
        # Show what's actually available for this make/body type
        if requested_make:
            make_cars = df[df['MAKE'] == requested_make].sort_values('RETAIL_DESIRABILITY_SCORE', ascending=False)
            if not make_cars.empty:
                context += f"\nAvailable {requested_make} cars (different body types, ranked by desirability):\n"
                for _, car in make_cars.head(5).iterrows():
                    desirability = f"{car['RETAIL_DESIRABILITY_SCORE']:.2f}"
                    context += f"- VRM: {car['VRM']} | {car['MAKE']} {car['MODEL']} ({car['AGE_GROUP']}, {car['BODY_TYPE']}, {car['FUEL_TYPE']}, £{car['PRICE']:,}, {car['MILEAGE']:,} miles, Desirability: {desirability})\n"
    
    return context

def select_diverse_cars(df, max_cars=10):
    """
    Select diverse cars prioritizing different makes/models when possible,
    but falling back to desirability ranking when diversity isn't possible.
    """
    if len(df) <= max_cars:
        return df.to_dict('records')
    
    # First, try to get diverse makes/models
    diverse_cars = []
    makes_models_seen = set()
    
    # Sort by desirability first to ensure quality
    df_sorted = df.sort_values('RETAIL_DESIRABILITY_SCORE', ascending=False)
    
    for _, car in df_sorted.iterrows():
        make_model = f"{car['MAKE']}_{car['MODEL']}"
        
        if make_model not in makes_models_seen:
            # This is a new make/model combination
            diverse_cars.append(car)
            makes_models_seen.add(make_model)
            
            if len(diverse_cars) >= max_cars:
                break
    
    # If we didn't get enough diverse cars, fill with top desirability
    if len(diverse_cars) < max_cars:
        remaining_cars = df_sorted[~df_sorted.index.isin([car.name for car in diverse_cars])]
        for _, car in remaining_cars.head(max_cars - len(diverse_cars)).iterrows():
            diverse_cars.append(car)
    
    return diverse_cars

def stream_claude_chat(messages, api_key: str, temperature: float, max_tokens: int):
    """
    Yields chunks of text from Claude's streaming chat API.
    """
    client = Anthropic(api_key=api_key)
    
    # Convert messages to Claude format
    claude_messages = []
    for msg in messages:
        if msg["role"] == "system":
            # Claude uses system message differently
            continue
        elif msg["role"] == "user":
            claude_messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant":
            claude_messages.append({"role": "assistant", "content": msg["content"]})
    
    # Add system message to the beginning
    system_message = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    
    stream = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=claude_messages,
        system=system_message,
        stream=True,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    
    # The Claude client yields content blocks
    for chunk in stream:
        if chunk.type == "content_block_delta":
            if chunk.delta.text:
                yield chunk.delta.text

def is_car_query(prompt):
    """Check if the query is car-related"""
    car_keywords = [
        'car', 'vehicle', 'bmw', 'audi', 'mercedes', 'ford', 'volkswagen', 
        'toyota', 'honda', 'nissan', 'suv', 'saloon', 'hatchback', 'estate', 
        'electric', 'petrol', 'diesel', 'hybrid', 'price', 'under', 'buy', 
        'purchase', 'kia', 'mazda', 'peugeot', 'renault', 'seat', 'skoda', 
        'volvo', 'jaguar', 'land rover', 'mini', 'fiat', 'alfa romeo', 
        'citroen', 'dacia', 'ds', 'hyundai', 'mg', 'mitsubishi', 'polestar', 
        'suzuki', 'vauxhall', 'vrm'
    ]
    return any(keyword in prompt.lower() for keyword in car_keywords)
