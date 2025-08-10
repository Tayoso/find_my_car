import os
import time
import streamlit as st
import pandas as pd
import ollama

# Load car data
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
        
        # Add sample cars with VRM for context
        sample_cars = df.head(5).to_dict('records')
        for car in sample_cars:
            car_summary += f"\n- VRM: {car['VRM']} | {car['MAKE']} {car['MODEL']} ({car['AGE_GROUP']}, {car['BODY_TYPE']}, {car['FUEL_TYPE']}, £{car['PRICE']:,}, {car['MILEAGE']:,} miles)"
        
        return car_summary, df
        
    except Exception as e:
        return f"Error loading car data: {str(e)}", None

# Load car data
car_data_summary, car_df = load_car_data()

# Function to get specific car data for LLM context
def get_car_context(query, df):
    """Get relevant car data based on query for LLM context"""
    if df is None:
        return ""
    
    query_lower = query.lower()
    context = ""
    
    # Extract criteria from query
    import re
    
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
        for _, car in filtered_df.head(10).iterrows():  # Show up to 10 for context
            context += f"- VRM: {car['VRM']} | {car['MAKE']} {car['MODEL']} ({car['AGE_GROUP']}, {car['BODY_TYPE']}, {car['FUEL_TYPE']}, £{car['PRICE']:,}, {car['MILEAGE']:,} miles)\n"
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
            make_cars = df[df['MAKE'] == requested_make]
            if not make_cars.empty:
                context += f"\nAvailable {requested_make} cars (different body types):\n"
                for _, car in make_cars.head(5).iterrows():
                    context += f"- VRM: {car['VRM']} | {car['MAKE']} {car['MODEL']} ({car['AGE_GROUP']}, {car['BODY_TYPE']}, {car['FUEL_TYPE']}, £{car['PRICE']:,}, {car['MILEAGE']:,} miles)\n"
    
    return context


# ----------------------
# Streamlit Page Config
# ----------------------
st.set_page_config(page_title="Car Sales Assistant (Llama)", page_icon="🚗", layout="centered")

st.title("🚗 Car Sales Assistant (Llama)")
st.caption("A smart car sales chatbot powered by Llama 3 with access to our car inventory.")

# ----------------------
# Sidebar Settings
# ----------------------
with st.sidebar:
    st.subheader("Settings")
    
    # Model selection
    model = st.selectbox(
        "Ollama Model",
        ["llama3.1", "llama3", "llama2", "mistral", "codellama"],
        help="Select the Ollama model to use",
    )
    
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
    max_tokens = st.number_input("Max tokens (response)", min_value=64, max_value=4096, value=512, step=64)
    
    system_prompt = st.text_area(
        "System prompt",
        value=f"""You are a car sales assistant with access to our car inventory. CRITICAL RULES:

1. ONLY recommend cars that are EXPLICITLY listed in the CAR INVENTORY DATA below
2. NEVER invent, create, or suggest cars that are not in the inventory data
3. NEVER change body types, makes, or models - use exactly what's in the data
4. ALWAYS include the VRM (Vehicle Registration Mark) when recommending specific cars
5. ONLY recommend vehicles that EXACTLY match the user's criteria (make, model, body type, price range, fuel type)
6. NEVER include thinking, reasoning, or corrections in your response - just provide the matches
7. NEVER repeat the same car multiple times
8. Use this format for car recommendations: "VRM: [VRM] | [MAKE] [MODEL] ([AGE_GROUP], [BODY_TYPE], [FUEL_TYPE], £[PRICE], [MILEAGE] miles)"
9. If asked about anything other than our car inventory, politely redirect to car-related questions
10. RESPONSE FORMAT:
    - If you have 3+ cars: "Here are 3 great options for you:" then list exactly 3
    - If you have 1-2 cars: "I found [X] car(s) that match your criteria:" then list what you have
    - If you have 0 cars: "I couldn't find any cars matching your exact criteria. Try broadening your search."
11. CRITICAL: If no cars match the criteria, say "No cars found" - DO NOT invent cars

CAR INVENTORY DATA:
{car_data_summary}""",
        height=400,
    )
    st.markdown("---")
    st.write("**Tip:** Make sure Ollama is running with `ollama serve`")
    
    st.markdown("---")
    st.subheader("🚗 Car Query Examples")
    st.write("Try asking about:")
    st.write("• 'I want a BMW below £20000'")
    st.write("• 'Show me electric vehicles for my family'")
    st.write("• 'I need a new diesel hatchback less than 17000'")
    st.write("• 'I need an Audi A3'")
    st.write("• 'I want a BMW between 15 and 20k'")
    st.write("• 'I want a BMW under 20k'")
    st.write("• 'I want a Nissan saloon between 12 and 20k'")


# ----------------------
# Session State (history)
# ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "assistant", "content": "Hello! I'm your car sales assistant. Ask me about our inventory! 🚗"},
    ]

# If the user edits the system prompt, keep the latest one at index 0
if st.session_state.messages and st.session_state.messages[0]["role"] == "system":
    if st.session_state.messages[0]["content"] != system_prompt.strip():
        st.session_state.messages[0]["content"] = system_prompt.strip()


# ----------------------
# Helper: stream from Ollama
# ----------------------
def stream_ollama_chat(messages, model: str, temperature: float, max_tokens: int):
    """
    Yields chunks of text from Ollama's streaming chat API.
    """
    client = ollama.Client()
    
    # Convert messages to Ollama format
    ollama_messages = []
    for msg in messages:
        if msg["role"] == "system":
            # Ollama uses system message as the first user message
            ollama_messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "user":
            ollama_messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant":
            ollama_messages.append({"role": "assistant", "content": msg["content"]})
    
    stream = client.chat(
        model=model,
        messages=ollama_messages,
        stream=True,
        options={
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    )
    
    # The Ollama client yields message objects
    for chunk in stream:
        if chunk.message.content:
            yield chunk.message.content


# ----------------------
# Render Chat History
# ----------------------
for m in st.session_state.messages:
    if m["role"] == "system":
        continue  # don't render the system message in the chat area
    with st.chat_message("assistant" if m["role"] == "assistant" else "user"):
        st.write(m["content"])


# ----------------------
# Chat Input & Response
# ----------------------
prompt = st.chat_input("Type your question…")
if prompt:
    # Show user's message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        try:
            # Check if this is a car-related query and add context
            car_keywords = ['car', 'vehicle', 'bmw', 'audi', 'mercedes', 'ford', 'volkswagen', 'toyota', 'honda', 'nissan', 'suv', 'saloon', 'hatchback', 'estate', 'electric', 'petrol', 'diesel', 'hybrid', 'price', 'under', 'buy', 'purchase', 'kia', 'mazda', 'peugeot', 'renault', 'seat', 'skoda', 'volvo', 'jaguar', 'land rover', 'mini', 'fiat', 'alfa romeo', 'citroen', 'dacia', 'ds', 'hyundai', 'mg', 'mitsubishi', 'polestar', 'suzuki', 'vauxhall', 'vrm']
            is_car_query = any(keyword in prompt.lower() for keyword in car_keywords)
            
            if is_car_query and car_df is not None:
                # Get specific car context for this query
                car_context = get_car_context(prompt, car_df)
                if car_context:
                    # Add car context to the system message temporarily
                    enhanced_system = system_prompt + f"\n\nRELEVANT CARS FOR THIS QUERY:\n{car_context}"
                    messages_with_context = [{"role": "system", "content": enhanced_system}] + st.session_state.messages[1:]
                else:
                    messages_with_context = st.session_state.messages
            else:
                messages_with_context = st.session_state.messages

            # Stream tokens into the UI
            response_stream = stream_ollama_chat(
                messages=messages_with_context,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            assistant_text = st.write_stream(response_stream)

            # Persist the assistant message in history
            st.session_state.messages.append({"role": "assistant", "content": assistant_text or ""})

        except Exception as e:
            st.error(
                "Oops—couldn't generate a reply. "
                "Check that Ollama is running and the model is available.\n\n"
                f"Error: {e}"
            )
            assistant_text = "There was an error connecting to Ollama."
            st.session_state.messages.append({"role": "assistant", "content": assistant_text})


# ----------------------
# Footer
# ----------------------
st.markdown(
    "<div style='text-align:center;color:gray;font-size:0.9em;margin-top:1rem;'>"
    "Powered by Streamlit + Ollama + Car Inventory Data</div>",
    unsafe_allow_html=True,
)
