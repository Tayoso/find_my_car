import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import the Anthropic client
from anthropic import Anthropic
ANTHROPIC_AVAILABLE = True

# Import utility functions
from src.utils import load_car_data, get_car_context, stream_claude_chat, is_car_query
from src.prompts import get_car_sales_system_prompt

# Load car data
car_data_summary, car_df = load_car_data()


# ----------------------
# Streamlit Page Config
# ----------------------
st.set_page_config(page_title="Car Sales Assistant", page_icon="🚗", layout="centered")

st.title("🚗 Car Sales Assistant")
st.caption("A smart car sales chatbot powered by Claude 3.5 Sonnet with access to our car inventory.")



# ----------------------
# Sidebar Settings
# ----------------------
with st.sidebar:
    st.subheader("Settings")
    
    # API Key - hidden, only get from environment
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.error("⚠️ ANTHROPIC_API_KEY not found in environment variables")
        st.info("Please set your API key in the .env file")
    else:
        st.success("✅ API key loaded from environment")
    
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
    max_tokens = st.number_input("Max tokens (response)", min_value=64, max_value=4096, value=512, step=64)
    
    # Get system prompt from prompts module
    system_prompt = get_car_sales_system_prompt(car_data_summary)
    
    st.markdown("---")
    st.subheader("🚗 Car Query Examples")
    st.write("Try asking about:")
    st.write("• 'I want a BMW below £20000'")
    st.write("• 'Show me electric vehicles for my family'")
    st.write("• 'Show me electric vehicles with lots of space'")
    st.write("• 'I need a new diesel hatchback less than 17000'")
    st.write("• 'I need an Audi A3'")
    st.write("• 'Who is the president of USA?'")
    st.write("• 'I want a BMW between 15 and 20k'")
    st.write("• 'I want a BM under 20k'")
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
        if not ANTHROPIC_AVAILABLE:
            st.error(
                "The `anthropic` Python package is not installed or failed to import.\n\n"
                "Install with: `pip install anthropic`\n"
                "Make sure you have a valid Anthropic API key."
            )
            assistant_text = "I couldn't connect to the Claude backend. Please check your setup."
            st.session_state.messages.append({"role": "assistant", "content": assistant_text})
        elif not api_key:
            st.error(
                "Please enter your Anthropic API key in the sidebar.\n\n"
                "Get your API key from https://console.anthropic.com/"
            )
            assistant_text = "I need an API key to connect to Claude."
            st.session_state.messages.append({"role": "assistant", "content": assistant_text})
        else:
            try:
                # Check if this is a car-related query and add context
                is_car_query = is_car_query(prompt)
                
                # The intelligent stuff happens here!
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
                response_stream = stream_claude_chat(
                    messages=messages_with_context,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                assistant_text = st.write_stream(response_stream)

                # Persist the assistant message in history
                st.session_state.messages.append({"role": "assistant", "content": assistant_text or ""})

            except Exception as e:
                st.error(
                    "Oops—couldn't generate a reply. "
                    "Check that your API key is valid and you have sufficient credits.\n\n"
                    f"Error: {e}"
                )
                assistant_text = "There was an error connecting to Claude."
                st.session_state.messages.append({"role": "assistant", "content": assistant_text})


# ----------------------
# Footer
# ----------------------
st.markdown(
    "<div style='text-align:center;color:gray;font-size:0.9em;margin-top:1rem;'>"
    "Powered by Streamlit + Claude 3.5 Sonnet + Car Inventory Data</div>",
    unsafe_allow_html=True,
)
