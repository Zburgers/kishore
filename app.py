import json
import streamlit as st
import ollama

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sentiments" not in st.session_state:
    st.session_state.sentiments = []
if "memory" not in st.session_state:
    st.session_state.memory = {"summary": "I'm Babu AI! Let's chat!", "update_flag": False}
if "model" not in st.session_state:
    st.session_state.model = "mistral"
if "creativity" not in st.session_state:
    st.session_state.creativity = 0.7

# Function to update memory asynchronously
def update_memory(user_input, ai_response):
    # Create compact summarization prompt
    summary_prompt = f"""
    Update conversation memory (1-2 sentences). Keep these always:
    - My name is Babu AI
    - Current user focus: {user_input}
    - My last response: {ai_response[:100]}
    Previous summary: {st.session_state.memory['summary'][:150]}
    New compact summary:"""
    
    response = ollama.chat(
        model=st.session_state.model,
        messages=[{
            'role': 'user',
            'content': summary_prompt
        }]
    )
    
    # Process and trim summary
    new_summary = response['message']['content'].strip().split("\n")[0][:200]  # Strict length limit
    st.session_state.memory['summary'] = new_summary
    st.session_state.memory['update_flag'] = True  # Mark completion

# Optimized response generation
def generate_response(prompt):
    # Build efficient context
    context = f"""
    [Babu AI's Memory]
    {st.session_state.memory['summary']}
    
    [Current Chat]
    User: {prompt}
    """
    
    messages = [
        {'role': 'system', 'content': "You're Babu AI - friendly, simple, creative. Keep responses under 2 sentences."},
        {'role': 'system', 'content': context},
        *st.session_state.messages[-2:]  # Only last 2 exchanges
    ]
    
    response = ollama.chat(
        model=st.session_state.model,
        messages=messages,
        stream=True,
        options={"temperature": st.session_state.creativity}
    )
    return response

# Streamlit UI
st.set_page_config(page_title="Babu AI", page_icon="🤖", layout="centered")

# Custom styling
st.markdown("""
    <style>
    .stChatInput { position: fixed; bottom: 2rem; width: 85%; }
    .stChatMessage { padding: 12px; margin: 8px 0; border-radius: 15px; }
    .assistant { background: #e3f2fd!important; border-left: 4px solid #2196f3; }
    .user { background: #f5f5f5!important; border-left: 4px solid #9e9e9e; }
    .memory-box { background: #fff3e0!important; padding: 1rem; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# Sidebar for settings and memory
with st.sidebar:
    st.subheader("⚙️ Settings")
    
    # Model selection
    st.session_state.model = st.selectbox(
        "Select Model",
        ["mistral", "deepseek-r1"],
        index=["mistral", "deepseek-r1"].index(st.session_state.model)
    )
    
    # Creativity slider
    st.session_state.creativity = st.slider(
        "Creativity",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.creativity,
        help="Higher values make responses more creative."
    )
    
    st.divider()
    
    # Memory customization
    st.subheader("🧠 Customize Memory")
    custom_memory = st.text_area(
        "Add or edit memory manually:",
        value=st.session_state.memory["summary"],
        height=100
    )
    if st.button("Update Memory"):
        st.session_state.memory["summary"] = custom_memory
        st.success("Memory updated successfully!")
    
    st.divider()
    
    # Clear memory button
    if st.button("Clear Memory"):
        st.session_state.memory = {"summary": "I'm Babu AI! Let's start fresh!", "update_flag": False}
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.title("Babu AI 🤖")
st.caption("Your memory-aware conversational companion")

# Display chat messages
for msg in st.session_state.messages[-6:]:  # Show last 6 messages
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input and processing
if prompt := st.chat_input("Message Babu AI..."):
    if prompt.strip():
        # Store user message and display immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""
            
            # Generate streamed response
            for chunk in generate_response(prompt):
                content = chunk.get('message', {}).get('content', '')
                full_response += content
                response_container.markdown(full_response + "▌")
            
            # Finalize response
            response_container.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            # Update memory immediately after response
            update_memory(prompt, full_response)

# Handle pending memory updates
if st.session_state.memory['update_flag']:
    # Force UI refresh to show updated memory
    st.session_state.memory['update_flag'] = False
    st.rerun()

# Sentiment display
if st.session_state.sentiments:
    st.divider()
    latest = st.session_state.sentiments[-1]
    st.metric("Current Mood", 
             f"{latest['sentiment'].capitalize()} {['😊','😐','😟'][['positive','neutral','negative'].index(latest['sentiment'])]}",
             help="Mood analysis of the conversation")