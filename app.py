import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load model and tokenizer from Hugging Face
model_name = "mandarchaudharii/gfairep"  # Update with your model's path
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Ensure the model is in evaluation mode
model.eval()

# Function to generate responses
def chat_with_bot(user_input):
    input_text = f"He: {user_input} GF:"
    inputs = tokenizer.encode(input_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    
    bot_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return bot_response.split("GF:")[-1].strip()

# Streamlit interface
st.set_page_config(page_title="Chatbot", layout="wide")

st.title("Chatbot")
st.write("Chat with GF! Type your message below:")

# Create a container for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    if message['role'] == 'user':
        st.markdown(f"<div style='text-align: right; color: blue;'>**You:** {message['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align: left; color: green;'>**GF:** {message['content']}</div>", unsafe_allow_html=True)

# Text input for user
user_input = st.text_input("Type your message:", key="input")

if st.button("Send"):
    if user_input:
        # Store the user's message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get the bot's response
        bot_response = chat_with_bot(user_input)
        
        # Store the bot's response
        st.session_state.messages.append({"role": "bot", "content": bot_response})
        
        # Clear the input box
        st.session_state.input = ""

# Add a footer
st.markdown("---")
