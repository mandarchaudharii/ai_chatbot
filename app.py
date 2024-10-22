import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch
from streamlit_chat import message

# Specify the path to your local model directory
model_path = "mandarchaudharii/gfairep"  # Update this path

# Load model configuration and set pad_token_id
config = GPT2Config.from_pretrained(model_path)
# Set the pad token ID

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
 # Ensure tokenizer knows the pad token
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_path, config=config)
model.eval()

# Function to generate responses
def chat_with_bot(user_input):
    input_text = f"He: {user_input} She:"
    inputs = tokenizer.encode(input_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(inputs, max_length=150, num_return_sequences=1)  # Set pad_token_id here
    
    bot_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return bot_response.split("She:")[-1].strip()

# Streamlit interface
st.title("SavithaGPT")
st.write("Chat with your own savitha bhabhi! Type your message below:")

# Initialize message history
if 'message_history' not in st.session_state:
    st.session_state.message_history = []

# Display previous messages
for msg in st.session_state.message_history:
    message(msg['text'], is_user=msg['is_user'])

# User input
user_input = st.chat_input("You:")

if user_input:
    # Append user message to history
    st.session_state.message_history.append({"text": user_input, "is_user": True})
    
    # Display user message immediately
    message(user_input, is_user=True)

    # Generate bot response
    bot_response = chat_with_bot(user_input)
    st.session_state.message_history.append({"text": bot_response, "is_user": False})
    
    # Display bot response
    message(bot_response, is_user=False)
