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
    input_text = f"He: {user_input} She:"
    inputs = tokenizer.encode(input_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    
    bot_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return bot_response.split("She:")[-1].strip()

# Streamlit interface
st.title("Chatbot")
st.write("Chat with the bot! Type your message below:")

user_input = st.text_input("You:")

if user_input:
    bot_response = chat_with_bot(user_input)
    st.write("Bot:", bot_response)
