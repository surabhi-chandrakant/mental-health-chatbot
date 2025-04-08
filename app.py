import streamlit as st
import datetime
import gspread
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from oauth2client.service_account import ServiceAccountCredentials
import json
from io import StringIO

# Set up Google Sheets
def init_gsheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    json_key = json.loads(st.secrets["gcp_service_account"])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(json_key, scope)
    client = gspread.authorize(creds)
    sheet = client.open("mood_log").sheet1
    return sheet

sheet = init_gsheet()

# Load Hugging Face DialoGPT
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Streamlit interface
st.set_page_config(page_title="Mental Health Bot", page_icon="🧠")
st.title("🧠 Mental Health Buddy")
st.markdown("Talk to me about how you're feeling. I'm here to listen 💙")

user_input = st.text_input("How are you feeling today?", "")

if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None
    st.session_state.step = 0

if user_input:
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    bot_input_ids = torch.cat([st.session_state.chat_history_ids, input_ids], dim=-1) if st.session_state.step > 0 else input_ids

    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=100,
        top_p=0.7,
        temperature=0.8
    )

    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    st.success(response)

    # Save to Google Sheet
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sheet.append_row([now, user_input, response])

    # Save session
    st.session_state.chat_history_ids = chat_history_ids
    st.session_state.step += 1
