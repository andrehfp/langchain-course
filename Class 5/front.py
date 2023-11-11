import streamlit as st

from agent import handle_chat

if "history" not in st.session_state:
    st.session_state["history"] = []

st.title("AI Chatbot")


def send_message():
    if st.session_state.user_input:
        user_message = st.session_state.user_input
        response = handle_chat(user_message)
        st.session_state["history"].append(("You", user_message))
        st.session_state["history"].append(("AI", response))
        st.session_state.user_input = ""


user_input = st.text_input(
    "Enter your message:", key="user_input", on_change=send_message
)


send_button = st.button("Send")

if send_button and st.session_state.user_input:
    send_message()

for idx, (user, message) in enumerate(reversed(st.session_state["history"])):
    if user == "You":
        st.text_area(f"You: {idx}", message, key=f"msg{idx}u", disabled=True)
    else:
        st.text_area(f"AI: {idx}", message, key=f"msg{idx}b", disabled=True)

st.markdown("---")
