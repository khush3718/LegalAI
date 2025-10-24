import streamlit as ST
import agent as Agent
import utils as Utils
import embed
import os as OS
import asyncio as ASYNCIO


def ensure_event_loop():
    """
    Ensure there is an asyncio event loop bound to the current (Streamlit) thread.
    Some libraries invoked by the agent expect an event loop to exist.
    """
    try:
        # Prefer get_running_loop (Py3.7+) to detect active loops
        ASYNCIO.get_running_loop()
    except RuntimeError:
        # No running loop; set one for this thread
        try:
            loop = ASYNCIO.get_event_loop()
        except RuntimeError:
            loop = ASYNCIO.new_event_loop()
            ASYNCIO.set_event_loop(loop)


def create_chat(id: str):
    ensure_event_loop()

    chat_container = ST.container()

    # Initialize chat history storage
    if "messages" not in ST.session_state:
        ST.session_state.messages = []

    # Cache NewsChat per chat id to avoid re-initialization on each rerun
    if "newschats" not in ST.session_state:
        ST.session_state.newschats = {}

    if id not in ST.session_state.newschats:
        try:
            ST.session_state.newschats[id] = Agent.NewsChat(id)
        except Exception as e:
            ST.error(f"Failed to initialize chat agent: {e}")
            return

    newschat = ST.session_state.newschats[id]

    # Display chat messages for this chat id
    for message in ST.session_state.messages:
        if message["id"] == id:
            chat_container.chat_message(message["role"]).write(message["content"])

    # Accept user input
    prompt = ST.chat_input(placeholder="Ask me about AI legal stuff in the EU", key=id)
    if prompt:
        chat_container.chat_message("user").write(prompt)
        with ST.spinner("Wait for it..."):
            try:
                assistant_response = newschat.ask(prompt)
            except Exception as e:
                assistant_response = f"Sorry, I ran into an error: {e}"
        chat_container.chat_message("assistant").write(f"{assistant_response}")

        ST.session_state.messages.append({"id": id, "role": "user", "content": prompt})
        ST.session_state.messages.append({"id": id, "role": "assistant", "content": assistant_response})


if __name__ == "__main__":
    ensure_event_loop()

    if not OS.path.exists(Utils.DB_FOLDER):
        document_name = "Artificial Intelligence Act"
        document_description = "Artificial Intelligence Act"
        text = embed.pdf_to_text(Utils.NITI_AYOG_URL)
        embed.embed_text_in_chromadb(text, document_name, document_description)

    create_chat("chat1")
