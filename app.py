import streamlit as st
import requests
import uuid
import logging
from datetime import datetime

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
API_URL = "http://192.168.0.35:8501/chat"

class ChatInterface:
    """
    An object-oriented class to manage the chat interface, state, and API communication.
    """
    def __init__(self):
        st.set_page_config(page_title="PlanetGuard AI", layout="wide")
        self.initialize_state()

    def initialize_state(self):
        """Initializes Streamlit's session state variables."""
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
            logging.info(f"New session created: {st.session_state.session_id}")

        if "messages" not in st.session_state:
            st.session_state.messages = [{
                "role": "assistant",
                "content": "Hello! I am the PlanetGuard AI Assistant. How can I help you with your security analysis today?",
                "timestamp": datetime.now().isoformat()
            }]

    def _call_api(self, message: str) -> dict:
        """Private method to handle the API call and error handling."""
        payload = {"message": message, "session_id": st.session_state.session_id}
        try:
            response = requests.post(API_URL, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            return {
                "response": "❌ I'm sorry, I'm having trouble connecting to my services. Please try again.",
                "timestamp": datetime.now().isoformat()
            }

    def _add_message(self, role: str, content: str, api_response: dict = None):
        """Adds a message to the session state history."""
        message = {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
        if api_response:
            message["options"] = api_response.get("options")
            message["data"] = api_response.get("data")
            message["timestamp"] = api_response.get("timestamp", message["timestamp"])
        st.session_state.messages.append(message)

    def _handle_user_input(self, prompt: str):
        """Processes user input from the chat box or a button click."""
        # Add user message to history
        self._add_message("user", prompt)

        # Get assistant's response
        with st.spinner("Analyzing..."):
            api_response = self._call_api(prompt)
            self._add_message("assistant", api_response.get("response"), api_response)

    # In your ChatInterface class in app.py

    def _render_message(self, message: dict, index: int):
        """Renders a single chat message, including any special data like alerts or buttons."""
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # --- CORRECTED LOGIC ---
            # This now checks if 'data' is a NON-EMPTY list.
            if message["role"] == "assistant" and isinstance(message.get("data"), list) and message.get("data"):
                self._render_alert_data(message["data"], index)

            # This will now correctly render if 'data' is empty but 'options' is not.
            elif message["role"] == "assistant" and isinstance(message.get("options"), list):
                self._render_option_buttons(message["options"], index)

    def _render_alert_data(self, data_list: list, msg_index: int):
        """Renders the detailed list of alerts with interactive elements."""
        for i, item in enumerate(data_list):
            item_key = f"item_{msg_index}_{i}"
            original_result = item.get("original_result", {})

            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{item.get('title', 'Untitled Alert')}**")
                st.caption(original_result.get('summary', 'No summary available.'))
            
            with col2:
                if payload := item.get("payload"):
                    if st.button("Select Alert", key=f"select_{item_key}"):
                        self._handle_user_input(payload)
                        st.rerun()

            with st.expander("View Details Table"):
                if table_data := original_result.get('details_table', []):
                    st.dataframe(table_data, use_container_width=True)
                else:
                    st.write("No detailed data available.")

    def _render_option_buttons(self, options: list, msg_index: int):
        """Renders a row of general purpose buttons."""
        cols = st.columns(len(options))
        for i, option in enumerate(options):
            with cols[i]:
                if st.button(option["title"], key=f"option_{msg_index}_{i}", use_container_width=True):
                    self._handle_user_input(option["payload"])
                    st.rerun()

    def run(self):
        """The main method to run the Streamlit application."""
        st.title("PlanetGuard AI Assistant")
        self._apply_custom_css()

        # Display chat history
        for i, msg in enumerate(st.session_state.messages):
            self._render_message(msg, i)

        # Handle new user input
        if prompt := st.chat_input("Ask about security events..."):
            self._handle_user_input(prompt)
            st.rerun()

    def _apply_custom_css(self):
        """Applies custom CSS styles to the application."""
        st.markdown("""
        <style>
            /* Main content area */
            .st-emotion-cache-1kyxpmj { padding-top: 2rem; }
            /* Chat message bubbles */
            .stChatMessage {
                border-radius: 0.75rem;
                padding: 1rem;
                margin-bottom: 1rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            /* User message bubble */
            [data-testid="chat-message-container"]:has([data-testid="chat-avatar-user"]) {
                background-color: #007bff;
                color: white;
            }
            /* Assistant message bubble */
            [data-testid="chat-message-container"]:has([data-testid="chat-avatar-assistant"]) {
                background-color: #f0f2f6;
                color: #31333F;
            }
            /* Style for interactive option buttons */
            .stButton>button {
                border-radius: 0.5rem;
                border: 1px solid #007bff;
                color: #007bff;
                background-color: transparent;
                width: 100%;
                margin-top: 0.3rem;
                margin-bottom: 0.3rem;
                transition: all 0.2s ease-in-out;
            }
            .stButton>button:hover {
                border-color: #0056b3;
                color: white;
                background-color: #0056b3;
            }
            .stExpander {
                border: 1px solid #e0e0e0 !important;
                border-radius: 0.5rem !important;
                margin-top: 0.5rem;
            }
        </style>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    chat_app = ChatInterface()
    chat_app.run()