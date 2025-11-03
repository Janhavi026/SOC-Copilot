# chatbot.py

import logging
import json
import asyncio
import re
import pprint
import time  # <--- IMPORTED
from typing import Dict, Any, List, Optional, Callable

import ollama
from query_classifier import QueryClassifier
from embedder import Embedder
from retriever import Retriever
from elasticsearch_realtime_query import ElasticsearchQueryHandler
from unified_response_orchestrator import UnifiedResponseOrchestrator

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s")
logger = logging.getLogger(__name__)


class DynamicChatbot:
    """
    A multi-faceted chatbot that classifies user intent and routes queries
    to the appropriate handler for real-time data, RAG, or direct actions.
    """
    def __init__(self, ollama_base_url: str, ollama_model: str, embedder: Embedder, retriever: Retriever, es_query_handler: ElasticsearchQueryHandler):
        """
        Initializes the chatbot and its components.
        """
        self.llm_model = ollama_model
        self.embedder = embedder
        self.retriever = retriever
        self.es_query_handler = es_query_handler
        self.session_contexts: Dict[str, Dict[str, Any]] = {}

        logger.info(f"Initializing Ollama client with base URL: {ollama_base_url} and model: {self.llm_model}")
        self.client = ollama.Client(host=ollama_base_url)

        self.query_classifier = QueryClassifier(
            ollama_chatbot=self,
            ollama_client=self.client,
            ollama_model=self.llm_model,
            data_directory="data"
        )
        
        self.response_orchestrator = UnifiedResponseOrchestrator(elastic_query_handler=self.es_query_handler)

        self.handler_map: Dict[str, Callable] = {
            'real_time_alert': self._handle_realtime_alert_query,
            'status_inquiry': self._handle_status_inquiry,
            'follow_up': self._handle_details_followup_query,
            'action_command': self._handle_action_command,
            'generic': self._handle_generic_query,
        }

        self.system_prompts = {
            "generic": "You are a helpful AI assistant. Your task is to answer the user's question using the information from the provided context. Synthesize a response that is helpful and directly answers the question based on the contextual documents."
        }

    # --- NEW HELPER METHOD ---
    def _wait_for_action_completion(self, sr_no: int) -> Dict[str, Any]:
        """
        Polls the database for an action's final status and returns the result message.
        This makes the API call block until the action is complete.
        """
        POLLING_INTERVAL_SECONDS = 2
        TIMEOUT_SECONDS = 240
        start_time = time.time()
        logger.info(f"Now waiting for completion of SR_NO: {sr_no}...")

        while time.time() - start_time < TIMEOUT_SECONDS:
            status_result = self.response_orchestrator.get_action_status_by_sr_no(sr_no)
            # Example status_result: {"db_status": "inprocess"}
            current_status = status_result.get("db_status", "pending").lower()

            # Check if the status has changed from a pending state
            if current_status not in ['inprocess', 'pending']:
                logger.info(f"Detected final status '{current_status}' for SR_NO {sr_no}.")
                final_status_message = "has been successfully completed."
                if current_status in ['failed', 'error']:
                    final_status_message = f"has failed with status: {current_status.upper()}."
                
                icon = "✅" if current_status not in ['failed', 'error'] else "❌"
                message = f"{icon} Update: Your action for **SR_NO {sr_no}** {final_status_message}"
                
                return {"message": message}

            time.sleep(POLLING_INTERVAL_SECONDS)
        
        # If we exit the loop because of a timeout
        logger.warning(f"Timed out waiting for SR_NO {sr_no}.")
        return {"message": f"⏳ Timed out after {TIMEOUT_SECONDS} seconds waiting for a response for action **SR_NO {sr_no}**. Please check its status manually later."}

    def _initialize_session(self, session_id: str) -> Dict[str, Any]:
        """Ensures a session context exists, creating one if it's new."""
        if session_id not in self.session_contexts:
            self.session_contexts[session_id] = {
                'history': [],
                'last_query_results': [],
                'selected_alert': None,
                'awaiting_confirmation': False,
                'pending_action': None,
                'pending_sr_numbers': [],
            }
        
        session = self.session_contexts[session_id]
        session['session_id'] = session_id
        return session

    def unified_response_orchestrator(self, user_prompt: str, session_id: str) -> Dict[str, Any]:
        """The main entry point for processing a user's message."""
        session = self._initialize_session(session_id)
        
        logger.info(f"--- Orchestrating response for session {session_id} ---")
        logger.info(f"User Prompt: '{user_prompt}'")
        
        if session.get('awaiting_confirmation'):
            return self._handle_action_confirmation(user_prompt, session)

        if user_prompt.startswith(('select_alert ', 'execute_action ', 'show_json ')):
            return self._handle_direct_command(user_prompt, session)

        classification = self.query_classifier.comprehensive_classify(user_prompt, session_id)
        logger.info("--- Query Classification Result ---")
        pprint.pprint(classification)
        logger.info("-----------------------------------")
        
        query_type = classification.get('primary_classification', {}).get('query_type', 'generic')
        logger.info(f"Classified query type as '{query_type}'. Routing to handler...")

        handler = self.handler_map.get(query_type, self._handle_generic_query)
        
        handler_args = {'session': session, 'classification': classification, 'user_prompt': user_prompt}
        
        if handler in [self._handle_realtime_alert_query, self._handle_status_inquiry]:
            return handler(classification=handler_args['classification'], session=handler_args['session'])
        elif handler in [self._handle_details_followup_query, self._handle_action_command]:
            return handler(user_prompt=handler_args['user_prompt'], session=handler_args['session'])
        else:
            return handler(user_prompt=handler_args['user_prompt'])

    def _handle_status_inquiry(self, classification: Dict[str, Any], session: Dict[str, Any]) -> Dict[str, Any]:
        """Handles queries asking for the status of an action."""
        logger.info("==> Handling: Status Inquiry")
        sr_no = classification.get("extracted_parameters", {}).get("sr_no")

        if sr_no:
            try:
                status_result = self.response_orchestrator.get_action_status_by_sr_no(int(sr_no))
                return {"message": status_result.get("message", "Could not retrieve status.")}
            except (ValueError, TypeError):
                return {"message": f"'{sr_no}' is not a valid tracking number."}

        alert_id = session.get('selected_alert', {}).get('full_document', {}).get('_id')
        if not alert_id:
            return {"message": "To check a status, please provide an SR_NO or select an alert first."}

        status_result = self.response_orchestrator.get_action_status(alert_id)
        return {"message": status_result.get("message", "Could not retrieve status.")}

    def _handle_direct_command(self, command: str, session: Dict[str, Any]) -> Dict[str, Any]:
        """Handles payloads from interactive buttons."""
        logger.info(f"==> Handling: Direct Command ('{command.split()[0]}')")
        command_type, *args = command.strip().split(' ', 1)
        payload = args[0] if args else ""

        if command_type == 'select_alert':
            response = self.response_orchestrator.handle_alert_selection(payload, session.get('last_query_results', []), session['session_id'])
            if response.get("status") == "success":
                session['selected_alert'] = response.get("selected_alert")
            return response
            
        elif command_type == 'execute_action':
            if not session.get('selected_alert'):
                return {"message": "❌ No alert selected. Please select an alert before choosing an action."}
            
            # --- MODIFICATION ---
            # Instead of returning immediately, we now wait for completion.
            response = self.response_orchestrator.execute_action(payload, session['selected_alert'], session['session_id'])
            
            # Check if the action was successfully submitted to the database
            if response.get("status") == "success" and (sr_no := response.get("data", [{}])[0].get("sr_no")):
                # Action was queued, now block and wait for the final result.
                return self._wait_for_action_completion(sr_no)

            # If submission failed, return the error message immediately.
            return response

        elif command_type == 'show_json':
            return self._show_result_json(payload, session)
            
        return {"message": f"❌ Unrecognized command: {command}"}

    def _show_result_json(self, result_id: str, session: Dict[str, Any]) -> Dict[str, Any]:
        """Formats and returns the full JSON for a specific result."""
        selected_item = next((item for item in session.get('last_query_results', []) if item.get('id') == result_id), None)
        if not selected_item:
            return {"message": f"❌ Could not find details for result '{result_id}'."}
        
        title = selected_item.get('title', result_id)
        full_details = selected_item.get('full_document', {})
        json_str = json.dumps(full_details, indent=2, default=str)
        
        return {"message": f"✅ Full JSON for **{title}**:\n\n```json\n{json_str}\n```"}

    def _handle_action_command(self, user_prompt: str, session: Dict[str, Any]) -> Dict[str, Any]:
        """Initiates a confirmation step for an action described in natural language."""
        logger.info(f"==> Handling: Action Command ('{user_prompt}')")
        session['awaiting_confirmation'] = True
        session['pending_action'] = user_prompt
        
        message = f"You requested to perform the action: **'{user_prompt}'**.\n\nPlease confirm."
        options = [{"title": "Yes, proceed", "payload": "Yes"}, {"title": "No, cancel", "payload": "No"}]
        return {"message": message, "options": options}

    def _handle_action_confirmation(self, user_response: str, session: Dict[str, Any]) -> Dict[str, Any]:
        """Handles the 'Yes' or 'No' response after an action confirmation."""
        logger.info(f"==> Handling: Action Confirmation (User said '{user_response}')")
        pending_action = session.pop('pending_action', None)
        session['awaiting_confirmation'] = False
        user_response_lower = user_response.lower().strip()

        if user_response_lower == 'yes':
            alert_context = session.get('selected_alert', {"id": "manual_action"})
            
            # --- MODIFICATION ---
            # Instead of returning immediately, we now wait for completion.
            response = self.response_orchestrator.execute_action(pending_action, alert_context, session['session_id'])
            
            # Check if the action was successfully submitted to the database
            if response.get("status") == "success" and (sr_no := response.get("data", [{}])[0].get("sr_no")):
                # Action was queued, now block and wait for the final result.
                return self._wait_for_action_completion(sr_no)
            
            # If submission failed, return the error message immediately.
            return response
            
        elif user_response_lower == 'no':
            return {"message": "Action cancelled."}
        else:
            session['awaiting_confirmation'] = True
            session['pending_action'] = pending_action
            options = [{"title": "Yes, proceed", "payload": "Yes"}, {"title": "No, cancel", "payload": "No"}]
            return {"message": "Invalid response. Please confirm with 'Yes' or 'No'.", "options": options}

    def _handle_realtime_alert_query(self, classification: Dict[str, Any], session: Dict[str, Any]) -> Dict[str, Any]:
        """Handles queries that require fetching live data from Elasticsearch."""
        logger.info("==> Handling: Real-time Alert Query")
        response = asyncio.run(self.response_orchestrator.handle_real_time_query(classification, session_id=session['session_id']))
        if response.get("status") == "success":
            session['last_query_results'] = response.get("original_results", [])
            logger.info(f"Stored {len(session['last_query_results'])} results in session context.")
        return response

    def _handle_details_followup_query(self, user_prompt: str, session: Dict[str, Any]) -> Dict[str, Any]:
        """Handles follow-up questions asking for details on a previous result."""
        logger.info(f"==> Handling: Details Follow-up ('{user_prompt}')")
        
        def _extract_index(prompt: str) -> Optional[int]:
            match = re.search(r'(\d+)(?:st|nd|rd|th)?', prompt.lower())
            if match: return int(match.group(1)) - 1
            word_map = {'first': 0, 'second': 1, 'third': 2, 'fourth': 3, 'fifth': 4}
            return next((idx for word, idx in word_map.items() if word in prompt.lower()), None)

        index = _extract_index(user_prompt)
        results = session.get('last_query_results', [])
        if index is not None and 0 <= index < len(results):
            result_id = results[index].get('id')
            return self._show_result_json(result_id, session)
        
        return {"message": "I couldn't determine which item you meant. Please refer to it by number (e.g., 'show me the first one')."}

    def _handle_generic_query(self, user_prompt: str) -> Dict[str, Any]:
        """Handles generic, knowledge-based questions using RAG."""
        logger.info("==> Handling: Generic/RAG Query")
        retrieved_docs = self.retriever.search(user_prompt, self.embedder)
        if not retrieved_docs:
            return {"message": "I couldn't find relevant information in my knowledge base for that query."}
        
        logger.info(f"Retrieved {len(retrieved_docs)} documents for RAG.")
        
        context = "\n\n---\n\n".join([doc.get('text', '') for doc in retrieved_docs])
        prompt = f"Context:\n{context}\n\nUser Question: {user_prompt}\n\nAnswer:"
        
        response_message = self._generate_llm_response(self.system_prompts["generic"], prompt)
        return {"message": response_message}

    def _generate_llm_response(self, system_prompt: str, user_prompt: str) -> str:
        """Sends a request to the Ollama model and returns the response content."""
        try:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            response = self.client.chat(
                model=self.llm_model, 
                messages=messages, 
                options={'temperature': 0.1}
            )
            return response['message']['content'].strip()
        except Exception as e:
            logger.error(f"Error during LLM call: {e}", exc_info=True)
            return "An error occurred while communicating with the AI service."

    def generate_response(self, user_prompt: str, session_id: str) -> str:
        """A direct passthrough to the LLM, used internally by the QueryClassifier."""
        return self._generate_llm_response("You are an expert AI assistant. Follow instructions precisely.", user_prompt)

    def clear_session(self, session_id: str):
        """Removes a session context from memory."""
        if session_id in self.session_contexts:
            del self.session_contexts[session_id]
            logger.info(f"Session context for {session_id} has been cleared.")