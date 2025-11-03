# query_classifier.py
import re
import json
import logging
from typing import Dict, Any
from pathlib import Path
from datetime import datetime

# +++ ADDED FOR OLLAMA +++
# Import for type hinting the client object
import ollama
# ++++++++++++++++++++++

# --- COMMENTED OUT FOR OLLAMA ---
# The OpenAI client is no longer needed in this file.
# from openai import OpenAI
# ------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s")

class QueryClassifier:
    # --- MODIFIED FOR OLLAMA ---
    # The __init__ signature is updated to accept the ollama_client and model
    # instead of OpenAI credentials.
    def __init__(self, ollama_chatbot: Any, data_directory: str, ollama_client: ollama.Client, ollama_model: str):
        # The 'ollama_chatbot' is the DynamicChatbot instance, used for callbacks.
        self.ollama_chatbot = ollama_chatbot 
        self.data_directory = Path(data_directory)
        self.classification_prompt = self._create_classification_prompt()

        # The ollama_client and ollama_model are passed but not directly used in this class,
        # as the actual LLM call is delegated back to the ollama_chatbot instance.
        # This simplifies the design and avoids managing multiple clients.
        self.llm_client = ollama_client
        self.llm_model = ollama_model
        
        # --- REMOVED FOR OLLAMA ---
        # self.client = OpenAI(api_key=openai_api_key)
        # self.openai_model = openai_model
        # --------------------------
    # ---------------------------

    def _create_classification_prompt(self) -> str:
        return """You are an expert cybersecurity query classifier. Your task is to classify user queries and extract parameters.

Respond in this exact JSON format:
{{"classification": "YOUR_CLASSIFICATION", "reasoning": "Your brief explanation", "parameters": {{"sr_no": "EXTRACTED_SR_NO_IF_ANY"}}}}

CLASSIFICATION RULES:
- `GENERIC`: General knowledge questions, definitions, best practices.
- `REAL_TIME_ALERT`: Queries for live or recent data, alerts, logs, or current status.
- `FOLLOW_UP`: Questions that refer to the previous response.
- `ACTION_COMMAND`: A direct instruction to perform a task.
- `STATUS_INQUIRY`: A question about the status of a previously requested action. Keywords: "status", "update", "did it work", "progress".

--- EXAMPLES ---

Query: "What is a brute force attack?"
{{"classification": "GENERIC", "reasoning": "This is a request for a definition.", "parameters": {{"sr_no": null}}}}

Query: "Show me the latest failed login alerts"
{{"classification": "REAL_TIME_ALERT", "reasoning": "This is a request for live alert data.", "parameters": {{"sr_no": null}}}}

Query: "block ip 192.168.1.100"
{{"classification": "ACTION_COMMAND", "reasoning": "This is a direct command to perform a network blocking action.", "parameters": {{"sr_no": null}}}}

Query: "check the progress on the IP block"
{{"classification": "STATUS_INQUIRY", "reasoning": "This is a request for an update on a previously initiated action.", "parameters": {{"sr_no": null}}}}

Query: "what is the status of srno 123"
{{"classification": "STATUS_INQUIRY", "reasoning": "User is asking for the status of a specific request using its SR_NO.", "parameters": {{"sr_no": "123"}}}}
--- END EXAMPLES ---

Now, classify the following user query. Respond ONLY with the JSON object.

User Query: "{query}"
"""

    def _llm_classify_query(self, user_query: str) -> Dict[str, Any]:
        prompt = self.classification_prompt.format(query=user_query)
        try:
            # This call already works because self.ollama_chatbot (DynamicChatbot)
            # has been updated to use the Ollama client in its generate_response method.
            response_text = self.ollama_chatbot.generate_response(prompt, "classification_session")
            
            json_response_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_response_match:
                json_string = json_response_match.group(0)
                parsed_json = json.loads(json_string)
                
                if "classification" in parsed_json:
                    if "parameters" not in parsed_json:
                        parsed_json["parameters"] = {}
                    return parsed_json
            
            logging.warning(f"Could not parse valid JSON from LLM classification response: {response_text}")
            return {"classification": "GENERIC", "parameters": {}, "reasoning": "Fallback due to parsing error."}

        except Exception as e:
            logging.error(f"LLM classification failed: {e}", exc_info=True)
            return {"classification": "GENERIC", "parameters": {}, "reasoning": "Fallback due to exception."}

    def _extract_parameters_from_query(self, user_query: str) -> Dict[str, Any]:
        params = {"filters": {}}
        time_match = re.search(r'last\s+(\d+)\s+(hour|day|minute)s?', user_query, re.IGNORECASE)
        if time_match:
            value, unit = time_match.groups()
            params["time_range"] = {"gte": f"now-{value}{unit[0]}", "lte": "now"}
        else:
            params["time_range"] = {"gte": "now-24h", "lte": "now"}

        ip_match = re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', user_query)
        if ip_match:
            params["filters"]["source.ip"] = ip_match.group(1)

        def extract_entity(keyword: str, query: str) -> str | None:
            pattern = rf"{keyword}\s*[:=\s]?\s*['\"]?([a-zA-Z0-9_.-]+)['\"]?"
            match = re.search(pattern, query, re.IGNORECASE)
            return match.group(1) if match else None

        username = extract_entity("user(?:name)?", user_query)
        if username:
            params["filters"]["user.name"] = username
        
        hostname = extract_entity("host(?:name)?", user_query)
        if hostname:
            params["filters"]["host.name"] = hostname
        
        return params

    def comprehensive_classify(self, user_query: str, session_id: str) -> Dict[str, Any]:
        ### --- ADD THIS RULE-BASED PRE-CHECK --- ###
        # Use regex to reliably detect SR_NO status queries before calling the LLM
        sr_no_match = re.search(r'\b(status\s(of\s)?|check|update\son)\s?(sr_no|srno|request)\s?(\d+)', user_query, re.IGNORECASE)
        if sr_no_match:
            sr_no = sr_no_match.group(4)
            logging.info(f"Rule-based classifier matched SR_NO: {sr_no}")
            return {
                "user_query": user_query,
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "primary_classification": {
                    "query_type": "status_inquiry",
                    "reasoning": "Rule-based match for SR_NO.",
                    "confidence": 1.0,
                    "method": "rule_based"
                },
                "extracted_parameters": {"sr_no": sr_no}
            }
        ### --- END OF ADDED BLOCK --- ###

        llm_classification = self._llm_classify_query(user_query)
        
        query_type_mapping = {
            "GENERIC": "generic",
            "REAL_TIME_ALERT": "real_time_alert",
            "FOLLOW_UP": "follow_up",
            "ACTION_COMMAND": "action_command",
            "STATUS_INQUIRY": "status_inquiry"
        }
        
        query_type = query_type_mapping.get(llm_classification.get("classification"), "generic")

        result = {
            "user_query": user_query,
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "primary_classification": {
                "query_type": query_type,
                "reasoning": llm_classification.get("reasoning"),
                "confidence": 0.95,
                "method": "llm_classification"
            },
        }

        if "parameters" in llm_classification:
            result["extracted_parameters"] = llm_classification["parameters"]
            
        if query_type == 'real_time_alert':
            result["extracted_parameters"] = self._extract_parameters_from_query(user_query)
        
        return result