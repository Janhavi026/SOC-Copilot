# elasticsearch_realtime_query.py

import json
import re
import os
import shutil
import configparser
import logging
import pprint 
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, MutableMapping

# --- Setup ---
try:
    # +++ ADDED FOR OLLAMA +++
    import ollama
    # ++++++++++++++++++++++
    from elasticsearch import Elasticsearch
except ImportError:
    # MODIFIED: Updated import guidance
    print("Please install required packages: pip install ollama elasticsearch")
    exit()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Standalone Utility Functions ---
# ... (These functions remain unchanged) ...
def _get_nested_field(document: Dict[str, Any], field_path: str) -> Any:
    """Safely retrieves a value from a nested dictionary."""
    value = document
    try:
        for field in field_path.split('.'):
            if not isinstance(value, dict): return None
            value = value.get(field)
        return value
    except (TypeError, AttributeError):
        return None

def _field_exists(document: Dict[str, Any], field_path: str) -> bool:
    """Checks if a nested field has a non-empty value."""
    return _get_nested_field(document, field_path) is not None

def _flatten_dict_for_table(d: MutableMapping, parent_key: str = '', sep: str = '.') -> List[Dict[str, str]]:
    """
    Flattens a nested dictionary into a list of dictionaries, suitable for a table.
    This version recursively unnests all objects and lists to ensure a clean table view.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        
        if isinstance(v, MutableMapping):
            if v:
                items.extend(_flatten_dict_for_table(v, new_key, sep=sep))
        elif isinstance(v, list):
            if not v:
                items.append({"Field": new_key, "Value": "(empty list)"})
            else:
                for i, item in enumerate(v):
                    indexed_key = f"{new_key}[{i}]"
                    if isinstance(item, MutableMapping):
                        items.extend(_flatten_dict_for_table(item, indexed_key, sep=sep))
                    else:
                        items.append({"Field": indexed_key, "Value": str(item) if item is not None else "N/A"})
        else:
            value_str = str(v) if v is not None else "N/A"
            items.append({"Field": new_key, "Value": value_str})
            
    return items

# --- Data Filtering and Ranking Class ---
class DataRetriever:
    # ... (This class remains unchanged) ...
    """
    Enhanced retriever that understands numerical constraints, time frames,
    and specific field requests to filter and score data.
    """
    def __init__(self):
        """Initializes predefined regex patterns for intent classification."""
        self.patterns = {
            'login_fail': [r'fail.*login', r'login.*fail', r'authentication.*fail', r'invalid.*password', r'wrong.*password', r'access.*denied'],
            'suspicious_activity': [r'suspicious', r'malicious', r'threat', r'attack', r'breach', r'compromise', r'intrusion'],
            'network_issues': [r'connection.*timeout', r'network.*down', r'port.*scan', r'brute.*force', r'ddos'],
            'ipsec_nat': [r'ipsec.*nat.*traversal', r'nat.*traversal.*port', r'ipsec.*port.*activity']
        }
        self.field_patterns = {
            'ip': ['ip', 'address', 'host.ip', 'source.ip'], 'user': ['user', 'username', 'account'],
            'hostname': ['hostname', 'host.name'], 'port': ['port'], 'alert': ['alert', 'rule'], 'time': ['time', 'timestamp']
        }
        self.entity_patterns = {
            'user': ['user', 'username', 'account'], 'ip': ['ip', 'address', 'host'], 'port': ['port', 'service'],
            'alert': ['alert', 'rule', 'detection'], 'network': ['network', 'traffic']
        }

    def _extract_user_intent(self, user_query: str) -> Dict[str, Any]:
        """Extracts structured intent from a natural language query."""
        query_lower = user_query.lower()
        intent = {
            "primary_intent": "general", "secondary_intents": [],
            "time_constraints": [], "entity_focus": [], "exact_matches": [],
            "numerical_constraints": {}, "field_requests": []
        }

        for intent_type, patterns in self.patterns.items():
            if any(re.search(p, query_lower) for p in patterns):
                if intent["primary_intent"] == "general":
                    intent["primary_intent"] = intent_type
                intent["secondary_intents"].append(intent_type)

        num_match = re.search(r'(\d+)\s*(ip|host|user|alert|result|event)s?', query_lower)
        if num_match:
            intent["numerical_constraints"]['count'] = int(num_match.group(1))
            intent["numerical_constraints"]['entity'] = num_match.group(2)

        intent["field_requests"] = [ftype for ftype, kwds in self.field_patterns.items() if any(k in query_lower for k in kwds)]
        intent["entity_focus"] = [etype for etype, kwds in self.entity_patterns.items() if any(k in query_lower for k in kwds)]
        intent["exact_matches"] = [word.lower() for word in re.findall(r'\b([a-zA-Z]{3,})\b', user_query)]

        time_patterns = {
            'today': r'today', 'yesterday': r'yesterday', 'this week': r'this week',
            'last hour': r'last hour|past hour', 'recent': r'recent|latest', 'last 24h': r'last 24|past 24|24 hour'
        }
        intent["time_constraints"] = [desc for desc, pat in time_patterns.items() if re.search(pat, query_lower)]

        logger.info(f"Extracted user intent: {intent}")
        return intent

    def _score_relevance(self, document: Dict[str, Any], intent: Dict[str, Any]) -> float:
        """Calculates a relevance score for a document based on user intent."""
        score = 0.0
        doc_text = json.dumps(document).lower()

        if intent["primary_intent"] != "general" and any(re.search(p, doc_text) for p in self.patterns.get(intent["primary_intent"], [])):
            score += 3.0
        score += sum(1.5 for intent_type in intent["secondary_intents"] if any(re.search(p, doc_text) for p in self.patterns.get(intent_type, [])))
        score += sum(1.0 for word in intent["exact_matches"] if word in doc_text)

        entity_fields = {
            'user': ['user.name', 'username'], 'ip': ['source.ip', 'destination.ip', 'host.ip'],
            'port': ['source.port', 'destination.port'], 'alert': ['kibana.alert.rule.name', 'rule.name'],
            'network': ['network.protocol', 'network.direction']
        }
        for entity in intent["entity_focus"]:
            if any(_field_exists(document, field) for field in entity_fields.get(entity, [])):
                score += 2.0
        return score

    def retrieve_relevant_data(self, raw_data: List[Dict[str, Any]], user_query: str) -> List[Dict[str, Any]]:
        """Filters, scores, and ranks raw data based on the user query."""
        if not raw_data:
            return []
        intent = self._extract_user_intent(user_query)
        
        scored_docs = [(doc, self._score_relevance(doc, intent)) for doc in raw_data]
        scored_docs = [(doc, score) for doc, score in scored_docs if score > 0.5]
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        max_results = intent.get("numerical_constraints", {}).get('count', 100)
        relevant_docs = [doc | {'_relevance_score': score} for doc, score in scored_docs]
        
        logger.info(f"Retriever filtered {len(raw_data)} documents to {len(relevant_docs)} relevant results")
        return relevant_docs[:max_results]


# --- Main Orchestration Class ---
class ElasticsearchQueryHandler:
    def __init__(self, config_path: str = 'config.ini', request_timeout: int = 60):
        self.config = self._load_config(config_path)
        self.es_client = self._setup_elasticsearch_client()
        self.ollama_client, self.llm_model = self._setup_ollama_client()
        self.retriever = DataRetriever()
        self.data_dir = os.path.join("data", "real_data")
        os.makedirs(self.data_dir, exist_ok=True)
        logger.info(f"Data directory set to: {self.data_dir}")
        
    def _load_config(self, config_path: str) -> configparser.ConfigParser:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        config = configparser.ConfigParser()
        config.read(config_path)
        return config

    def _setup_elasticsearch_client(self) -> Optional[Elasticsearch]:
        try:
            es_config = self.config['ELASTICSEARCH']
            client = Elasticsearch(
                hosts=[es_config.get('host')],
                http_auth=(es_config.get('user'), es_config.get('password')),
                timeout=60, verify_certs=False,
            )
            if client.ping():
                logger.info("Elasticsearch client connected successfully.")
                return client
            logger.error("Elasticsearch connection failed.")
        except Exception as e:
            logger.error(f"Failed to initialize Elasticsearch client: {e}", exc_info=True)
        return None
        
    def _setup_ollama_client(self) -> Tuple[Optional[ollama.Client], Optional[str]]:
        try:
            ollama_config = self.config['OLLAMA_SETTINGS']
            base_url = ollama_config.get('ollama_api_base')
            model = ollama_config.get('ollama_model')

            if not base_url or not model:
                logger.error("Ollama 'ollama_api_base' or 'ollama_model' is missing in config.ini.")
                return None, None
            
            host_url = re.match(r'https?://[^/]+', base_url).group(0)
            client = ollama.Client(host=host_url)
            logger.info(f"Ollama client initialized for model '{model}' at host '{host_url}'.")
            return client, model
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}", exc_info=True)
        return None, None
        
    def _build_llm_prompt(self, user_query: str) -> str:
        available_fields = [
            "@timestamp", "kibana.alert.rule.name", "rule.name", "event.action", "event.category",
            "event.outcome", "event.severity", "kibana.alert.severity", "kibana.alert.reason",
            "kibana.alert.rule.description", "kibana.alert.workflow_status", "host.name", "host.hostname",
            "host.ip", "host.mac", "host.os.full", "host.architecture", "user.name", "user.domain", "user.id",
            "source.ip", "source.port", "source.mac", "destination.ip", "destination.port",
            "network.protocol", "network.direction", "process.pid", "process.ppid", "process.name",
            "process.executable", "process.command_line", "ip_list", "logger.ip"
        ]
        return f"""
You are an expert at creating Elasticsearch queries. Based on the user's query, suggest a JSON object
containing a single key, "_source", which should be a list of relevant fields to fetch.
Select fields from this list: {available_fields}
Respond ONLY with the JSON object.

USER QUERY: "{user_query}"
"""

    def _generate_basic_query(self, user_query: str) -> Tuple[str, Dict[str, Any]]:
        default_fields = [
            "@timestamp", "kibana.alert.rule.name", "rule.name", "event.action", "event.category",
            "event.outcome", "event.severity", "kibana.alert.severity", "kibana.alert.reason",
            "kibana.alert.rule.description", "kibana.alert.workflow_status", "host.name", "host.hostname",
            "host.ip", "host.mac", "host.os.full", "host.architecture", "user.name", "user.domain", "user.id",
            "source.ip", "source.port", "source.mac", "destination.ip", "destination.port",
            "network.protocol", "network.direction", "process.pid", "process.ppid", "process.name",
            "process.executable", "process.command_line", "ip_list","logger_ip","kibana.alert.rule.rule_id"
        ]
        source_fields = default_fields

        if self.ollama_client and self.llm_model:
            try:
                prompt = self._build_llm_prompt(user_query)
                
                response = self.ollama_client.chat(
                    model=self.llm_model, 
                    messages=[{"role": "user", "content": prompt}],
                    options={'temperature': 0.0}
                )
                content = response['message']['content']

                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    llm_body = json.loads(json_match.group(0))
                    if isinstance(llm_body.get('_source'), list) and llm_body['_source']:
                        llm_fields = llm_body['_source']
                        combined_fields = set(default_fields + llm_fields)
                        source_fields = list(combined_fields)
                        logger.info(f"Successfully combined {len(llm_fields)} LLM-suggested fields with {len(default_fields)} default fields.")
            except Exception as e:
                logger.error(f"Could not parse LLM response, using default fields. Error: {e}")
        
        query_body = { "size": 200, "sort": [{"@timestamp": {"order": "desc"}}], "_source": source_fields, "query": {"match_all": {}}}
        return ".alerts-security.alerts-default", query_body

    def _execute_raw_data_query(self, index: str, query_body: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.es_client:
            logger.error("Elasticsearch client is not available.")
            return []
        try:
            logger.info(f"Executing query on index '{index}'")
            response = self.es_client.search(index=index, body=query_body)
            return [hit['_source'] | {'_id': hit['_id']} for hit in response['hits']['hits']]
        except Exception as e:
            logger.error(f"Error executing raw data query: {e}", exc_info=True)
            return []

    # ... (format_for_frontend method remains unchanged) ...
    def format_for_frontend(self, query_result: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        if query_result.get("status") == "error":
            return {"response": query_result.get("message", "An error occurred."), "type": "error", **query_result}

        results = query_result.get("data", {}).get("results", [])
        if not results:
            response_text = f"❌ No relevant results found for your query: '{user_query}'"
        else:
            response_lines = [
                f"**Found {len(results)} relevant results for your query:** '{user_query}'",
                "Showing top 10 most relevant results. Please select an option:\n"
            ]
            for i, result in enumerate(results[:10]):
                result_id = result.get('id')
                response_lines.extend([
                    f"**{i+1}. {result.get('title', 'Untitled Result')}**",
                    f"   - **Time**: {result.get('timestamp', 'N/A')}",
                    f"   - **Summary**: {result.get('summary', 'No summary available.')}",
                    f"   - **Actions**: [view_details:{result_id}] Investigate [/view_details] [show_json:{result_id}] Show JSON [/show_json]\n"
                ])
            if len(results) > 10:
                response_lines.append(f" **... and {len(results) - 10} more results available.**")
            response_text = "\n".join(response_lines)
            
        return {
            "response": response_text, "type": "elasticsearch_results", "content": response_text,
            "data": query_result, "session_id": query_result.get("session_id", ""),
            "timestamp": datetime.now().isoformat(), "status": "success"
        }

    def handle_query(self, user_query: str, session_id: str = "") -> Dict[str, Any]:
        logger.info(f"Stage 1: Fetching raw data for query: '{user_query}'")
        index, query_body = self._generate_basic_query(user_query)

        logger.info("--- Elasticsearch Query Body ---")
        pprint.pprint(query_body)
        logger.info("--------------------------------")

        raw_data = self._execute_raw_data_query(index, query_body)

        if not raw_data:
            return {"status": "error", "message": "No data found in Elasticsearch.", "data": {}}

        # +++ CHANGE: Save raw data to a file as requested +++
        try:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"raw_es_data_{session_id}_{timestamp_str}.json"
            filepath = os.path.join(self.data_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(raw_data, f, indent=4)
            logger.info(f"Saved {len(raw_data)} raw documents to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save raw Elasticsearch data: {e}", exc_info=True)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++

        logger.info(f"Fetched {len(raw_data)} raw documents from Elasticsearch.")
        logger.info("--- Sample Raw ES Document ---")
        pprint.pprint(raw_data[0] if raw_data else "None")
        logger.info("------------------------------")

        logger.info(f"Stage 2: Filtering {len(raw_data)} documents using retriever.")
        relevant_data = self.retriever.retrieve_relevant_data(raw_data, user_query)
        
        if relevant_data:
            logger.info(f"Filtered down to {len(relevant_data)} relevant documents.")
            logger.info("--- Sample Relevant Document (after scoring) ---")
            pprint.pprint(relevant_data[0])
            logger.info("--------------------------------------------------")
        else:
            logger.warning("No relevant documents found after filtering and scoring.")
            
        formatted_results = self._format_results_clickable(relevant_data)
        
        return {
            "status": "success", "message": f"Found {len(relevant_data)} relevant results.",
            "user_query": user_query, "session_id": session_id,
            "data": {"results": formatted_results, "original_results": formatted_results, "raw_data_count": len(raw_data)}
        }
    
    # ... (_format_results_clickable, _generate_title, _generate_summary remain unchanged) ...
    def _format_results_clickable(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted = []
        for i, result in enumerate(results):
            full_document_copy = result.copy()
            full_document_copy.pop('_relevance_score', None)
            
            table_data = _flatten_dict_for_table(full_document_copy)
            
            formatted.append({
                "id": f"result_{i+1}",
                "title": self._generate_title(result),
                "timestamp": result.get('@timestamp', 'N/A'),
                "summary": self._generate_summary(result),
                "relevance_score": result.get('_relevance_score', 0),
                "full_document": result,
                "details_table": table_data
            })
        return formatted

    def _generate_title(self, result: Dict[str, Any]) -> str:
        rule_name = (_get_nested_field(result, 'kibana.alert.rule.name') or 
                     _get_nested_field(result, 'rule.name') or 
                     _get_nested_field(result, 'event.action') or 
                     _get_nested_field(result, 'event.category') or 
                     "Alert")
        
        base_title = f" {rule_name}"
        details = []
        user_fields = ['user.name', 'user.username', 'user.id', 'winlog.event_data.TargetUserName']
        for field in user_fields:
            if user_value := _get_nested_field(result, field):
                details.append(f"User: {user_value}")
                break
        
        ips_used = set()

        def process_ip_field(field_path, prefix):
            ips = _get_nested_field(result, field_path)
            if not ips: return
            if isinstance(ips, str): ips = [ips]
            if isinstance(ips, list):
                for ip in ips:
                    if ip and ip not in ips_used:
                        ips_used.add(ip)
                        details.append(f"{prefix}: {ip}")

        process_ip_field('source.ip', "Src")
        process_ip_field('destination.ip', "Dest")
        
        if host_name := (_get_nested_field(result, 'host.hostname') or _get_nested_field(result, 'host.name')):
            details.append(f"Host: {host_name}")
        
        process_ip_field('host.ip', "IP")

        if process := _get_nested_field(result, 'process.name'):
            details.append(f"Process: {process}")
            
        if port := (_get_nested_field(result, 'destination.port') or _get_nested_field(result, 'source.port')):
            details.append(f"Port: {port}")
        
        return f"{base_title} ({', '.join(details)})" if details else base_title

    def _generate_summary(self, result: Dict[str, Any]) -> str:
        parts = []
        if severity := (_get_nested_field(result, 'event.severity') or _get_nested_field(result, 'kibana.alert.severity')):
            try:
                severity_int = int(severity)
                icons = {4: "🔴 Critical", 3: "🟠 High", 2: "🟡 Medium", 1: "🟢 Low"}
                parts.append(f"Severity: {icons.get(severity_int, '⚪ Unknown')}")
            except (ValueError, TypeError):
                parts.append(f"Severity: {severity}")
        
        if outcome := _get_nested_field(result, 'event.outcome'):
            icon = "✅" if outcome.lower() in ['success', 'allow'] else "❌"
            parts.append(f"Outcome: {outcome.capitalize()} {icon}")
        
        if desc := (_get_nested_field(result, 'kibana.alert.rule.description') or _get_nested_field(result, 'rule.description')):
            if len(desc) > 80: desc = desc[:77] + "..."
            parts.append(f"Desc: {desc}")
        
        return " | ".join(parts) if parts else "Click to see details."

if __name__ == "__main__":
    if not os.path.exists('config.ini'):
        print("Creating dummy 'config.ini'. Please update it with your credentials.")
        config_content = """
[ELASTICSEARCH]
host = http://localhost:9200
user = elastic
password = changeme
timeout = 60

[OLLAMA_SETTINGS]
ollama_api_base = http://localhost:11434/v1
ollama_model = llama3

[EMBEDDER_SETTINGS]
embedding_model_name = all-MiniLM-L6-v2

[DATABASE]
host = localhost
user = youruser
password = yourpassword
database = yourdatabase
"""
        with open('config.ini', 'w') as f:
            f.write(config_content)