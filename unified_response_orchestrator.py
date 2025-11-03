# unified_response_orchestrator.py

import logging
import json
import os
import re
import configparser
import pprint
from datetime import datetime
from typing import Dict, Any, List, Optional

try:
    import mysql.connector
    from mysql.connector import Error
except ImportError:
    mysql = None
    Error = None
    print("WARNING: 'mysql-connector-python' is not installed. Database logging will be disabled.")

class UnifiedResponseOrchestrator:
    def __init__(self, elastic_query_handler):
        self.elastic_query_handler = elastic_query_handler
        self.logger = logging.getLogger(__name__)
        self.action_log_dir = "data/action_logs"
        self.action_kb_dir = "data/action"
        os.makedirs(self.action_log_dir, exist_ok=True)
        os.makedirs(self.action_kb_dir, exist_ok=True)
        self.db_config = self._load_db_config()
        self.session_context = {}
        
        if self.db_config:
            self.logger.info("Database configuration loaded successfully.")
        else:
            self.logger.warning("Database configuration not found or mysql-connector not installed. DB logging is disabled.")

    def _load_db_config(self) -> Optional[Dict[str, Any]]:
        if not mysql: return None
        try:
            config = configparser.ConfigParser()
            config_path = 'config.ini'
            if not os.path.exists(config_path): return None
            config.read(config_path)
            if 'DATABASE' in config: return dict(config['DATABASE'])
        except Exception as e:
            self.logger.error(f"Could not load database configuration: {e}")
        return None

    def _get_db_connection(self):
        if not self.db_config: return None
        try:
            conn = mysql.connector.connect(**self.db_config)
            if conn.is_connected(): return conn
        except Error as e:
            self.logger.error(f"Error connecting to MySQL database: {e}")
        return None

    def _is_action_already_completed(self, alert_index_id: str) -> bool:
        """Checks if a 'success' status action already exists for the given alert ID."""
        if not alert_index_id:
            return False
            
        conn = self._get_db_connection()
        if not conn:
            self.logger.warning("Cannot check for completed actions; DB connection failed. Allowing action to proceed.")
            return False

        cursor = None
        try:
            cursor = conn.cursor()
            # +++ FIXED: Changed 'Completed' to 'success' and made it case-insensitive +++
            sql = "SELECT 1 FROM currentrequeststatus WHERE `INDEX_ID` = %s AND LOWER(`STATUS`) = 'success' LIMIT 1"
            cursor.execute(sql, (alert_index_id,))
            result = cursor.fetchone()
            
            if result:
                self.logger.info(f"Found an existing successful action for alert ID {alert_index_id}. Blocking new action.")
                return True
            return False
        except Error as e:
            self.logger.error(f"DB error when checking for successful action on {alert_index_id}: {e}")
            return False
        finally:
            if cursor: cursor.close()
            if conn and conn.is_connected(): conn.close()

    def _get_nested_field(self, document: Dict[str, Any], field_path: str) -> Any:
        value = document
        try:
            for field in field_path.split('.'):
                if not isinstance(value, dict): return None
                value = value.get(field)
            return value
        except (TypeError, AttributeError):
            return None

    def _extract_severity(self, alert_details: Dict[str, Any]) -> str:
        severity_val = alert_details.get('kibana.alert.severity') or alert_details.get('event.severity')

        if severity_val is not None:
            sev_str = str(severity_val).lower()
            if sev_str in ["critical", "high", "4", "3"]:
                return "high"
            if sev_str in ["medium", "warn", "warning", "2"]:
                return "medium"
            if sev_str in ["low", "informational", "notice", "1"]:
                return "low"
        
        return "unknown"
    
    def _create_soar_request_json(self, action_name: str, details: Dict[str, Any]) -> str:
        suspicious_value = ""
        action_lower = action_name.lower()
        
        if "block ip" in action_lower:
            suspicious_value = self._get_nested_field(details, 'source.ip') or ""
        elif "enable firewall" in action_lower:
            suspicious_value = details.get('logger_ip') or ""
        elif "block account" in action_lower:
            suspicious_value = self._get_nested_field(details, 'host.ip') or ""
        
        timestamp_str = "N/A"
        raw_timestamp = details.get('@timestamp')
        if raw_timestamp:
            try:
                dt_obj = datetime.fromisoformat(raw_timestamp.replace('Z', '+00:00'))
                timestamp_str = dt_obj.strftime('%d-%m-%Y %H:%M:%S')
            except (ValueError, TypeError):
                timestamp_str = raw_timestamp

        payload = {
            "severity": self._extract_severity(details),
            "ipaddress": self._get_nested_field(details, 'source.ip') or "N/A",
            "actionType": "Manual",
            "password": "",
            "action": action_name,
            "id": details.get('_id') or "N/A",
            "suspicious": suspicious_value,
            "typerules": details.get('kibana.alert.rule.name') or "N/A",
            "timestamp": timestamp_str,
        }
        
        return json.dumps([payload], default=str)

    def execute_action(self, action_name: str, alert_data: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        try:
            if alert_data is None:
                alert_data = self.session_context.get(session_id, {}).get('selected_alert')
            if alert_data is None:
                return {"status": "error", "message": "❌ No alert data available. Please select an alert first."}
            
            details = alert_data.get('full_document', {})
            if not details:
                return {"status": "error", "message": "❌ Could not find the details of the selected alert."}

            alert_index_id = details.get('_id')

            if self._is_action_already_completed(alert_index_id):
                return {
                    "status": "error",
                    "message": "❌ An action has already been successfully completed for this alert. No further actions are permitted."
                }

            event_timestamp_str = None
            raw_timestamp = details.get('@timestamp')
            if raw_timestamp:
                try:
                    dt_obj = datetime.fromisoformat(raw_timestamp.replace('Z', '+00:00'))
                    event_timestamp_str = dt_obj.strftime('%d-%m-%Y %H:%M:%S')
                except (ValueError, TypeError):
                    event_timestamp_str = raw_timestamp

            suspicious_db_value = None
            action_lower = action_name.lower()
            if "block ip" in action_lower:
                suspicious_db_value = self._get_nested_field(details, 'source.ip')
            elif "enable firewall" in action_lower or "disable firewall" in action_lower:
                suspicious_db_value = details.get('logger_ip')
            elif "block account" in action_lower:
                suspicious_db_value = self._get_nested_field(details, 'host.ip')

            request_json_payload = self._create_soar_request_json(action_name, details)
            
            db_record = {
                "SR_NO": None, "ACTION": action_name, "ACTIONTYPE": "Automation",
                "DATE": event_timestamp_str, "INDEX_ID": details.get('_id'),
                "IPADDRESS": details.get('logger_ip'), "PASSWORD": None, "REQUESTID": None,
                "REQUESTJSON": request_json_payload, "REQUEST_COUNT": 1, "REQUEST_TYPE": None,
                "STATUS": "Inprocess", "SUSPICIOUS": suspicious_db_value,
                "TYPE_RULES": details.get('kibana.alert.rule.name'), "BRANCH_NAME": "velox",
                "EVENTID": None, "CLOSURE_TIME": None, "RESPONSE_TIME": None,
                "SEVERITY": self._extract_severity(details), "PLAYBOOKNAME": None,
                "ACTION_TAKEN_BY": None, "VT_Response": None,
            }
            
            file_payload = {"action_id": f"action_{datetime.now().strftime('%Y%m%d_%H%M%S')}", "db_record": db_record, "original_alert_details": details}
            sr_no = self._log_action(file_payload, db_record)
            
            if sr_no:
                message = ""
            else:
                message = f"✅ Action '{action_name}' has been initiated and logged, but a tracking number could not be generated."

            return {"status": "success", "message": message, "options": [], "data": [{"sr_no": sr_no, "status": "inprocess"}]}
        except Exception as e:
            self.logger.error(f"Error executing action: {e}", exc_info=True)
            return {"status": "error", "message": f"❌ Failed to execute action: {str(e)}"}

    def _log_action(self, file_payload: Dict[str, Any], db_record: Dict[str, Any]) -> Optional[int]:
        try:
            filename = f"{file_payload['action_id']}.json"
            filepath = os.path.join(self.action_log_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(file_payload, f, indent=4, default=str)
            self.logger.info(f"Action log saved to file: {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving action log to file: {e}")
        return self._insert_log_to_db(db_record)

    def _insert_log_to_db(self, db_record: Dict[str, Any]) -> Optional[int]:
        conn = self._get_db_connection()
        if not conn:
            return None
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(SR_NO) FROM currentrequeststatus")
            max_id = cursor.fetchone()[0]
            next_sr_no = (max_id or 0) + 1
            
            sql = """
                INSERT INTO currentrequeststatus (
                    `SR_NO`, `ACTION`, `ACTIONTYPE`, `DATE`, `INDEX_ID`, `IPADDRESS`, `PASSWORD`, `REQUESTID`,
                    `REQUESTJSON`, `REQUEST_COUNT`, `REQUEST_TYPE`, `STATUS`, `SUSPICIOUS`, `TYPE_RULES`,
                    `BRANCH_NAME`, `EVENTID`, `CLOSURE_TIME`, `RESPONSE_TIME`, `SEVERITY`, `PLAYBOOKNAME`,
                    `ACTION_TAKEN_BY`, `VT_Response`
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            values = (
                next_sr_no, db_record['ACTION'], db_record['ACTIONTYPE'], db_record['DATE'],
                db_record['INDEX_ID'], db_record['IPADDRESS'], db_record['PASSWORD'], db_record['REQUESTID'],
                db_record['REQUESTJSON'], db_record['REQUEST_COUNT'], db_record['REQUEST_TYPE'],
                db_record['STATUS'], db_record['SUSPICIOUS'], db_record['TYPE_RULES'],
                db_record['BRANCH_NAME'], db_record['EVENTID'], db_record['CLOSURE_TIME'],
                db_record['RESPONSE_TIME'], db_record['SEVERITY'], db_record['PLAYBOOKNAME'],
                db_record['ACTION_TAKEN_BY'], db_record['VT_Response']
            )
            cursor.execute(sql, values)
            conn.commit()
            self.logger.info(f"Action log successfully inserted. SR_NO: {next_sr_no}")
            return next_sr_no
        except Error as e:
            self.logger.error(f"Failed to insert record into table: {e}")
            return None
        finally:
            if cursor: cursor.close()
            if conn and conn.is_connected(): conn.close()
    
    def get_action_status_by_sr_no(self, sr_no: int) -> Dict[str, Any]:
        conn = self._get_db_connection()
        if not conn:
            return {"status": "error", "message": "Could not connect to the database.", "db_status": "error"}
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT `STATUS`, `ACTION`, `DATE` FROM currentrequeststatus WHERE `SR_NO` = %s", (sr_no,))
            result = cursor.fetchone()
            if result:
                return {"status": "success", "message": f"📊 Status for SR_NO **{sr_no}** (Action: '{result['ACTION']}') is: **{result['STATUS'].upper()}**.", "db_status": result['STATUS']}
            else:
                return {"status": "not_found", "message": f"ℹ️ No action found with SR_NO: {sr_no}.", "db_status": "not_found"}
        except Error as e:
            self.logger.error(f"Failed to fetch status for SR_NO {sr_no}: {e}")
            return {"status": "error", "message": "A database error occurred.", "db_status": "error"}
        finally:
            if cursor: cursor.close()
            if conn and conn.is_connected(): conn.close()
            
    def get_action_status(self, alert_index_id: str) -> Dict[str, Any]:
        if not alert_index_id:
            return {"status": "error", "message": "No alert ID was provided."}
        conn = self._get_db_connection()
        if not conn:
            return {"status": "error", "message": "Could not connect to the database."}
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT `STATUS`, `ACTION`, `DATE` FROM currentrequeststatus WHERE `INDEX_ID` = %s ORDER BY `SR_NO` DESC LIMIT 1", (alert_index_id,))
            result = cursor.fetchone()
            if result:
                return {"status": "success", "message": f"📊 The status for action **'{result['ACTION']}'** on {result['DATE']} is: **{result['STATUS'].upper()}**."}
            else:
                return {"status": "not_found", "message": "ℹ️ I could not find any logged action for this specific alert."}
        except Error as e:
            self.logger.error(f"Failed to fetch action status from DB: {e}")
            return {"status": "error", "message": f"A database error occurred: {e}"}
        finally:
            if cursor: cursor.close()
            if conn and conn.is_connected(): conn.close()

    async def handle_real_time_query(self, classification_result: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        try:
            user_query = classification_result.get("user_query", "")
            query_result = self.elastic_query_handler.handle_query(user_query, session_id)
            if query_result.get("status") == "error":
                return { "status": "error", "message": query_result.get("message", "Query failed") }
            
            results = query_result.get("data", {}).get("results", [])
            formatted_response = self._generate_legacy_response_with_options(results, user_query)
            if session_id not in self.session_context: 
                self.session_context[session_id] = {}
            
            self.session_context[session_id]['last_results'] = formatted_response.get("original_results", [])
            return {"status": "success", "message": formatted_response.get("message"), "options": formatted_response.get("options"), "data": formatted_response.get("data"), "original_results": formatted_response.get("original_results")}
        except Exception as e:
            self.logger.error(f"Error handling real-time query: {e}", exc_info=True)
            return { "status": "error", "message": f"Failed to process real-time query: {str(e)}" }

    def _generate_legacy_response_with_options(self, results: List[Dict[str, Any]], user_query: str) -> Dict[str, Any]:
        if not results:
            return {"message": "❌ No alerts found matching your criteria.", "options": [], "data": [], "original_results": []}
        
        options, data_payload, valid_results = [], [], []
        for i, result in enumerate(results[:10]):
            alert_id = self._get_nested_field(result, 'full_document._id')
            if alert_id:
                title = result.get('title', f'Alert {i+1}')
                payload = f"select_alert {alert_id}"
                options.append({"title": title, "payload": payload})
                data_payload.append({"title": title, "payload": payload, "original_result": result})
                valid_results.append(result)
        
        if not options:
            return {"message": "❌ Found alerts but none have valid IDs for selection.", "options": [], "data": [], "original_results": []}
        
        message = (f"**Found {len(results)} alerts** for: '{user_query}'\n\n"
                   f"**Please select an alert to view available actions:**\n"
                   f"Showing top {len(options)} most relevant results")
        return {"message": message, "options": options, "data": data_payload, "original_results": valid_results}

    def handle_alert_selection(self, alert_id: str, original_results: List[Dict[str, Any]], session_id: str) -> Dict[str, Any]:
        selected_alert = next((r for r in original_results if self._get_nested_field(r, 'full_document._id') == alert_id), None)
        
        if not selected_alert and session_id in self.session_context:
            session_results = self.session_context[session_id].get('last_results', [])
            selected_alert = next((r for r in session_results if self._get_nested_field(r, 'full_document._id') == alert_id), None)
        
        if not selected_alert:
            return {"status": "error", "message": "❌ Alert not found. Please search again."}
        
        if session_id not in self.session_context: self.session_context[session_id] = {}
        self.session_context[session_id]['selected_alert'] = selected_alert
        
        alert_details = selected_alert.get('full_document', {})
        action_options = self._get_actions_for_rule(alert_details.get('kibana.alert.rule.name') or 'security alert', alert_details)
        message = f"🛡️ **Selected Alert:** {selected_alert.get('title', 'N/A')}\n\n**Available Actions:**"
        return {"status": "success", "message": message, "options": action_options, "original_results": [selected_alert], "selected_alert": selected_alert}

    def handle_action_confirmation(self, action_name: str, session_id: str) -> Dict[str, Any]:
        selected_alert = self.session_context.get(session_id, {}).get('selected_alert')
        if not selected_alert:
            return {"status": "error", "message": "❌ No alert selected. Please select an alert first."}
        return self.execute_action(action_name, selected_alert, session_id)

    def _get_actions_for_rule(self, rule_name: str, alert_details: Dict[str, Any]) -> List[Dict[str, Any]]:
        base_actions = []
        try:
            for filename in os.listdir(self.action_kb_dir):
                if filename.endswith('.json'):
                    with open(os.path.join(self.action_kb_dir, filename), 'r', encoding='utf-8') as f: data = json.load(f)
                    for rule_data in (data if isinstance(data, list) else [data]):
                        if (json_rule_name := rule_data.get("kibana.alert.rule.name", "").strip().lower()) and json_rule_name in rule_name.strip().lower():
                            base_actions = rule_data.get("action", [])
                            break
                    if base_actions: break
        except Exception as e:
            self.logger.error(f"Error reading action knowledge base: {e}", exc_info=True)
        
        if not base_actions:
            base_actions = ["Investigate Alert", "Notify Security Team", "Block IP","Enable Firewall", "Disable Firewall", "Block Account"," Unblock Account"]
        
        return [{"title": f"⚡ {name.strip()}", "payload": f"execute_action {name.strip()}"} for name in base_actions if name]

    def clear_session(self, session_id: str):
        if session_id in self.session_context:
            del self.session_context[session_id]