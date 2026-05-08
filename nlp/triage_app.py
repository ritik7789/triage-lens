import json
import re
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(title="DeepSeek-R1 Triage Engine")

# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-r1:latest"

# ---------------------------------------------------------
# 2. Data Schemas
# ---------------------------------------------------------
class BugReport(BaseModel):
    ticket_id: str
    subject: str
    description: str
    submitter_role: Optional[str] = "End User"

class TriageResult(BaseModel):
    category: str
    priority: str
    assigned_team: str
    explanation: str
    confidence_score: float

# ---------------------------------------------------------
# 3. DeepSeek-Specific Response Cleaning
# ---------------------------------------------------------
# ---------------------------------------------------------
# 3. DeepSeek-Specific Response Cleaning (UPDATED)
# ---------------------------------------------------------
def extract_json_from_deepseek(raw_text: str) -> dict:
    """Strips <think> tags and markdown to find the pure JSON."""
    if not raw_text:
        return None
        
    # 1. Remove the <think>...</think> block
    clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
    
    # 2. Remove markdown code blocks (```json ... ```) which DeepSeek loves to use
    clean_text = re.sub(r'```json', '', clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r'```', '', clean_text)
    
    # 3. Extract the JSON object
    match = re.search(r'\{.*\}', clean_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as e:
            print(f"[DEBUG] JSON Decode Error: {e}")
            print(f"[DEBUG] Text that failed to parse: {match.group()}")
            return None
    return None

# ---------------------------------------------------------
# 4. Triage Logic (UPDATED WITH DEBUGGING & LONGER TIMEOUT)
# ---------------------------------------------------------
def run_deepseek_triage(report: BugReport):
    prompt = f"""
    Analyze this IT bug report and provide a triage assessment.
    
    TICKET:
    Subject: {report.subject}
    Description: {report.description}
    Role: {report.submitter_role}
    
    Output your final assessment in the following JSON format:
    {{
      "category": "Network",
      "priority": "P2",
      "assigned_team": "Tech Team",
      "explanation": "Reasoning here",
      "confidence_score": 0.9
    }}
    """

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1 # Lowered temperature for stricter JSON adherence
        }
    }

    try:
        # INCREASED TIMEOUT to 120 seconds. DeepSeek needs time to "think" locally!
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=120)
        response.raise_for_status()
        
        raw_output = response.json().get("response", "")
        
        # Print the raw output to your terminal so you can see exactly what the AI said
        print("\n--- RAW OLLAMA OUTPUT ---")
        print(raw_output)
        print("-------------------------\n")
        
        parsed_json = extract_json_from_deepseek(raw_output)
        if parsed_json:
            return parsed_json
        else:
            print("[DEBUG] Failed to extract valid JSON from the raw output.")
            return None
            
    except requests.exceptions.Timeout:
        print("[DEBUG] Error: Connection Timed Out. DeepSeek took longer than 120 seconds to respond.")
        return None
    except requests.exceptions.ConnectionError:
        print("[DEBUG] Error: Connection Refused. Is Ollama running on localhost:11434?")
        return None
    except Exception as e:
        print(f"[DEBUG] DeepSeek Error: {e}")
        return None

# ---------------------------------------------------------
# 5. API Endpoint
# ---------------------------------------------------------
@app.post("/api/v1/triage")
async def process_ticket(report: BugReport):
    result = run_deepseek_triage(report)
    
    if not result:
        raise HTTPException(status_code=500, detail="AI Analysis failed. Check Ollama logs.")
    
    # Map model output to our TriageResult schema
    return TriageResult(**result)

# Run with: uvicorn filename:app --reload