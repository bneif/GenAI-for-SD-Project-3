import json
import openai
import google.generativeai as genai
import time
import pandas as pd
from tenacity import retry, wait_exponential, stop_after_attempt
import re
from tqdm import tqdm  # Import tqdm
import xlsxwriter

# Load your API keys
OPENAI_API_KEY = [redacted]
GEMINI_API_KEY = [redacted]

# Setup OpenAI and Gemini clients
openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

# Retry logic for API calls
@retry(wait=wait_exponential(min=4, max=30), stop=stop_after_attempt(5))
def call_openai(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o-2024-08-06",  # Update model if needed
        messages=[{"role": "user", "content": prompt}]
    )
    return response

@retry(wait=wait_exponential(min=4, max=30), stop=stop_after_attempt(5))
def call_gemini(prompt):
    response = genai.GenerativeModel("gemini-2.5-flash-preview-04-17").generate_content(contents=prompt)
    return response

def extract_final_code(response_text):
    code_pattern = r"```(.*?)```"
    code_matches = re.findall(code_pattern, response_text, re.DOTALL)
    if code_matches:
        return code_matches[-1]  # Return the last code block found
    return None

def run_batch():
    # Load your responses.json
    with open("responses.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Create an Excel writer to save the results into a single file
    with pd.ExcelWriter("responses_comparison_two.xlsx", engine="xlsxwriter") as writer:
        for task in tqdm(data["tasks"], desc="Processing tasks", ncols=100):
            task_name = task["task"]
            zero_shot_prompt = task["zero_shot"]
            chain_of_thought_prompt = task["chain_of_thought"]
            
            # Call OpenAI for Zero-Shot
            openai_zero_shot_response = call_openai(zero_shot_prompt)
            openai_zero_shot_text = openai_zero_shot_response['choices'][0]['message']['content']
            
            # Call Gemini for Zero-Shot
            gemini_zero_shot_response = call_gemini(zero_shot_prompt)
            gemini_zero_shot_text = gemini_zero_shot_response.text
            
            # Call OpenAI for Chain of Thought
            openai_chain_of_thought_response = call_openai(chain_of_thought_prompt)
            openai_chain_of_thought_text = openai_chain_of_thought_response['choices'][0]['message']['content']
            
            # Call Gemini for Chain of Thought
            gemini_chain_of_thought_response = call_gemini(chain_of_thought_prompt)
            gemini_chain_of_thought_text = gemini_chain_of_thought_response.text
            
            # Extract final code
            openai_zero_shot_code = extract_final_code(openai_zero_shot_text)
            gemini_zero_shot_code = extract_final_code(gemini_zero_shot_text)
            openai_chain_of_thought_code = extract_final_code(openai_chain_of_thought_text)
            gemini_chain_of_thought_code = extract_final_code(gemini_chain_of_thought_text)
            
            # Prepare data for the sheet
            data_for_sheet = {
                "Prompt": [
                    zero_shot_prompt, 
                    chain_of_thought_prompt
                ],
                "OpenAI Entire Output": [
                    openai_zero_shot_text, 
                    openai_chain_of_thought_text
                ],
                "OpenAI Code Output": [
                    openai_zero_shot_code, 
                    openai_chain_of_thought_code
                ],
                "Gemini Entire Output": [
                    gemini_zero_shot_text, 
                    gemini_chain_of_thought_text
                ],
                "Gemini Code Output": [
                    gemini_zero_shot_code, 
                    gemini_chain_of_thought_code
                ]
            }
            
            # Convert to DataFrame
            df = pd.DataFrame(data_for_sheet)
            # Ensure the sheet name is <= 31 characters
            sheet_name = task_name[:31]  # Truncate to the first 31 characters if it's too long

            # Write the DataFrame to Excel
            df.to_excel(writer, sheet_name=sheet_name, index=False)

# Run the batch processing
run_batch()
