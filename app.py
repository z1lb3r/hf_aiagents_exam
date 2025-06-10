import os
import gradio as gr
import requests
import pandas as pd
import re
from smolagents import CodeAgent, InferenceClientModel, tool, WebSearchTool

# Constants
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# Custom tool for answer formatting
@tool
def format_final_answer(raw_answer: str, answer_type: str = "auto") -> str:
    """
    Formats the answer according to GAIA requirements.
    
    Args:
        raw_answer: The raw answer from your analysis
        answer_type: Type hint - "number", "string", "list" or "auto" for automatic detection
    
    Returns:
        Properly formatted answer string
    """
    answer = str(raw_answer).strip()
    
    # Remove common prefixes that might interfere
    prefixes_to_remove = [
        "final answer:", "answer:", "the answer is:", "result:", 
        "solution:", "conclusion:", "final_answer:", "FINAL ANSWER:"
    ]
    
    for prefix in prefixes_to_remove:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()
    
    # Remove quotes if present
    if (answer.startswith('"') and answer.endswith('"')) or (answer.startswith("'") and answer.endswith("'")):
        answer = answer[1:-1]
    
    # Auto-detect type if not specified
    if answer_type == "auto":
        # Check if it's a number
        try:
            float(answer.replace(',', ''))
            answer_type = "number"
        except ValueError:
            # Check if it's a list (contains commas)
            if ',' in answer:
                answer_type = "list"
            else:
                answer_type = "string"
    
    # Format based on type
    if answer_type == "number":
        # Remove commas and units for numbers
        answer = re.sub(r'[,$%]', '', answer)
        try:
            # Try to convert to number to validate
            num = float(answer)
            # Return as integer if it's a whole number
            if num.is_integer():
                return str(int(num))
            return str(num)
        except ValueError:
            return answer
    
    elif answer_type == "list":
        # Split by comma and clean each element
        items = [item.strip() for item in answer.split(',')]
        cleaned_items = []
        for item in items:
            # Remove articles and abbreviations
            item = re.sub(r'\b(a|an|the)\s+', '', item, flags=re.IGNORECASE)
            cleaned_items.append(item)
        return ', '.join(cleaned_items)
    
    else:  # string
        # Remove articles and abbreviations
        answer = re.sub(r'\b(a|an|the)\s+', '', answer, flags=re.IGNORECASE)
        # Write numbers in plain text if asked for string
        return answer
    
    return answer

class GAIAAgent:
    def __init__(self):
        """Initialize the GAIA Agent with smolagents CodeAgent"""
        print("Initializing GAIA Agent with smolagents...")
        
        # Initialize the model with token if available
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            print("Using PRO model with token")
            self.model = InferenceClientModel(model_id="meta-llama/Llama-3.3-70B-Instruct", token=hf_token)
        else:
            print("No token, using default")
            self.model = InferenceClientModel()
        # Initialize search tool
        search_tool = WebSearchTool()
        
        # Create the agent with tools
        self.agent = CodeAgent(
            tools=[search_tool, format_final_answer],
            model=self.model,
            add_base_tools=True,  # Adds Python interpreter and other base tools
            additional_authorized_imports=['requests', 'json', 'csv', 'math', 'statistics', 'datetime', 'urllib.parse']
        )
        
        # Customize the system prompt for GAIA format
        self.agent.system_prompt = """You are a specialized AI agent designed to solve GAIA benchmark questions with maximum accuracy.

CRITICAL: Your final answer must be precisely formatted according to GAIA requirements:
- For numbers: No commas, no units (like $ or %), no decimals unless necessary
- For strings: No articles (a, an, the), no abbreviations, plain text for digits
- For lists: Comma-separated, each item following above rules

Process:
1. Understand the question thoroughly
2. Use available tools to gather information and perform calculations
3. Analyze and synthesize the information
4. Format your final answer using the format_final_answer tool
5. Double-check your answer format

Remember: GAIA uses exact match evaluation, so precision in formatting is crucial for scoring."""

    def __call__(self, question: str) -> str:
        """
        Process a question and return the formatted answer
        
        Args:
            question: The GAIA question to solve
            
        Returns:
            Formatted answer string
        """
        try:
            print(f"Processing question: {question[:100]}...")
            
            # Enhanced prompt with clear instructions
            enhanced_question = f"""
GAIA Question: {question}

Please solve this step-by-step:
1. Analyze what information you need
2. Use tools to gather data or perform calculations
3. Synthesize your findings
4. Use the format_final_answer tool to properly format your response

Remember: The answer must be exact and properly formatted for GAIA evaluation.
"""
            
            # Run the agent
            result = self.agent.run(enhanced_question)
            
            # Extract the final answer if it's wrapped in additional text
            if isinstance(result, str):
                # Look for the last occurrence of a formatted answer
                lines = result.split('\n')
                for line in reversed(lines):
                    line = line.strip()
                    if line and not line.startswith('Step') and not line.startswith('Duration'):
                        result = line
                        break
            
            print(f"Agent result: {result}")
            return str(result)
            
        except Exception as e:
            print(f"Error in GAIAAgent: {e}")
            return f"Error processing question: {str(e)}"

def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the GAIA Agent on them, submits all answers,
    and displays the results.
    """
    # Determine HF Space Runtime URL and Repo URL
    space_id = os.getenv("SPACE_ID")
    
    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None
    
    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"
    
    # 1. Instantiate Agent
    try:
        agent = GAIAAgent()
        print("GAIA Agent initialized successfully")
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    
    # Agent code link
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(f"Agent code URL: {agent_code}")
    
    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=30)
        response.raise_for_status()
        questions_data = response.json()
        
        if not questions_data:
            print("Fetched questions list is empty.")
            return "Fetched questions list is empty or invalid format.", None
            
        print(f"Fetched {len(questions_data)} questions.")
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None
    
    # 3. Run GAIA Agent
    results_log = []
    answers_payload = []
    print(f"Running GAIA agent on {len(questions_data)} questions...")
    
    for i, item in enumerate(questions_data, 1):
        task_id = item.get("task_id")
        question_text = item.get("question")
        
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        
        print(f"Processing question {i}/{len(questions_data)}: {task_id}")
        
        try:
            submitted_answer = agent(question_text)
            answers_payload.append({
                "task_id": task_id, 
                "submitted_answer": submitted_answer
            })
            results_log.append({
                "Task ID": task_id, 
                "Question": question_text[:100] + "..." if len(question_text) > 100 else question_text,
                "Submitted Answer": submitted_answer
            })
            print(f"Answer for {task_id}: {submitted_answer}")
            
        except Exception as e:
            error_msg = f"AGENT ERROR: {e}"
            print(f"Error running agent on task {task_id}: {e}")
            answers_payload.append({
                "task_id": task_id, 
                "submitted_answer": error_msg
            })
            results_log.append({
                "Task ID": task_id, 
                "Question": question_text[:100] + "..." if len(question_text) > 100 else question_text,
                "Submitted Answer": error_msg
            })
    
    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)
    
    # 4. Prepare Submission
    submission_data = {
        "username": username.strip(), 
        "agent_code": agent_code, 
        "answers": answers_payload
    }
    
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)
    
    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=120)
        response.raise_for_status()
        result_data = response.json()
        
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        
        print("Submission successful!")
        print(final_status)
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
        
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except:
            error_detail += f" Response: {e.response.text[:500]}"
        
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
        
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df

# Build Gradio Interface
with gr.Blocks(title="GAIA Agent Evaluation") as demo:
    gr.Markdown("# GAIA Agent Evaluation Runner")
    gr.Markdown("""
    **Enhanced GAIA Agent using smolagents**
    
    This agent uses:
    - CodeAgent from smolagents library
    - Llama-3.3-70B-Instruct model for reasoning
    - Web search and Python execution capabilities
    - Specialized answer formatting for GAIA benchmark
    
    **Instructions:**
    1. Clone this space to your profile
    2. Log in with your Hugging Face account
    3. Click 'Run Evaluation & Submit All Answers'
    4. Wait for the agent to process all questions
    """)
    
    gr.LoginButton()
    
    run_button = gr.Button("Run Evaluation & Submit All Answers", variant="primary")
    
    status_output = gr.Textbox(
        label="Run Status / Submission Result", 
        lines=8, 
        interactive=False
    )
    
    results_table = gr.DataFrame(
        label="Questions and Agent Answers", 
        wrap=True,
        interactive=False
    )
    
    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*50 + " GAIA Agent Starting " + "-"*50)
    
    # Check environment variables
    space_host = os.getenv("SPACE_HOST")
    space_id = os.getenv("SPACE_ID")
    
    if space_host:
        print(f"✅ SPACE_HOST found: {space_host}")
        print(f"   Runtime URL: https://{space_host}.hf.space")
    else:
        print("ℹ️ SPACE_HOST not found (running locally?)")
    
    if space_id:
        print(f"✅ SPACE_ID found: {space_id}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id}")
    else:
        print("ℹ️ SPACE_ID not found (running locally?)")
    
    print("-" * 120 + "\n")
    print("Launching GAIA Agent Evaluation Interface...")
    demo.launch(debug=True, share=False)