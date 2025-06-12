import os
import gradio as gr
import requests
import pandas as pd
import re # <-- NEW: Добавлен импорт для регулярных выражений в format_for_gaia
from smolagents import CodeAgent, InferenceClientModel, WebSearchTool, tool # <-- MODIFIED: Добавлен импорт 'tool' для декоратора

# Constants
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- Custom Tools ---
# <-- NEW SECTION START: Определение инструмента format_for_gaia -->
@tool
def format_for_gaia(raw_answer: str, answer_type: str) -> str:
    """
    Форматирует окончательный ответ в соответствии с требованиями GAIA. Этот инструмент
    критически важен для правильной оценки. Агент ДОЛЖЕН определить необходимый тип ответа
    ('number', 'string', или 'list') из контекста вопроса.
    
    Args:
        raw_answer (str): Необработанный ответ, найденный агентом.
        answer_type (str): Тип ответа. Должен быть одним из "number", "string", или "list".
    """
    answer = str(raw_answer).strip()

    # Удаляем распространенные префиксы, которые может добавить модель
    prefixes_to_remove = [
        "final answer:", "answer:", "the answer is:", "result:", 
        "solution:", "conclusion:", "final_answer:", "FINAL ANSWER:",
        "the formatted answer is:" # <-- Добавлен этот префикс
    ]
    for prefix in prefixes_to_remove:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()

    # Удаляем кавычки, если они окружают ответ (например, "ответ")
    if (answer.startswith('"') and answer.endswith('"')) or \
       (answer.startswith("'") and answer.endswith("'")):
        answer = answer[1:-1]
    
    # Снова удаляем пробелы после удаления кавычек/префиксов
    answer = answer.strip()

    # Форматирование согласно правилам GAIA в зависимости от типа 
    if answer_type.lower() == "number":
        # Удаляем все символы, кроме цифр, точек и знака минуса
        answer = re.sub(r'[^\d.-]', '', answer)
        try:
            num = float(answer)
            if num.is_integer():
                return str(int(num)) # Для целых чисел возвращаем без десятичной части
            return str(num)
        except ValueError:
            # Если не удалось преобразовать в число, возвращаем исходную строку.
            # Это, скорее всего, приведет к неправильному ответу на GAIA.
            return raw_answer 
            
    elif answer_type.lower() == "list":
        # Разбиваем по запятым или точкам с запятой, чистим пробелы и убираем артикли
        items = [item.strip() for item in re.split(r'[,;]\s*', answer)]
        cleaned_items = []
        for item in items:
            item = re.sub(r'^\b(a|an|the)\s+', '', item, flags=re.IGNORECASE) # Убираем артикли
            if item: # Добавляем только непустые элементы после очистки
                cleaned_items.append(item)
        return ', '.join(cleaned_items)
            
    elif answer_type.lower() == "string":
        # Убираем артикли и лишние пробелы в строке
        answer = re.sub(r'^\b(a|an|the)\s+', '', answer, flags=re.IGNORECASE)
        return answer.strip()
            
    else:
        # Если тип не указан или неверен, возвращаем максимально очищенный ответ как строку
        return answer.strip()
# <-- NEW SECTION END: Определение инструмента format_for_gaia -->

# Basic Agent Definition using smolagents
class GAIAAgent:
    def __init__(self):
        print("GAIAAgent initialized.")
        
        # Initialize model with token if available
        hf_token = os.getenv("HF_TOKEN")
        # <-- MODIFIED: Изменена модель на 72B для потенциально лучшей производительности,
        # как обсуждалось ранее. Если это вызовет проблемы с ресурсами Space,
        # можно вернуться к 7B.
        if hf_token:
            self.model = InferenceClientModel(model_id="Qwen/Qwen2.5-72B-Instruct", token=hf_token)
        else:
            self.model = InferenceClientModel(model_id="Qwen/Qwen2.5-72B-Instruct") 
        
        # Initialize search tool
        search_tool = WebSearchTool()
        
        # Create agent with the new formatting tool
        self.agent = CodeAgent(
            tools=[search_tool, format_for_gaia],  # <-- MODIFIED: Добавлен формат_для_gaia инструмент здесь
            model=self.model, 
            add_base_tools=True,
            # <-- MODIFIED: Обновлены разрешенные импорты, чтобы включить 're' для нового инструмента,
            # а также другие полезные библиотеки, такие как 'json', 'urllib.parse' для общих возможностей агента.
            additional_authorized_imports=['requests', 'bs4', 'json', 'urllib.parse', 're'] 
        )

        # <-- NEW SECTION START: Модифицированный системный промпт для обучения агента форматированию -->
        self.agent.prompt_templates["system_prompt"] = (
            self.agent.prompt_templates["system_prompt"] +
            """

            КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА ДЛЯ ТЕСТА GAIA:
            1. **Определите тип ответа**: Для каждого вопроса вы ДОЛЖНЫ определить ожидаемый тип ответа из контекста вопроса. Тип ДОЛЖЕН быть одним из "number" (число), "string" (строка) или "list" (список). Это абсолютно критично для правильного форматирования и оценки.
            2. **Используйте инструмент `format_for_gaia`**: После нахождения необработанного ответа вы ДОЛЖНЫ использовать инструмент `format_for_gaia` для правильного форматирования вашего ответа перед его отправкой.
               Пример для числа: `formatted_answer = format_for_gaia(raw_answer="1,234.5$", answer_type="number")`
               Пример для строки: `formatted_answer = format_for_gaia(raw_answer="The Apple", answer_type="string")`
               Пример для списка: `formatted_answer = format_for_gaia(raw_answer="item1; item2, item3", answer_type="list")`
            3. **Окончательный ответ (`final_answer`)**: Для завершения задачи вы ДОЛЖНЫ вызвать функцию `final_answer()` ТОЛЬКО с **отформатированным результатом**, полученным от `format_for_gaia`. Это должно быть вашим самым последним действием.
               Пример: `final_answer(formatted_answer)`
            4. **Избегайте лишнего текста**: НЕ включайте никаких вводных фраз, таких как "The final answer is:", "Answer:", или "FINAL ANSWER:" в ваш вызов `final_answer()`. Инструмент `format_for_gaia` сам обработает очистку, но убедитесь, что ваш `final_answer` содержит только отформатированную строку.
            5. **Точность для чисел**: При ответе числом убедитесь, что оно в чистом числовом формате (например, "1234.5", а не "1,234.50").
            6. **Элементы списка**: Для списков убедитесь, что элементы разделены запятыми и индивидуально очищены (например, "яблоко, банан, апельсин").
            7. **Строковые ответы**: Для строковых ответов предоставьте точную, краткую строку.
            """
        )
        # <-- NEW SECTION END: Модифицированный системный промпт -->

    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        
        try:
            submitted_answer = self.agent.run(question)
            print(f"Agent returning answer: {submitted_answer}")
            return str(submitted_answer)
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"Agent error: {error_msg}")
            return error_msg

# The rest of your run_and_submit_all function and Gradio interface remains the same.
# В этой части кода изменений нет, она осталась прежней.
def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the GAIAAgent on them, submits all answers,
    and displays the results.
    """
    space_id = os.getenv("SPACE_ID") 
    
    if profile:
        username= f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    try:
        agent = GAIAAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            print("Fetched questions list is empty.")
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
        print(f"Error decoding JSON response from questions endpoint: {e}")
        print(f"Response text: {response.text[:500]}")
        return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            submitted_answer = agent(question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            display_answer = str(submitted_answer)[:200] + "..." if len(str(submitted_answer)) > 200 else str(submitted_answer)
            results_log.append({"Task ID": task_id, "Question": question_text[:100] + "...", "Submitted Answer": display_answer})
            
        except Exception as e:
            print(f"Error running agent on task {task_id}: {e}")
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        simple_results = [{"Task": i+1, "Status": "Completed"} for i in range(len(answers_payload))]
        results_df = pd.DataFrame(simple_results)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df

# -- Build Gradio Interface using Blocks --
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**
        
        1. Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2. Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3. Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.
        
        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )
    
    gr.LoginButton()
    
    run_button = gr.Button("Run Evaluation & Submit All Answers")
    
    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)
    
    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") 
    
    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f" Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️ SPACE_HOST environment variable not found (running locally?).")
    
    if space_id_startup: 
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f" Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f" Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️ SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")
    
    print("-"*(60 + len(" App Starting ")) + "\n")
    
    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)