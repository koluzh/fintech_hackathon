from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import argparse
import asyncio
import logging
import json
import os
import pandas as pd

load_dotenv()
LLM_API_KEY = os.getenv("LLM_API_KEY")

logging.basicConfig(filename="grader.log", filemode="a", level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class GradeResponse(BaseModel):
    grade: str = Field(..., description="Оценка ответа")
    reasoning: str = Field(..., description="Обоснование оценки")

class Grader:
    def __init__(self, grading_model, comparison = False):
        self.grading_model = grading_model
        self.comparison = comparison
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1",api_key=LLM_API_KEY)
     
    def get_grading_prompts(self, question, answer_for_grading):
        return {
            "system_prompt": f"""
    You are expert with task of judging RAG answers.
    You will receive a question-answer pair in Russian. Your task is to evaluate and score the answer based on its relevance to the question provided.
Instructions:

1. Reasoning: 
   Analyze the block by identifying key information and how it relates to the query. Consider whether the block provides direct answers, partial insights, or background context relevant to the query. Explain your reasoning in a few sentences, referencing specific elements of the block to justify your evaluation. Avoid assumptions—focus solely on the content provided.

2. Relevance Score (0 to 1, in increments of 0.1):
   0 = Completely Irrelevant: The block has no connection or relation to the query.
   0.1 = Virtually Irrelevant: Only a very slight or vague connection to the query.
   0.2 = Very Slightly Relevant: Contains an extremely minimal or tangential connection.
   0.3 = Slightly Relevant: Addresses a very small aspect of the query but lacks substantive detail.
   0.4 = Somewhat Relevant: Contains partial information that is somewhat related but not comprehensive.
   0.5 = Moderately Relevant: Addresses the query but with limited or partial relevance.
   0.6 = Fairly Relevant: Provides relevant information, though lacking depth or specificity.
   0.7 = Relevant: Clearly relates to the query, offering substantive but not fully comprehensive information.
   0.8 = Very Relevant: Strongly relates to the query and provides significant information.
   0.9 = Highly Relevant: Almost completely answers the query with detailed and specific information.
   1 = Perfectly Relevant: Directly and comprehensively answers the query with all the necessary specific information.

3. Additional Guidance:
   - Objectivity: Evaluate block based only on their content relative to the query.
   - Clarity: Be clear and concise in your justifications.
   - No assumptions: Do not infer information beyond what's explicitly stated in the block.

    Answer in Russian, other languages are not valid. You MUST provide feedback in format:

    Feedback:::
    Score: 
    Reasoning: (justify the score)
""".strip(),
            "user_prompt": f"""
    Даны следующие вопрос и ответ:

    ```
    Вопрос: {question}
    Ответ: {answer_for_grading}
    ```

    Оцени ответ и дай обратную связь
        """.strip()
        }

    def get_comparing_prompts(self, question, test_answer, baseline_answer): 
        return {
            "system_prompt": f"""
        Тебе будет выданы вопрос пользователя и два варианта ответа от Test и Baseline. 
        Твоя задача - выбрать, какой из ответов будет более полезен и понятен пользователю, задавшему вопрос.

        Также предоставь обратную связь в виде:

        Обратная связь:::
        Оценка: (какой из ответов выбран, Test или Baseline)
        Обоснование: (опиши причины для оценки в виде текста)

        """.strip(),
            "user_prompt": f"""
        Даны следующие вопрос и ответ:

        ```
        Вопрос: {question}
        Ответ Test: {test_answer}
        Ответ Baseline: {baseline_answer}
        ```

        Твоя задача - выбрать, какой из ответов будет более полезен и понятен пользователю, задавшему вопрос. 

        Обратная связь:::
        Оценка: (какой из ответов выбран, Test или Baseline)
        Обоснование: (опиши причины для оценки в виде текста)

        Ты должен предоставить оценку и обоснование. 
        """.strip()
    }

    def send_answer(self, prompts):
        response = None
        try: 
            response = self.client.chat.completions.parse(
                model = self.grading_model,
                messages = [
                    {
                        "role": "system",
                        "content": prompts["system_prompt"]
                    },
                    {
                        "role": "user",
                        "content": prompts["user_prompt"]
                    }
                ],
                response_format=GradeResponse
            )
            
            return response.choices[0].message.parsed
        except Exception as e:
            logger.debug(f"Error: {e}, Response: {response}")
            return e

    def grade_answer(self, data):
        try:
            if self.comparison == True:
                question_id, question, answer, baseline_answer = data 
                prompts = self.get_comparing_prompts(question, answer, baseline_answer)
            else:
                question_id, question, answer = data
                prompts = self.get_grading_prompts(question, answer)                   
            response = self.send_answer(prompts)
        except Exception as e:
            logger.debug(f"Error in grade_answer: {e}, Response: {response}")
            return e
        
        result = {
            "question_id": question_id, 
            "question": question,
            "answer": answer,
            "grade" : response.grade,
            "reasoning": response.reasoning
        }

        if self.comparison == True and baseline_answer is not None:
            result['baseline_answer'] = baseline_answer
    
        return result
    
    async def start(self, df):
        max_workers = 20
               
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=max_workers)

        tasks = []
        tasks_count = len(df.index)

        for i, (index, row) in enumerate(df.iterrows(), start=1):
            data = row.to_list()
            fut = loop.run_in_executor(executor, self.grade_answer, data)
            tasks.append(fut) 
            print(f"[SCHED] Task {i}/{tasks_count} scheduled - {len(data)}")
        
        results = []
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Answering questions"):
            try:
                res = await f
                results.append(res)
                print("[OK] one result collected")
            except Exception as e:
                print("[ERROR] task raised:", repr(e))
                results.append(None)
        
        executor.shutdown(wait=True)

        return results
 
async def main(grade_threshold, comparison):
    results = await grader.start(df=test_data)
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    print(f"Результаты:")
    print(f"Оценивающая модель: {grading_model}")
    print(f"Всего вопросов: {len(results)}")
    if comparison == True:
        print(f"Выбран ответ Test: {len(results_df.loc[results_df['grade'] == "Test"])}")
        print(f"Выбран ответ Baseline: {len(results_df.loc[results_df['grade'] == "Baseline"])}")
    else:
        answers_below_threshold = results_df.loc[results_df['grade'] < grade_threshold]
        print(f"Количество ответов с оценкой ниже порога {grade_threshold}: {len(answers_below_threshold.index)}")
        print(f"   • Ответы: {answers_below_threshold['question_id'].to_list()}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Оценка")

    parser.add_argument(
        "--input_path",
        type=str,
        required=False,
        default="submission.csv",
        help="CSV файл с ответами для оценки",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        default="grades.csv",
        help="Путь для результата",
    )

    parser.add_argument(
        "--grading_model",
        type=str,
        required=False,
        default="qwen/qwen3-vl-8b-instruct",
        help="Оценивающая модель",
    )

    parser.add_argument(
        "--grade_threshold",
        type=float,
        required=False,
        default="1",
        help="Граница для прохождения проверки",
    )

    parser.add_argument(
        "--baseline_path",
        type=str,
        required=False,
        help="Бейслайн для сравнения"
    )

    args = parser.parse_args()
    baseline_path = args.baseline_path
    input_path = args.input_path
    output_path = args.output_path
    grading_model = args.grading_model
    grade_threshold = args.grade_threshold

    if os.path.exists(input_path):
        test_data = pd.read_csv(input_path)
    try:
        if baseline_path is not None and os.path.exists(baseline_path):
            comparison = True
            baseline_data = pd.DataFrame(pd.read_csv(baseline_path), columns=['ID вопроса', 'Ответы на вопрос']).rename(columns={'Ответы на вопрос': 'Бейслайн'}, errors="raise")
            if len(test_data) != len(baseline_data):
                raise ValueError("Количество ответов в тесте и бейслайне отличаются")
            test_data = pd.merge(test_data, baseline_data, on='ID вопроса')
        else:
            comparison = False

        grader = Grader(grading_model, comparison)
        test_data = test_data

        start = perf_counter()
        asyncio.run(main(grade_threshold, comparison))
        stop = perf_counter()

        print("time taken: ", stop - start)

    except ValueError as e:
        print(f"❌ {e}")


