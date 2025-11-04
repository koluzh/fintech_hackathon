from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from time import perf_counter
import argparse
import asyncio
import os
import pandas as pd

load_dotenv()
LLM_API_KEY = os.getenv("LLM_API_KEY")

class GradeResponse(BaseModel):
    grade: int = Field(..., description="Оценка ответа")
    reasoning: str = Field(..., description="Обоснование оценки")

class Grader:
    def __init__(self, grading_model):
        self.grading_model = grading_model
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1",api_key=LLM_API_KEY)
     
    def get_system_prompt(self):
        return f"""
    Тебе будет выдана пара вопрос-ответ. Твоя задача - оценить насколько ответ полезен и понятен пользователю, задавшему вопрос.

    Дай оценку по шкале от 1 до 5:
    1: Бесполезный ответ, либо не связан с вопросом
    2: Краткий ответ, в ответе упущены ключевые моменты 
    3: Полезный, но недостаточно полный ответ
    4: Хороший ответ, есть моменты для улучшения
    5: Отличный ответ, понятно и подробно отвечает на вопрос

    Также предоставь обратную связь в виде:

    Обратная связь:::
    Оценка: (твоя оценка по шкале от 1 до 5)
    Обоснование: (опиши причины для оценки в виде текста)

    """.strip()

    def get_grading_prompt(self, question, answer_for_grading):
        return f"""
    Даны следующие вопрос и ответ:

    ```
    Вопрос: {question}
    Ответ: {answer_for_grading}
    ```

    Оцени ответ по шкале 1 до 5, и дай обратную связь
        """.strip()

    def send_answer(self, question, answer):
        response = self.client.chat.completions.parse(
                model = self.grading_model,
                messages = [
                    {
                        "role": "system",
                        "content": self.get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": self.get_grading_prompt(question, answer)
                    }
                ],
                response_format=GradeResponse
        )

        return response.choices[0].message.parsed

    async def grade_answer(self, question, answer):
        try:
            response: GradeResponse = self.send_answer(question, answer)
        except Exception as e:
            return print(e, response)
        
        return {
            "question": question,
            "answer": answer,
            "grade" : response.grade,
            "reasoning": response.reasoning
        }

    async def start(self, df, batch_size: int = 3, output_path: str = 'grades.csv'):
        questions = df['question'].tolist()
        answers = df['result'].tolist()

        results = []

        if len(questions) != len(answers):
            raise "Количество вопросов и ответов не совпадает"
                
        batches_total = (len(questions) + batch_size - 1) // batch_size

        for batch_idx, i in enumerate(range(0, len(questions), batch_size), start=1):
            batch = questions[i : i + batch_size], answers[i : i + batch_size]

            print(f"Начался батч {batch_idx}/{batches_total} (вопросы {i + 1}-{min(i + batch_size, len(questions))})")
            print(f"Batch: {batch}")

            batch_grades = await asyncio.gather(
                *[self.grade_answer(question, answer) for question , answer in zip(*batch)]
            )

            results.extend(batch_grades)
            pd.DataFrame(results).to_csv(output_path, index=False)

            print(
            f"Завершен батч {batch_idx}/{batches_total}."
            f"Обработано вопросов: {len(results)}/{len(questions)}"
            )

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
        "--batch_size",
        type=int,
        required=False,
        default="3",
        help="Размер батча",
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
        type=int,
        required=False,
        default="3",
        help="Граница для прохождения проверки",
    )

    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    batch_size = args.batch_size
    grading_model = args.grading_model
    grade_threshold = args.grade_threshold

    if os.path.exists(input_path):
        df = pd.read_csv(input_path)
        columns = ['', 'question', 'result']
        test_data = pd.DataFrame(df, columns=columns)
    try:
        grader = Grader(grading_model)
    except ValueError as e:
        print(f"❌ {e}")

    start = perf_counter()
    asyncio.run(grader.start(df=test_data, batch_size=batch_size, output_path=output_path))
    stop = perf_counter()

    print("time taken: ", stop - start)
    if os.path.exists(output_path):
        results = pd.read_csv(output_path)

        passed = results[results['grade'] >= grade_threshold].count()
        failed = results[results['grade'] < grade_threshold].count()
        missing = results[results['grade'] == ''].count()    

        print(f"📊 Результаты:")
        print(f"   • Оценивающая модель: {grading_model}")
        print(f"   • Всего вопросов: {len(results)}")
        print(f"   • Ответы прошедшие проверку: {passed} ({passed/len(results)*100:.1f}%)")
        print(f"   • Ответы провалившие проверку: {failed} ({passed/len(results)*100:.1f}%)")
        print(f"   • Нет оценки: {missing} ({missing/len(results)*100:.1f}%)")