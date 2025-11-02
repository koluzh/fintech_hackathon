import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import os

# Подключаем все переменные из окружения
load_dotenv()
# Подключаем ключ для LLM-модели
LLM_API_KEY = os.getenv("LLM_API_KEY")
# Подключаем ключ для EMBEDDER-модели
EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")


# Функция для генерации ответа по заданному вопросу, вы можете изменять ее в процессе работы, однако
# просим оставить структуру обращения, т.к. при запуске на сервере, потребуется корректно указанный путь 
# для формирования ответов. Также не вставляйте ключ вручную, поскольку при запуске ключ подтянется автоматически
def answer_generation(question):
    # Подключаемся к модели
    client = OpenAI(
        # Базовый url - сохранять без изменения
        base_url="https://ai-for-finance-hack.up.railway.app/",
        # Указываем наш ключ, полученный ранее
        api_key=LLM_API_KEY,
    )
    # Формируем запрос к клиенту
    response = client.chat.completions.create(
        # Выбираем любую допступную модель из предоставленного списка
        model="openrouter/mistralai/mistral-small-3.2-24b-instruct",
        # Формируем сообщение
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Ответь на вопрос: {question}"
                    }
                ]
            }
        ]
    )
    # Формируем ответ на запрос и возвращаем его в результате работы функции
    return response.choices[0].message.content


# Блок кода для запуска. Пожалуйста оставляйте его в самом низу вашего скрипта,
# при необходимости добавить код - опишите функции выше и вставьте их вызов в блок после if
# в том порядке, в котором они нужны для запуска решения, пути к файлам оставьте неизменными.
if __name__ == "__main__":    
    # Считываем список вопросов
    questions = pd.read_csv('./questions.csv')
    # Выделяем список вопросов
    questions_list = questions['Вопрос'].tolist()
    # Создаем список для хранения ответов
    answer_list = []
    # Проходимся по списку вопросов
    for current_question in tqdm(questions_list, desc="Генерация ответов"):
        # Отправляем запрос на генерацию ответа
        answer = answer_generation(question=current_question)
        # Добавляем ответ в список
        answer_list.append(answer)
    # Добавляем в данные список ответов
    questions['Ответы на вопрос'] = answer_list
    # Сохраняем submission
    questions.to_csv('submission.csv', index=False)