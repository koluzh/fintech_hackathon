import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
import requests
import time
import warnings
import argparse
import json
import asyncio
import aiohttp
import re
import os
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# Загружаем переменные окружения
load_dotenv()

class FinancialAssistant:
    def __init__(self):
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
        self.embedding_model = "openai/text-embedding-3-small"
        self.generation_model = "meta-llama/llama-3-70b-instruct"
        self.embedding_dim = 1536
        
        self.index = None
        self.documents = []
        
        # Раздельные счетчики для эмбеддингов и генерации
        self.embedding_requests = 0
        self.generation_requests = 0
        self.last_embedding_time = time.time()
        self.last_generation_time = time.time()

    async def get_embeddings_async(self, texts: list, session: aiohttp.ClientSession) -> np.ndarray:
        """Асинхронное получение эмбеддингов с улучшенным управлением лимитами"""
        embeddings = []
        
        async def fetch_embedding(text: str):
            try:
                # Проверяем лимит для эмбеддингов (60 в минуту)
                current_time = time.time()
                if current_time - self.last_embedding_time > 60:
                    self.embedding_requests = 0
                    self.last_embedding_time = current_time
                
                if self.embedding_requests >= 50:  # Берем 50 вместо 60 для надежности
                    wait_time = 60 - (current_time - self.last_embedding_time) + 1
                    if wait_time > 0:
                        print(f"⚠️ Лимит эмбеддингов. Ожидание {wait_time:.1f} секунд...")
                        await asyncio.sleep(wait_time)
                    self.embedding_requests = 0
                    self.last_embedding_time = time.time()
                
                self.embedding_requests += 1
                
                async with session.post(
                    "https://openrouter.ai/api/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.openrouter_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.embedding_model,
                        "input": text[:1500]
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        if 'data' in data and len(data['data']) > 0:
                            return data['data'][0]['embedding']
                    elif response.status == 429:
                        error_data = await response.text()
                        print(f"⚠️ Rate limit exceeded: {error_data}")
                        await asyncio.sleep(10)
                        return None
                    else:
                        error_data = await response.text()
                        print(f"❌ Ошибка эмбеддинга {response.status}: {error_data}")
                        return None
                        
            except Exception as e:
                print(f"❌ Исключение при получении эмбеддинга: {str(e)}")
                return None
        
        # Ограничиваем параллельные запросы
        semaphore = asyncio.Semaphore(3)  # Уменьшили для стабильности
        
        async def bounded_fetch(text):
            async with semaphore:
                return await fetch_embedding(text)
        
        # Показываем прогресс для эмбеддингов
        with tqdm(total=len(texts), desc="Создание эмбеддингов") as pbar:
            tasks = []
            for text in texts:
                task = asyncio.create_task(bounded_fetch(text))
                task.add_done_callback(lambda x: pbar.update(1))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception) or result is None:
                # Создаем случайный эмбеддинг в качестве fallback
                random_embedding = np.random.normal(0, 0.1, self.embedding_dim).tolist()
                embeddings.append(random_embedding)
                print(f"⚠️ Использован случайный вектор для текста {i+1}")
            else:
                embeddings.append(result)
                successful += 1
        
        print(f"✅ Успешных эмбеддингов: {successful}/{len(texts)}")
        return np.array(embeddings, dtype=np.float32)

    def _split_text_into_chunks(self, text: str, chunk_size: int = 800, overlap: int = 100) -> list:
        """Умное разбиение текста на чанки с ограничением количества"""
        # Сначала разбиваем на абзацы
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Если абзац сам по себе достаточно большой, используем его как чанк
            if len(paragraph) >= chunk_size // 2:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Разбиваем большой абзац на части
                words = paragraph.split()
                for i in range(0, len(words), chunk_size - overlap):
                    chunk = ' '.join(words[i:i + chunk_size])
                    if len(chunk) > 100:  # Минимальная длина чанка
                        chunks.append(chunk)
            else:
                # Объединяем маленькие абзацы
                if len(current_chunk + " " + paragraph) <= chunk_size:
                    current_chunk += " " + paragraph
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks[:20]  # Ограничиваем количество чанков на документ

    def build_knowledge_base(self, train_data_path: str, max_documents: int = 500):
        """Построение оптимизированной векторной базы знаний"""
        print("📚 Загрузка и создание базы знаний...")
        train_data = pd.read_csv(train_data_path)
        
        # Ограничиваем количество обрабатываемых документов
        if len(train_data) > max_documents:
            print(f"⚠️ Ограничение: используем первые {max_documents} документов из {len(train_data)}")
            train_data = train_data.head(max_documents)
        
        # Собираем тексты с умным чанкингом
        all_texts = []
        for _, row in tqdm(train_data.iterrows(), total=len(train_data), desc="Обработка документов"):
            text = row['text']
            
            # Очистка текста
            cleaned_text = re.sub(r'#{1,6}\s*', '', text)
            cleaned_text = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned_text)
            cleaned_text = re.sub(r'\n+', ' ', cleaned_text)
            
            # Умное разбиение на чанки
            chunks = self._split_text_into_chunks(cleaned_text)
            all_texts.extend(chunks)
            
            # Останавливаемся если достигли разумного предела
            if len(all_texts) >= 800:
                break
        
        self.documents = all_texts
        print(f"✅ Создано {len(all_texts)} чанков из {len(train_data)} документов")
        
        if not all_texts:
            print("❌ Нет данных для создания базы знаний")
            return
        
        # Создание эмбеддингов
        print("🔄 Создание эмбеддингов...")
        
        async def build_embeddings():
            async with aiohttp.ClientSession() as session:
                return await self.get_embeddings_async(all_texts, session)
        
        embeddings = asyncio.run(build_embeddings())
        
        # Создание FAISS индекса
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype(np.float32))
        print(f"✅ Векторная база знаний создана. Размерность: {dimension}")

    async def search_relevant_documents_async(self, query: str, session: aiohttp.ClientSession, top_k: int = 3) -> list:
        """Асинхронный поиск релевантных документов"""
        if self.index is None:
            return []
        
        try:
            # Получаем эмбеддинг запроса
            query_embedding = await self.get_embeddings_async([query], session)
            if len(query_embedding) == 0:
                return []
            
            faiss.normalize_L2(query_embedding)
            similarities, indices = self.index.search(query_embedding.astype(np.float32), top_k)
            
            relevant_docs = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents) and similarities[0][i] > 0.1:
                    relevant_docs.append(self.documents[idx])
            
            return relevant_docs
        except Exception as e:
            print(f"❌ Ошибка при поиске: {e}")
            return []

    async def generate_answer_async(self, question: str, context_docs: list, session: aiohttp.ClientSession) -> str:
        """Асинхронная генерация ответа с улучшенным управлением лимитами"""
        try:
            # Проверяем лимит для генерации
            current_time = time.time()
            if current_time - self.last_generation_time > 60:
                self.generation_requests = 0
                self.last_generation_time = current_time
            
            if self.generation_requests >= 50:  # 50 вместо 60 для надежности
                wait_time = 60 - (current_time - self.last_generation_time) + 1
                if wait_time > 0:
                    print(f"⚠️ Лимит генерации. Ожидание {wait_time:.1f} секунд...")
                    await asyncio.sleep(wait_time)
                self.generation_requests = 0
                self.last_generation_time = time.time()
            
            self.generation_requests += 1
            
            # Подготовка контекста
            context = "\n".join([f"- {doc}" for i, doc in enumerate(context_docs[:2])])
            
            prompt = f"""Ты - финансовый эксперт. Ответь на вопрос пользователя максимально точно и полезно.

Контекстная информация:
{context}

Вопрос: {question}

Требования к ответу:
- Будь точным и информативным
- Используй предоставленную информацию как основу
- Если информации недостаточно, дай общие рекомендации
- Отвечай на русском языке
- Будь кратким, но содержательным

Ответ:"""

            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/financial-assistant",
                    "X-Title": "Financial Assistant"
                },
                json={
                    "model": self.generation_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "max_tokens": 500,
                    "temperature": 0.3
                },
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    answer = result['choices'][0]['message']['content'].strip()
                    
                    # Проверка качества ответа
                    if len(answer) < 25 or "не могу" in answer.lower() or "не знаю" in answer.lower():
                        return "На основе предоставленной информации невозможно дать точный ответ. Рекомендуется обратиться к официальным финансовым источникам."
                    
                    return answer
                    
                elif response.status == 429:
                    error_data = await response.text()
                    print(f"⚠️ Rate limit exceeded for generation: {error_data}")
                    await asyncio.sleep(15)
                    return "Не удалось сгенерировать ответ из-за ограничения частоты запросов. Попробуйте позже."
                    
                else:
                    error_data = await response.text()
                    print(f"❌ Ошибка генерации {response.status}: {error_data}")
                    return "Ошибка при генерации ответа."
                    
        except Exception as e:
            print(f"❌ Исключение при генерации: {e}")
            return "Не удалось сгенерировать ответ."

    async def process_questions_batch(self, questions_batch: list, progress_callback=None) -> list:
        """Обработка батча вопросов"""
        answers = []
        
        async with aiohttp.ClientSession() as session:
            for i, question in enumerate(questions_batch):
                try:
                    # Поиск релевантных документов
                    relevant_docs = await self.search_relevant_documents_async(question, session)
                    
                    # Отладочная информация для первых вопросов
                    if i < 2:
                        print(f"\n🔍 Анализ вопроса {i+1}:")
                        print(f"   Вопрос: {question}")
                        print(f"   Найдено документов: {len(relevant_docs)}")
                    
                    # Генерация ответа
                    answer = await self.generate_answer_async(question, relevant_docs, session)
                    answers.append(answer)
                    
                    # Прогресс
                    if progress_callback:
                        progress_callback()
                        
                    # Пауза между запросами
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    print(f"❌ Ошибка обработки вопроса: {e}")
                    answers.append("Ошибка обработки вопроса.")
                    if progress_callback:
                        progress_callback()
        
        return answers

def main():
    parser = argparse.ArgumentParser(description='Финансовый ассистент с OpenRouter - оптимизированный')
    parser.add_argument('--num_questions', type=int, default=10,
                       help='Количество вопросов для обработки')
    parser.add_argument('--max_documents', type=int, default=300,
                       help='Максимальное количество документов для обработки')
    parser.add_argument('--skip_build', action='store_true',
                       help='Пропустить построение базы знаний')
    parser.add_argument('--batch_size', type=int, default=3,
                       help='Размер батча для обработки')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 ФИНАНСОВЫЙ АССИСТЕНТ С OPENROUTER - ОПТИМИЗИРОВАННЫЙ")
    print("=" * 60)
    
    # Проверка API ключа
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Ошибка: OPENROUTER_API_KEY не найден в переменных окружения")
        print("   Добавьте ваш API ключ в файл .env или установите переменную окружения")
        return
    
    # Инициализация ассистента
    try:
        assistant = FinancialAssistant()
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    # Построение базы знаний
    if not args.skip_build:
        print("\n📦 Этап 1: Построение оптимизированной векторной базы знаний...")
        build_start = time.time()
        assistant.build_knowledge_base('./train_data.csv', max_documents=args.max_documents)
        build_time = time.time() - build_start
        print(f"✅ База знаний построена за {build_time:.1f} секунд")
    
    # Загрузка вопросов
    print("\n📋 Этап 2: Загрузка вопросов...")
    questions_df = pd.read_csv('./questions.csv')
    questions_list = questions_df['Вопрос'].tolist()
    
    if args.num_questions > 0:
        questions_list = questions_list[:args.num_questions]
        questions_df = questions_df.head(args.num_questions)
    
    print(f"✅ Загружено {len(questions_list)} вопросов")
    
    # Обработка вопросов
    print(f"\n🎯 Этап 3: Обработка {len(questions_list)} вопросов...")
    
    async def process_all_questions():
        all_answers = []
        total_questions = len(questions_list)
        
        with tqdm(total=total_questions, desc="Обработка вопросов") as pbar:
            for i in range(0, total_questions, args.batch_size):
                batch = questions_list[i:i + args.batch_size]
                
                batch_answers = await assistant.process_questions_batch(
                    batch, 
                    progress_callback=lambda: pbar.update(1)
                )
                all_answers.extend(batch_answers)
                
                # Пауза между батчами
                if i + args.batch_size < total_questions:
                    await asyncio.sleep(3)  # Увеличили паузу
        
        return all_answers
    
    process_start = time.time()
    answers_list = asyncio.run(process_all_questions())
    process_time = time.time() - process_start
    
    # Сохранение результатов
    print("\n💾 Этап 4: Сохранение результатов...")
    questions_df['Ответы на вопрос'] = answers_list
    output_file = f'submission_openrouter_optimized_{len(questions_list)}.csv'
    questions_df.to_csv(output_file, index=False, encoding='utf-8')
    
    # Анализ результатов
    total_time = process_time + (0 if args.skip_build else build_time)
    
    # Категоризация ответов
    excellent = sum(1 for ans in answers_list if len(ans) > 80 and "информация отсутствует" not in ans.lower())
    good = sum(1 for ans in answers_list if 40 <= len(ans) <= 80 and "информация отсутствует" not in ans.lower())
    basic = len(answers_list) - excellent - good
    
    print("=" * 60)
    print("🎉 ОБРАБОТКА ЗАВЕРШЕНА!")
    print(f"📊 Результаты:")
    print(f"   • Всего вопросов: {len(questions_list)}")
    print(f"   • Отличные ответы: {excellent} ({excellent/len(questions_list)*100:.1f}%)")
    print(f"   • Хорошие ответы: {good} ({good/len(questions_list)*100:.1f}%)")
    print(f"   • Базовые ответы: {basic} ({basic/len(questions_list)*100:.1f}%)")
    print(f"   • Время обработки: {process_time/60:.1f} минут")
    if not args.skip_build:
        print(f"   • Время построения базы: {build_time:.1f} секунд")
    print(f"   • Общее время: {total_time/60:.1f} минут")
    print(f"   • Файл результатов: {output_file}")
    print("=" * 60)
    
    # Примеры ответов
    print("\n📝 Примеры сгенерированных ответов:")
    print("-" * 80)
    
    sample_indices = list(range(min(5, len(questions_list))))
    for i in sample_indices:
        print(f"\n❓ Вопрос {i+1}: {questions_list[i]}")
        print(f"💡 Ответ: {answers_list[i]}")
        print("-" * 80)

if __name__ == "__main__":
    main()