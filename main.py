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
from concurrent.futures import ThreadPoolExecutor
import os
warnings.filterwarnings('ignore')

class FinancialAssistant:
    def __init__(self, ollama_url="http://localhost:11434"):
        self.ollama_url = ollama_url
        self.index = None
        self.documents = []
        self.embedding_dim = 768
        
        self._check_ollama_available()
    
    def _check_ollama_available(self):
        """Проверка доступности Ollama сервера"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                print("✅ Ollama сервер доступен")
                return True
            else:
                print("❌ Ошибка подключения к Ollama серверу")
                return False
        except Exception as e:
            print(f"❌ Не удалось подключиться к Ollama серверу: {e}")
            return False

    async def get_embeddings_async(self, texts: list, session: aiohttp.ClientSession, model_name: str = "embeddinggemma:300m") -> np.ndarray:
        """Асинхронное получение эмбеддингов"""
        embeddings = []
        
        async def fetch_embedding(text: str):
            try:
                async with session.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={
                        "model": model_name,
                        "prompt": text[:800]  # Уменьшили длину для стабильности
                    },
                    timeout=aiohttp.ClientTimeout(total=20)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        embedding = data.get('embedding', [])
                        if len(embedding) == self.embedding_dim:
                            return embedding
                    return None
            except Exception as e:
                return None
        
        # Ограничиваем параллельные запросы
        semaphore = asyncio.Semaphore(3)  # Уменьшили для стабильности
        
        async def bounded_fetch(text):
            async with semaphore:
                return await fetch_embedding(text)
        
        tasks = [bounded_fetch(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception) or result is None:
                embeddings.append([0.0] * self.embedding_dim)
            else:
                embeddings.append(result)
        
        return np.array(embeddings, dtype=np.float32)
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 300, overlap: int = 30) -> list:
        """Разбивка текста на перекрывающиеся чанки"""
        # Упрощенный чанкинг - берем первые абзацы
        paragraphs = text.split('\n\n')
        chunks = []
        
        for paragraph in paragraphs:
            if len(paragraph.strip()) > 50:  # Игнорируем очень короткие абзацы
                words = paragraph.split()
                # Берем первые chunk_size слов из каждого абзаца
                chunk = ' '.join(words[:chunk_size])
                chunks.append(chunk)
                
        return chunks[:10]  # Ограничиваем количество чанков на документ
    
    def build_knowledge_base(self, train_data_path: str):
        """Построение векторной базы знаний"""
        print("📚 Загрузка и обработка тренировочных данных...")
        train_data = pd.read_csv(train_data_path)
        
        # Обработка документов и создание чанков
        all_chunks = []
        for _, row in tqdm(train_data.iterrows(), total=len(train_data), desc="Обработка документов"):
            text = row['text']
            chunks = self._split_text_into_chunks(text)
            all_chunks.extend(chunks)
        
        self.documents = all_chunks
        print(f"✅ Создано {len(all_chunks)} чанков")
        
        # Создание эмбеддингов
        print("🔄 Создание эмбеддингов...")
        
        async def build_embeddings():
            async with aiohttp.ClientSession() as session:
                return await self.get_embeddings_async(all_chunks, session, "embeddinggemma:300m")
        
        embeddings = asyncio.run(build_embeddings())
        
        # Создание FAISS индекса
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype(np.float32))
        print(f"✅ Векторная база знаний создана. Размерность: {dimension}")
    
    async def search_relevant_documents_async(self, query: str, session: aiohttp.ClientSession, top_k: int = 5) -> list:
        """Асинхронный поиск релевантных документов"""
        if self.index is None:
            return []
        
        try:
            # Получаем эмбеддинг запроса
            query_embedding = await self.get_embeddings_async([query], session, "embeddinggemma:300m")
            if len(query_embedding) == 0:
                return []
            
            faiss.normalize_L2(query_embedding)
            similarities, indices = self.index.search(query_embedding.astype(np.float32), top_k)
            
            relevant_docs = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents) and similarities[0][i] > 0.1:  # Фильтр по сходству
                    relevant_docs.append(self.documents[idx])
            
            return relevant_docs
        except Exception as e:
            print(f"❌ Ошибка при поиске: {e}")
            return []

    async def generate_answer_async(self, question: str, context_docs: list, session: aiohttp.ClientSession) -> str:
        """Асинхронная генерация ответа"""
        try:
            if not context_docs:
                return "В предоставленной базе знаний нет информации по этому вопросу. Рекомендуется обратиться к официальным финансовым источникам."
            
            # Более качественный и менее строгий промпт
            context = "\n".join([f"- {doc}" for i, doc in enumerate(context_docs[:3])])
            
            prompt = f"""На основе следующей финансовой информации ответь на вопрос:

Информация:
{context}

Вопрос: {question}

Инструкции:
- Ответь кратко и по существу на русском языке
- Используй только предоставленную информацию
- Если информации недостаточно для полного ответа, дай частичный ответ на основе того, что есть
- Будь полезным и информативным

Ответ:"""

            async with session.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "gemma3:270m",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Немного увеличили для креативности
                        "num_predict": 400,
                        "top_k": 40,
                        "top_p": 0.9
                    }
                },
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    answer = result.get('response', '').strip()
                    
                    # Упрощенная проверка качества ответа
                    if len(answer) < 20 or "не знаю" in answer.lower() or "пожалуйста" in answer.lower():
                        return "На основе предоставленной информации невозможно дать точный ответ. Рекомендуется обратиться к официальным источникам."
                    
                    return answer
                else:
                    return "Не удалось сгенерировать ответ"
                    
        except Exception as e:
            print(f"❌ Ошибка при генерации: {e}")
            return "Ошибка при генерации ответа"

    async def process_questions_batch(self, questions_batch: list, progress_callback=None) -> list:
        """Обработка батча вопросов"""
        answers = []
        
        async with aiohttp.ClientSession() as session:
            for i, question in enumerate(questions_batch):
                try:
                    # Поиск релевантных документов
                    relevant_docs = await self.search_relevant_documents_async(question, session)
                    
                    # Для первых 3 вопросов покажем отладочную информацию
                    if i < 3:
                        print(f"\n🔍 Отладка вопроса {i+1}:")
                        print(f"   Вопрос: {question}")
                        print(f"   Найдено документов: {len(relevant_docs)}")
                        if relevant_docs:
                            print(f"   Пример документа: {relevant_docs[0][:100]}...")
                    
                    # Генерация ответа
                    answer = await self.generate_answer_async(question, relevant_docs, session)
                    answers.append(answer)
                    
                    # Прогресс
                    if progress_callback:
                        progress_callback()
                        
                    # Пауза между запросами
                    await asyncio.sleep(0.3)
                    
                except Exception as e:
                    print(f"❌ Ошибка обработки вопроса: {e}")
                    answers.append("Ошибка обработки вопроса")
                    if progress_callback:
                        progress_callback()
        
        return answers

def main():
    parser = argparse.ArgumentParser(description='Финансовый ассистент - улучшенная версия')
    parser.add_argument('--num_questions', type=int, default=500,
                       help='Количество вопросов для обработки')
    parser.add_argument('--ollama_url', type=str, default="http://localhost:11434",
                       help='URL Ollama сервера')
    parser.add_argument('--skip_build', action='store_true',
                       help='Пропустить построение базы знаний')
    parser.add_argument('--batch_size', type=int, default=5,
                       help='Размер батча для обработки')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 ФИНАНСОВЫЙ АССИСТЕНТ - УЛУЧШЕННАЯ ВЕРСИЯ")
    print("=" * 60)
    
    # Инициализация ассистента
    assistant = FinancialAssistant(ollama_url=args.ollama_url)
    
    # Построение базы знаний
    if not args.skip_build:
        print("\n📦 Этап 1: Построение векторной базы знаний...")
        build_start = time.time()
        assistant.build_knowledge_base('./train_data.csv')
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
    
    # Обработка вопросов батчами
    print(f"\n🎯 Этап 3: Обработка {len(questions_list)} вопросов...")
    
    async def process_all_questions():
        all_answers = []
        total_questions = len(questions_list)
        
        with tqdm(total=total_questions, desc="Обработка вопросов") as pbar:
            # Разбиваем на батчи
            for i in range(0, total_questions, args.batch_size):
                batch = questions_list[i:i + args.batch_size]
                
                # Обрабатываем батч
                batch_answers = await assistant.process_questions_batch(
                    batch, 
                    progress_callback=lambda: pbar.update(1)
                )
                all_answers.extend(batch_answers)
                
                # Пауза между батчами
                if i + args.batch_size < total_questions:
                    await asyncio.sleep(1)
        
        return all_answers
    
    process_start = time.time()
    answers_list = asyncio.run(process_all_questions())
    process_time = time.time() - process_start
    
    # Сохранение результатов
    print("\n💾 Этап 4: Сохранение результатов...")
    questions_df['Ответы на вопрос'] = answers_list
    output_file = f'submission_improved.csv'
    questions_df.to_csv(output_file, index=False, encoding='utf-8')
    
    # Статистика
    total_time = process_time + (0 if args.skip_build else build_time)
    print("=" * 60)
    print("🎉 ОБРАБОТКА ЗАВЕРШЕНА!")
    print(f"📊 Статистика:")
    print(f"   • Вопросов обработано: {len(questions_list)}")
    print(f"   • Время обработки: {process_time/60:.1f} минут")
    if not args.skip_build:
        print(f"   • Время построения базы: {build_time:.1f} секунд")
    print(f"   • Общее время: {total_time/60:.1f} минут")
    print(f"   • Файл результатов: {output_file}")
    print("=" * 60)
    
    # Примеры результатов
    print("\n📝 Примеры сгенерированных ответов:")
    print("-" * 80)
    sample_size = min(5, len(questions_list))
    for i in range(sample_size):
        print(f"\n❓ Вопрос {i+1}: {questions_list[i]}")
        print(f"💡 Ответ: {answers_list[i]}")
        print("-" * 80)

if __name__ == "__main__":
    main()