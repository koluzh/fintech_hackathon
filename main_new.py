import pandas as pd
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import os
import asyncio
from tqdm.asyncio import tqdm_asyncio
from typing_extensions import TypedDict
from typing import List
from langchain_openai import ChatOpenAI
import time
import aiohttp
from aiohttp import TCPConnector
import aiohappyeyeballs
import orjson
import numpy as np

class Paragraph(TypedDict):
    title: str
    context: str
    problem: str
    solution: str

class ParagraphedText(TypedDict):
    paragraphs: List[Paragraph]

class LLMChunker:
    def __init__(self, model, corpus, max_concurrent=20):
        self.model = model.with_structured_output(ParagraphedText)
        self.corpus = corpus
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.chunks = []
        self.failed_docs = []
        self.raw_paragraphs = []

    async def chunk_text(self, annotation, text, row_id=None):
        async with self.semaphore:
            try:
                query = f"""
                    Разбей этот текст на один или несколько параграфов.
                    Требования:
                    - Используй только русский язык.
                    - Каждый параграф должен быть самостоятельным текстом, отвечающим на конкретный вопрос.
                    - Перед обработкой текста, начни с краткого чек-листа (3-7 пунктов), описывающего основные шаги, которые ты выполнишь.
                    - Для каждого параграфа создай отдельный JSON-объект со следующими полями:
                      title, context, problem, solution.
                    - Используй ТОЛЬКО информацию, предоставленную в исходном тексте.
                    Текст:
                    {annotation} 
                    {text}"""
                result = await asyncio.to_thread(self.model.invoke, query)
                self.raw_paragraphs.append(result)
                return result['paragraphs']
            except Exception as e:
                print(f"Error on doc {row_id}: {e}")
                self.failed_docs.append(row_id)
                return []

    @staticmethod
    def tag_text(text, tag):
        return f'<{tag}>\n{text}\n</{tag}>'

    def dict_to_xml(self, doc_dict):
        return '\n'.join(self.tag_text(v, k) for k, v in doc_dict.items())

    async def chunk_corpus(self):
        tasks = [
            self.chunk_text(row['annotation'], row['text'], row.get('id'))
            for _, row in self.corpus.iterrows()
        ]
        results = await tqdm_asyncio.gather(*tasks, desc="Processing chunks")

        for result in results:
            for x in result:
                self.chunks.append(self.dict_to_xml(x))

async def rerank_docs_async(session, query, documents, api_key, top_k=1, threshold=0.0, truncation_limit=1500, max_attempts=2):
    for attempt in range(max_attempts):
        try:
            url = "https://ai-for-finance-hack.up.railway.app/rerank"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            truncated_docs = [doc[:truncation_limit] if len(doc) > truncation_limit else doc for doc in documents]
            
            payload = {
                "model": "deepinfra/Qwen/Qwen3-Reranker-4B",
                "query": query,
                "documents": truncated_docs
            }
            
            json_data = orjson.dumps(payload)
            
            start_time = time.time()
            
            try:
                async with asyncio.timeout(8):  
                    async with session.post(url, headers=headers, data=json_data) as response:
                        if response.status == 200:
                            result = await response.json()
                            end_time = time.time()
                            
                            sorted_results = sorted(result['results'], key=lambda x: x['relevance_score'], reverse=True)
                            top_documents = []
                            for res in sorted_results[:top_k]:
                                if res['relevance_score'] >= threshold:
                                    top_documents.append({
                                        'content': documents[res['index']],
                                        'relevance_score': res['relevance_score']
                                    })
                            return top_documents, end_time - start_time
                        else:
                            print(f"Rerank error (attempt {attempt + 1}): {response.status}")
            except asyncio.TimeoutError:
                print(f"Rerank timeout (attempt {attempt + 1})")
                continue
                
            await asyncio.sleep(1 * (attempt + 1))
        except Exception as e:
            print(f"Rerank exception (attempt {attempt + 1}): {str(e)}")
            await asyncio.sleep(1 * (attempt + 1))
    
    return None, 0


class Query(TypedDict):
    question: str
    problem: str
    answer_structure: str

class UltraOptimizedMyKAG:
    def __init__(self, model, store, reranker_api_key=None, reranker_top_k=1, reranker_threshold=0.0, truncation_limit=1500):
        self.model = model
        self.store = store
        self.reranker_api_key = reranker_api_key
        self.reranker_top_k = reranker_top_k
        self.reranker_threshold = reranker_threshold
        self.truncation_limit = truncation_limit
        self.query_cache = {}
        self.knowledge_cache = {}
        self._session = None

    async def get_session(self):
        if self._session is None:
            connector = TCPConnector(
                limit=100,
                limit_per_host=20,
                use_dns_cache=True,
                ttl_dns_cache=300,
                keepalive_timeout=30,
            )
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=10),
                json_serialize=lambda x: orjson.dumps(x).decode(),
            )
        return self._session

    async def close_session(self):
        if self._session:
            await self._session.close()
            self._session = None

    def formulate_query(self, question):
        cache_key = hash(question)
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
            
        prompt = f"""
        Из заданного вам вопроса сформулируйте структурированный запрос на векторную базу данных.
        Отвечайте кратко. В answer_structure также укажите то, что клиент хочет узнать из ответа на вопрос.
        question: вопрос без изменений
        problem: проблема, которая поставлена в вопросе
        answer_structure: структура ответа, которая полностью покроет заданный вопрос.
        Укажи темы, которые обязательно должны быть покрыты.
        Вопрос, на который нужно ответить:
        <question>
        {question}
        </question>
        """
        query = self.model.with_structured_output(Query).invoke(prompt)
        self.query_cache[cache_key] = query
        return query

    async def get_knowledge_async(self, query):
        query_str = str(query)
        cache_key = hash(query_str)
        
        if cache_key in self.knowledge_cache:
            return self.knowledge_cache[cache_key]
        
        docs = await asyncio.to_thread(self.store.similarity_search, query=query_str, k=20)
        knowledge_candidates = [doc.page_content for doc in docs]
        
        if knowledge_candidates and self.reranker_api_key:
            session = await self.get_session()

            reranked_docs, reranker_time = await rerank_docs_async(
                session=session,
                query=query_str,
                documents=knowledge_candidates,
                api_key=self.reranker_api_key,
                top_k=self.reranker_top_k,
                threshold=self.reranker_threshold,
                truncation_limit=self.truncation_limit,
                max_attempts=2
            )

            
            if reranked_docs:
                knowledge = '\n'.join([doc['content'] for doc in reranked_docs])
                self.knowledge_cache[cache_key] = knowledge
                return knowledge
        
        knowledge = '\n'.join(knowledge_candidates[:self.reranker_top_k])
        self.knowledge_cache[cache_key] = knowledge
        return knowledge

    async def answer_async(self, question):
        query = await asyncio.to_thread(self.formulate_query, question)
        knowledge_task = asyncio.create_task(self.get_knowledge_async(query))
        knowledge = await knowledge_task
        prompt = f"""
        Ответьте на вопрос клиента, используя только релевантную предоставленную информацию.
        Не используйте не относящиеся к вопросу данные. Приведите ответ на русском языке.
        Если в вопросе есть запрос на какие-то численные данные, и эти данные есть в предоставленных вам знаниях,
        обязательно предоставьте их. Отвечайте на вопрос так, чтобы помочь клиенту.
        Ответ на вопрос ДОЛЖЕН БЫТЬ СВЯЗАН С проблемой, поставленной в вопросе:
        <Problem>
        {query['problem']}
        </Problem>
        Используйте структуру ответа:
        <Structure>
        {query['answer_structure']}
        </Structure>
        <Question>
        {question}
        </Question>
        <Knowledge>
        {knowledge}
        </Knowledge>
        """
        answer = await asyncio.to_thread(self.model.invoke, prompt)
        return {
            'question': question,
            'query': query,
            'result': answer.content,
            'knowledge': knowledge
        }

async def ultra_optimized_answer_questions(kag, questions, max_concurrent=20):
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_single(question):
        async with semaphore:
            try:
                start_time = time.time()
                result = await kag.answer_async(question)
                processing_time = time.time() - start_time
                
                print(f"Question processing time: {processing_time:.2f}с")
                    
                return result
            except Exception as e:
                print(f"Error: {e}")
                return None
    
    tasks = [process_single(question) for question in questions]
    
    results = []
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Ultra optimized processing"):
        result = await future
        if result is not None:
            results.append(result)
    
    return results

async def smart_processing(questions, vectorstore, reranker_api_key, max_attempts=3):
    model = ChatOpenAI(
        base_url=BASE_URL, 
        model="openrouter/x-ai/grok-3-mini", 
        api_key=LLM_API_KEY,
        timeout=45,
        max_retries=2
    )
    
    kag = UltraOptimizedMyKAG(
        model=model,
        store=vectorstore,
        reranker_api_key=reranker_api_key,
        reranker_top_k=1,
        reranker_threshold=0.0,
        truncation_limit=1200
    )
    
    try:
        all_answers = []
        remaining_questions = questions.copy()
        
        for attempt in range(max_attempts):
            if not remaining_questions:
                break
                
            print(f"Попытка {attempt + 1}: {len(remaining_questions)} вопросов")
            
            answers = await ultra_optimized_answer_questions(kag, remaining_questions, max_concurrent=20)
            
            successful_answers = [a for a in answers if a is not None]
            all_answers.extend(successful_answers)
            
            answered_questions = set(a['question'] for a in successful_answers)
            remaining_questions = [q for q in remaining_questions if q not in answered_questions]
            
            print(f"Успешно: {len(successful_answers)}, осталось: {len(remaining_questions)}")
            
            if remaining_questions:
                await asyncio.sleep(1)
        
        return all_answers
    finally:
        await kag.close_session()

if __name__ == '__main__':
    load_dotenv()
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")
    RERANKER_API_KEY = os.getenv("EMBEDDER_API_KEY")
    
    BASE_URL = 'https://ai-for-finance-hack.up.railway.app/'
    
    questions = pd.read_csv('questions.csv').iloc[:10]
    corpus = pd.read_csv('train_data.csv').iloc[:10]
    
    print(f"Загружено {len(questions)} вопросов")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", base_url=BASE_URL, api_key=EMBEDDER_API_KEY)
    model = ChatOpenAI(base_url=BASE_URL, model="openrouter/x-ai/grok-3-mini", api_key=LLM_API_KEY)
    
    print("Подготовка чанков...")
    start_prep = time.time()
    chunker = LLMChunker(model, corpus)
    asyncio.run(chunker.chunk_corpus())
    my_chunks = chunker.chunks
    print(f"Чанкинг: {time.time() - start_prep:.2f}с")
    
    vectorstore = asyncio.run(InMemoryVectorStore.afrom_texts(my_chunks, embedding=embeddings))
    
    print("Запуск обработки...")
    start_time = time.time()
    
    questions_list = list(questions['Вопрос'])
    all_answers = asyncio.run(smart_processing(questions_list, vectorstore, RERANKER_API_KEY))
    
    total_time = time.time() - start_time
    print(f"Общее время: {total_time:.2f}с")
    print(f"Вопросов в секунду: {len(all_answers)/total_time if total_time > 0 else 0:.2f}")
    
    if all_answers:
        answers_df = pd.DataFrame(all_answers)
        result = questions.merge(answers_df, left_on='Вопрос', right_on='question', how='left')
        final_result = result.drop(columns=['question', 'query', 'knowledge']).rename({'result': 'Ответы на вопрос'})
        final_result.to_csv('submission.csv', index=False)
        print("Результаты сохранены")
