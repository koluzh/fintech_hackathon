import pandas as pd
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import requests
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import os
import asyncio
from tqdm.asyncio import tqdm_asyncio
from typing_extensions import Annotated, TypedDict
from typing import List
from langchain_openai import ChatOpenAI
import time

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

def rerank_docs_with_retry(query, documents, api_key, top_k=1, threshold=0.0, truncation_limit=1500, max_attempts=3, delay=2):
    for attempt in range(max_attempts):
        try:
            url = "https://ai-for-finance-hack.up.railway.app/rerank"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            truncated_docs = []
            for doc in documents:
                if len(doc) > truncation_limit:
                    truncated_docs.append(doc[:truncation_limit])
                else:
                    truncated_docs.append(doc)
            
            payload = {
                "model": "deepinfra/Qwen/Qwen3-Reranker-4B",
                "query": query,
                "documents": truncated_docs
            }
            
            start_time = time.time()
            response = requests.post(url, headers=headers, json=payload, timeout=12)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                sorted_results = sorted(result['results'], key=lambda x: x['relevance_score'], reverse=True)
                top_documents = []
                for res in sorted_results[:top_k]:
                    if res['relevance_score'] >= threshold:
                        doc_index = res['index']
                        top_documents.append({
                            'content': documents[doc_index],
                            'relevance_score': res['relevance_score']
                        })
                return top_documents, end_time - start_time
            else:
                print(f"Rerank error (attempt {attempt + 1}): {response.status_code}")
                if attempt < max_attempts - 1:
                    print(f"Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                    delay *= 1.5
                else:
                    return None, end_time - start_time
        except Exception as e:
            print(f"Rerank exception (attempt {attempt + 1}): {str(e)}")
            if attempt < max_attempts - 1:
                print(f"Waiting {delay} seconds before retry...")
                time.sleep(delay)
                delay *= 1.5
            else:
                return None, 0

def estimate_tokens(text):
    return len(text) // 2.4548

class Query(TypedDict):
    question: str
    problem: str
    answer_structure: str


class MyKAG:
    def __init__(self, model, store, reranker_api_key, reranker_top_k=1, reranker_threshold=0.0, truncation_limit=1500):
        self.model = model
        self.store = store
        self.queries = []
        self.results = []
        self.reranker_api_key = reranker_api_key
        self.reranker_top_k = reranker_top_k
        self.reranker_threshold = reranker_threshold
        self.truncation_limit = truncation_limit

    def formulate_query(self, question):
        prompt = f"""
        Из заданного вам вопроса сформулируйте структурированный запрос на векторную базу данных.
        Отвечайте кратко. В answer_structure также укажите то, что клиент хочет узнать из ответа на вопрос.
        question: вопрос с раскрытыми терминами, таким образом, чтобы не было двусмысленности или недосказанности.
        problem: проблема, которая поставлена в вопросе, а также толкование вопроса.
        answer_structure: структура ответа, которая полностью покроет заданный вопрос.
        Укажи темы, которые обязательно должны быть покрыты.
        Вопрос, на который нужно ответить:
        <question>
        {question}
        </question>
        """
        query = self.model.with_structured_output(Query).invoke(prompt)
        self.queries.append({'question': question, 'query': query})
        return query

    def get_knowledge(self, query):
        docs = self.store.similarity_search(query=str(query), k=20)
        knowledge_candidates = [doc.page_content for doc in docs]

        if knowledge_candidates and self.reranker_api_key:
            reranked_docs, reranker_time = rerank_docs_with_retry(
                query=str(query),
                documents=knowledge_candidates,
                api_key=self.reranker_api_key,
                top_k=self.reranker_top_k,
                threshold=self.reranker_threshold,
                truncation_limit=self.truncation_limit
            )

            if reranked_docs:
                knowledge = '\n'.join([doc['content'] for doc in reranked_docs])
                print(
                    f"Reranker applied: selected {len(reranked_docs)} documents with scores: {[doc['relevance_score'] for doc in reranked_docs]}")
                return knowledge

        knowledge = '\n'.join(knowledge_candidates[:self.reranker_top_k])
        return knowledge

    def answer(self, question):
        query = self.formulate_query(question)
        knowledge = self.get_knowledge(query)
        prompt = f"""
        Ответьте на вопрос клиента, используя только релевантную предоставленную информацию.
        Не используйте не относящиеся к вопросу данные. Приведите ответ на русском языке.
        Если в вопросе есть запрос на какие-то численные данные, и эти данные есть в предоставленных вам знаниях,
        обязательно предоставьте их. Отвечайте на вопрос так, чтобы помочь клиенту.
        Если в релевантных знаниях есть упоминание законодательной базы, обязательно предоставьте их. 
        Не упоминайте вещи, которых нет в ответе. Раскрывайте аббревиатуры. Не упоминай клиента или меня.
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
        answer = self.model.invoke(prompt).content
        result = {
            'question': question,
            'query': query,
            'result': answer,
            'knowledge': knowledge,
            'prompt': prompt
        }
        self.results.append(result)
        return result

async def answer_questions_with_executor(kag, questions, max_workers=20):
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=max_workers)

    tasks = []
    for i, q in enumerate(questions, start=1):
        fut = loop.run_in_executor(executor, kag.answer, q)
        tasks.append(fut)
        print(f"[SCHED] Task {i}/{len(questions)} scheduled")

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

async def main(questions, vectorstore, reranker_api_key):
    #model = ChatOpenAI(base_url=BASE_URL, model="x-ai/grok-3-mini", api_key=LLM_API_KEY)
    model = ChatOpenAI(base_url=BASE_URL, model="openrouter/x-ai/grok-3-mini", api_key=LLM_API_KEY)
    kag = MyKAG(
        model=model,
        store=vectorstore,
        reranker_api_key=reranker_api_key,
        reranker_top_k=10,
        reranker_threshold=0.2,
        truncation_limit=1500  
    )
    
    results = await answer_questions_with_executor(kag, questions, max_workers=20)
    return results

if __name__ == '__main__':
    load_dotenv()
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")
    RERANKER_API_KEY = os.getenv("EMBEDDER_API_KEY")  # юзаем их EMBEDDER_API_KEY с https://ai-for-finance-hack.up.railway.app/rerank
    
    BASE_URL = 'https://ai-for-finance-hack.up.railway.app/' 
    # BASE_URL = 'https://openrouter.ai/api/v1'
    client = OpenAI(
        base_url=BASE_URL,
        api_key=LLM_API_KEY,
    )
    
    questions = pd.read_csv('questions.csv').iloc[:10]
    corpus = pd.read_csv('train_data.csv').iloc[:10]
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", base_url=BASE_URL, api_key=EMBEDDER_API_KEY)
    # model = ChatOpenAI(base_url=BASE_URL, model="x-ai/grok-3-mini", api_key=LLM_API_KEY)
    model = ChatOpenAI(base_url=BASE_URL, model="openrouter/x-ai/grok-3-mini", api_key=LLM_API_KEY)
    
    chunker = LLMChunker(model, corpus)
    asyncio.run(chunker.chunk_corpus())
    my_chunks = chunker.chunks
    
    vectorstore = asyncio.run(InMemoryVectorStore.afrom_texts(
        my_chunks,
        embedding=embeddings,
    ))
    
    partial_answers = []
    unanswered_questions = list(questions['Вопрос'])
    answers = pd.DataFrame([], columns=['question', 'query', 'result', 'knowledge', 'prompt'])
    attempts = 0
    
    while len(partial_answers) < len(questions) and attempts < 5:
        partial_answers = asyncio.run(
            main(
                unanswered_questions,
                vectorstore,
                RERANKER_API_KEY
            )
        )
        partial_answers = pd.DataFrame(partial_answers)
        unanswered_questions = list(set(partial_answers['question']) ^ set(unanswered_questions))
        answers = pd.concat([answers, partial_answers], axis=0)
        attempts += 1
    
    result = questions.merge(answers, left_on='Вопрос', right_on='question', how='left').drop_duplicates('result')
    final_result = result.drop(columns=['question', 'query', 'knowledge', 'prompt']).rename({'result': 'Ответы на вопрос'})
    final_result.to_csv('submission.csv', index=False)
