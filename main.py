import pandas as pd
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
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
import time
from pydantic import BaseModel


class Paragraph(BaseModel):
    title: str
    context: str
    problem: str
    solution: str


class ParagraphedText(BaseModel):
    paragraphs: List[Paragraph]


class Query(BaseModel):
    question: str
    problem: str
    answer_structure: str


class LLMChunker:
    def __init__(self, client, model_name, corpus, max_concurrent=20):
        self.client = client
        self.model_name = model_name
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

                # Заменяем вызов LangChain на прямой вызов OpenAI
                response = await asyncio.to_thread(
                    self.client.beta.chat.completions.parse,
                    model=self.model_name,
                    messages=[{"role": "user", "content": query}],
                    response_format=ParagraphedText
                )

                result = response.choices[0].message.parsed
                self.raw_paragraphs.append(result)
                return {'text_id': row_id, 'paragraphs': result.paragraphs}
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
            self.chunk_text(row['annotation'], row['text'], row['id'])
            for _, row in self.corpus.iterrows()
        ]
        results = await tqdm_asyncio.gather(*tasks, desc="Processing chunks")

        for result in results:
            if len(result) == 0:
                continue
            paragraphs = result['paragraphs']
            text_id = result['text_id']
            for x in paragraphs:
                x_dict = x.dict()
                self.chunks.append({'chunk': self.dict_to_xml(x_dict), 'text_id': text_id})


def rerank_docs_with_retry(query, documents, api_key, top_k=1, threshold=0.0, truncation_limit=1500, max_attempts=3,
                           delay=2):
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

def retry(times):
    """
    Retry Decorator
    Retries the wrapped function/method `times` times if the exceptions listed
    in ``exceptions`` are thrown
    :param times: The number of times to repeat the wrapped function/method
    :type times: Int
    :param Exceptions: Lists of exceptions that trigger a retry attempt
    :type Exceptions: Tuple of Exceptions
    """
    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    return func(*args, **kwargs)
                except:
                    print(
                        'Exception thrown when attempting to run %s, attempt '
                        '%d of %d' % (func, attempt, times)
                    )
                    attempt += 1
            return func(*args, **kwargs)
        return newfn
    return decorator


class MyKAG:
    def __init__(self, client, model_name, store, reranker_api_key, reranker_top_k=1, reranker_threshold=0.0,
                 truncation_limit=1500, corpus=None):
        self.client = client
        self.model_name = model_name
        self.store = store
        self.queries = []
        self.results = []
        self.reranker_api_key = reranker_api_key
        self.reranker_top_k = reranker_top_k
        self.reranker_threshold = reranker_threshold
        self.truncation_limit = truncation_limit
        self.corpus = corpus
    @retry(3)
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

        response = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format=Query
        )

        query = response.choices[0].message.parsed
        self.queries.append({'question': question, 'query': query})
        return query

    def get_knowledge(self, query):
        docs = self.store.similarity_search(query=str(query.dict()), k=20)
        knowledge_candidates = [doc.page_content for doc in docs]

        # if knowledge_candidates and self.reranker_api_key:
        #     reranked_docs, reranker_time = rerank_docs_with_retry(
        #         query=str(query.dict()),
        #         documents=knowledge_candidates,
        #         api_key=self.reranker_api_key,
        #         top_k=self.reranker_top_k,
        #         threshold=self.reranker_threshold,
        #         truncation_limit=self.truncation_limit
        #     )
        #
        #     if reranked_docs:
        #         knowledge = '\n'.join([doc['content'] for doc in reranked_docs])
        #         if len(reranked_docs) > 0:
        #             best_text_id = [doc for doc in docs if reranked_docs[0]['content'] in doc.page_content]
        #             if len(best_text_id) > 0:
        #                 best_text_id = best_text_id[0].metadata['text_id']
        #                 try:
        #                     best_text = corpus[corpus['id'] == best_text_id].iloc[0]
        #                 except:
        #                     best_text = ''
        #                 knowledge = '\n'.join([knowledge, f'# Полный текст: {best_text}'])
        #         print(
        #             f"Reranker applied: selected {len(reranked_docs)} documents with scores: {[doc['relevance_score'] for doc in reranked_docs]}")
        #         return knowledge

        try:
            best_text_id = docs[0].metadata['text_id']
            best_text = corpus[corpus['id'] == best_text_id].iloc[0]['text']
        except:
            best_text = ''

        knowledge = '\n'.join(knowledge_candidates[:self.reranker_top_k] + [best_text])
        return knowledge

    def answer(self, question):
        query = self.formulate_query(question)
        knowledge = self.get_knowledge(query)
        prompt = f"""Используя следующую информацию, дай структурированный ответ на вопрос.

Информация:
{knowledge}

Вопрос: {question}

Требования к ответу:
- Начни сразу с ответа, без приветствий и вводных фраз
- НЕ упоминай, что ты AI или консультант
- НЕ упоминай наличие или отсутствие информации в контексте
- Используй структуру с заголовками (###) и подразделами
- Включи практические примеры и конкретные цифры где возможно
- Будь информативным, но конкретным - фокусируйся на главном
- Оптимальный размер ответа: около 3000-3500 символов
- Пиши на русском языке

Ответ:"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=3200
        )

        answer = response.choices[0].message.content
        result = {
            'question': question,
            'query': query.dict(),
            'result': answer,
            'knowledge': knowledge,
            'prompt': prompt
        }
        self.results.append(result)
        return result


async def answer_questions_with_executor(kag, questions, max_workers=10):
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=max_workers)

    tasks = []
    for i, q in enumerate(questions, start=1):
        fut = loop.run_in_executor(executor, kag.answer, q)
        tasks.append(fut)
        print(f"[SCHED] Task {i}/{len(questions)} scheduled")

    results = []
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Answering questions"):

        # res = await f
        # results.append(res)
        try:
            res = await f
            results.append(res)
            print("[OK] one result collected")
        except Exception as e:
            print("[ERROR] task raised:", repr(e))
            results.append(None)

    executor.shutdown(wait=True)
    return results


async def main(questions, vectorstore, reranker_api_key, client, model_name, corpus):
    kag = MyKAG(
        client=client,
        model_name=model_name,
        store=vectorstore,
        reranker_api_key=reranker_api_key,
        reranker_top_k=10,
        reranker_threshold=0.01,
        truncation_limit=1500,
        corpus=corpus
    )

    results = await answer_questions_with_executor(kag, questions, max_workers=10)
    return results


if __name__ == '__main__':
    load_dotenv()
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")
    RERANKER_API_KEY = os.getenv("EMBEDDER_API_KEY")

    BASE_URL = 'https://ai-for-finance-hack.up.railway.app/'
    MODEL_NAME = "openrouter/x-ai/grok-3-mini"
    # BASE_URL = 'https://openrouter.ai/api/v1'
    # MODEL_NAME = "x-ai/grok-3-mini"

    client = OpenAI(
        base_url=BASE_URL,
        api_key=LLM_API_KEY,
    )

    questions = pd.read_csv('questions.csv')
    corpus = pd.read_csv('train_data.csv')

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", base_url=BASE_URL, api_key=EMBEDDER_API_KEY)

    chunker = LLMChunker(client=client, model_name=MODEL_NAME, corpus=corpus)
    asyncio.run(chunker.chunk_corpus())
    my_chunks = chunker.chunks
    my_docs = [Document(page_content=chunk['chunk'], metadata={'text_id': chunk['text_id']}) for chunk in my_chunks]

    vectorstore = asyncio.run(InMemoryVectorStore.afrom_documents(
        my_docs,
        embedding=embeddings,
    ))
    vectorstore.dump('./mystore.db')

    # vectorstore = InMemoryVectorStore.load('mystore.db', embedding=embeddings)
    partial_answers = []
    unanswered_questions = list(questions['Вопрос'])
    answers = pd.DataFrame([], columns=['question', 'query', 'result', 'knowledge', 'prompt'])
    attempts = 0

    while len(answers) < len(questions) and attempts < 5:
        partial_answers = asyncio.run(
            main(
                unanswered_questions,
                vectorstore,
                RERANKER_API_KEY,
                client,
                MODEL_NAME,
                corpus
            )
        )
        partial_answers = pd.DataFrame(partial_answers)
        if len(partial_answers) > 0 and 'question' in partial_answers.columns:
            unanswered_questions = list(set(unanswered_questions) - set(partial_answers['question']))
            answers = pd.concat([answers, partial_answers], axis=0)
        attempts += 1

    result = questions.merge(answers, left_on='Вопрос', right_on='question', how='left').drop_duplicates('Вопрос')
    # result.to_csv('not_submission.csv', index=False)
    final_result = result.drop(columns=['question', 'query', 'knowledge', 'prompt']).rename(
        {'result': 'Ответы на вопрос'})
    final_result.to_csv('submission.csv', index=False)