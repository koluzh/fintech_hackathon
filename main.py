import pandas as pd
import asyncio
import time
import requests
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from langchain_core.vectorstores import InMemoryVectorStore  
from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import List
import re
from math import log
from collections import defaultdict, Counter

load_dotenv()
LLM_API_KEY = os.getenv("LLM_API_KEY")
EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def get_client_config(mode):
    if mode == "prod":
        return {
            "base_url": "https://ai-for-finance-hack.up.railway.app/",
            "llm_api_key": LLM_API_KEY,
            "embedder_api_key": EMBEDDER_API_KEY,
            "llm_model": "openrouter/x-ai/grok-3-mini",
            "embedder_model": "text-embedding-3-small"
        }
    else:  # test mode
        return {
            "base_url": "https://openrouter.ai/api/v1",
            "llm_api_key": OPENROUTER_API_KEY,
            "embedder_api_key": OPENROUTER_API_KEY,
            "llm_model": "x-ai/grok-3-mini",
            "embedder_model": "text-embedding-3-small"
        }

def answer_generation(messages, mode="prod", temperature=0.7):
    config = get_client_config(mode)
    client = OpenAI(
        base_url=config["base_url"],
        api_key=config["llm_api_key"],
    )
    response = client.chat.completions.create(
        model=config["llm_model"],
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message.content

def get_embedding(text, mode="prod"):
    config = get_client_config(mode)
    client = OpenAI(
        base_url=config["base_url"],
        api_key=config["embedder_api_key"],
    )
    response = client.embeddings.create(
        model=config["embedder_model"],
        input=text
    )
    return response.data[0].embedding

def rerank_docs(query, documents, key, top_k=1, threshold=0.0, truncation_limit=1500):
    url = "https://ai-for-finance-hack.up.railway.app/rerank"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
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
    
    try:
        start_time = time.time()
        response = requests.post(url, headers=headers, json=payload, timeout=12)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(result)
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
            print(f"Rerank error: {response.status_code}")
            return None, end_time - start_time
    except Exception as e:
        print(f"Rerank exception: {str(e)}")
        return None, 0

class CustomEmbeddings:
    def __init__(self, mode="prod"):
        self.mode = mode
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            embedding = get_embedding(text, self.mode)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        return get_embedding(text, self.mode)

class SimpleBM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.doc_freqs = defaultdict(int)
        self.word_doc_freq = defaultdict(int)
        self.vocab = set()
        
    def preprocess_text(self, text):
        """Простая предобработка текста"""
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return words
    
    def fit(self, documents):
        """Построение индекса BM25 для коллекции документов"""
        self.corpus = documents
        self.doc_lengths = [len(self.preprocess_text(doc)) for doc in documents]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        
        # Строим частоты терминов в документах
        for doc in documents:
            words = self.preprocess_text(doc)
            unique_words = set(words)
            for word in unique_words:
                self.word_doc_freq[word] += 1
            self.vocab.update(words)
        
        self.total_docs = len(documents)
    
    def calculate_idf(self, term):
        """Вычисление IDF для термина"""
        if term not in self.word_doc_freq:
            return 0
        return log((self.total_docs - self.word_doc_freq[term] + 0.5) / 
                  (self.word_doc_freq[term] + 0.5) + 1.0)
    
    def score(self, query, doc_index):
        """Вычисление BM25 score для запроса и документа"""
        if doc_index >= len(self.corpus):
            return 0
            
        doc_words = self.preprocess_text(self.corpus[doc_index])
        query_words = self.preprocess_text(query)
        
        score = 0.0
        doc_term_freq = Counter(doc_words)
        doc_length = self.doc_lengths[doc_index]
        
        for word in query_words:
            if word not in self.vocab:
                continue
                
            tf = doc_term_freq.get(word, 0)
            idf = self.calculate_idf(word)
            
            # Формула BM25
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            
            if denominator > 0:
                score += idf * numerator / denominator
                
        return score
    
    def search(self, query, top_k=5):
        """Поиск топ-K документов по BM25"""
        scores = []
        for i in range(len(self.corpus)):
            score = self.score(query, i)
            scores.append((i, score))
        
        # Сортируем по убыванию score
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

async def basic_retrieval(vectorstore, embeddings, query: str, k: int = 5, threshold: float = 0.0, 
                         use_bm25: bool = False, bm25_index=None):
    start_time = time.time()
    
    if use_bm25 and bm25_index:
        # Используем BM25 для поиска
        search_start_time = time.time()
        bm25_results = bm25_index.search(query, top_k=k)
        search_time = time.time() - search_start_time
        
        filtered_documents = []
        for doc_index, score in bm25_results:
            if score >= threshold:
                # Получаем документ из vectorstore.store
                store_keys = list(vectorstore.store.keys())
                if doc_index < len(store_keys):
                    doc_id = store_keys[doc_index]
                    doc_data = vectorstore.store[doc_id]
                    filtered_documents.append({
                        'content': doc_data['text'],
                        'similarity_score': score
                    })
        
        total_time = time.time() - start_time
        return filtered_documents, total_time, 0, search_time
    
    else:
        # Используем стандартный эмбеддинговый поиск
        embedding_start_time = time.time()
        query_embedding = embeddings.embed_query(query)
        embedding_time = time.time() - embedding_start_time
        
        search_start_time = time.time()
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            results = await loop.run_in_executor(
                executor, 
                lambda: vectorstore.similarity_search_with_score(query, k=k)
            )
        search_time = time.time() - search_start_time
        
        end_time = time.time()
        total_time = end_time - start_time
        
        filtered_documents = []
        for doc, score in results:
            if score >= threshold:
                filtered_documents.append({
                    'content': doc.page_content,
                    'similarity_score': score
                })
        
        return filtered_documents, total_time, embedding_time, search_time

def estimate_tokens(text):
    return len(text) // 2.4548

async def process_question(vectorstore, embeddings, question: str, question_id: int, retriever_k: int, 
                          reranker_k: int, reranker_threshold: float, retriever_threshold: float,
                          truncation_limit: int, mode: str, temperature: float, use_bm25: bool, bm25_index):
    print(f"Question ID: {question_id}")
    print(f"Question: {question}")
    print(f"Temperature: {temperature}")
    print(f"Using BM25: {use_bm25}")
    
    total_start_time = time.time()
    retriever_time = 0
    retriever_embedding_time = 0
    retriever_search_time = 0
    reranker_time = 0
    generation_time = 0
    embedding_tokens = 0
    reranker_tokens = 0
    generation_tokens = 0
    
    result_data = {
        'question_id': question_id,
        'question_text': question,
        'retriever_k': retriever_k,
        'reranker_k': reranker_k,
        'reranker_threshold': reranker_threshold,
        'retriever_threshold': retriever_threshold,
        'truncation_limit': truncation_limit,
        'mode': mode,
        'temperature': temperature,
        'use_bm25': use_bm25,
        'retriever_docs_count': 0,
        'retriever_top1_score': None,
        'reranker_returned_doc': False,
        'reranker_relevance_score': None,
        'answer_text': '',
        'retriever_time': 0,
        'retriever_embedding_time': 0,
        'retriever_search_time': 0,
        'reranker_time': 0,
        'generation_time': 0,
        'total_time': 0,
        'embedding_tokens': 0,
        'reranker_tokens': 0,
        'generation_tokens': 0
    }
    
    # Этап 1: Ретривер
    documents_with_scores, retriever_time, retriever_embedding_time, retriever_search_time = await basic_retrieval(
        vectorstore, embeddings, question, k=retriever_k, threshold=retriever_threshold,
        use_bm25=use_bm25, bm25_index=bm25_index
    )
    
    result_data['retriever_time'] = retriever_time
    result_data['retriever_embedding_time'] = retriever_embedding_time
    result_data['retriever_search_time'] = retriever_search_time
    
    if not use_bm25:
        embedding_tokens = estimate_tokens(question)
        result_data['embedding_tokens'] = embedding_tokens
        print(f"Estimated tokens for embedding: {embedding_tokens}")
    
    print(f"Retriever time: {retriever_time:.2f}s (embedding: {retriever_embedding_time:.2f}s, search: {retriever_search_time:.2f}s)")
    
    if documents_with_scores:
        result_data['retriever_docs_count'] = len(documents_with_scores)
        result_data['retriever_top1_score'] = documents_with_scores[0]['similarity_score']
        
        print(f"Retriever found {len(documents_with_scores)} documents (after threshold filtering)")
        for i, doc in enumerate(documents_with_scores):
            print(f"Document {i+1} similarity score: {doc['similarity_score']:.4f}")
        
        documents_content = [doc['content'] for doc in documents_with_scores]
        
        # Этап 2: Реранкер
        rerank_result, reranker_time = await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: rerank_docs(question, documents_content, EMBEDDER_API_KEY, 
                              top_k=reranker_k, threshold=reranker_threshold,
                              truncation_limit=truncation_limit)
        )
        result_data['reranker_time'] = reranker_time
        print(f"Reranker time: {reranker_time:.2f}s")
        
        reranker_input = question + " ".join(documents_content)
        reranker_tokens = estimate_tokens(reranker_input)
        result_data['reranker_tokens'] = reranker_tokens
        print(f"Estimated tokens for reranker: {reranker_tokens}")
        
        if rerank_result and len(rerank_result) > 0:
            selected_docs = rerank_result[:reranker_k]  # Берем столько документов, сколько запрошено
            
            result_data['reranker_returned_doc'] = True
            result_data['reranker_relevance_score'] = selected_docs[0]['relevance_score']  # Score самого релевантного
            result_data['reranker_docs_count'] = len(selected_docs)  # Количество документов от реранкера
            
            print(f"Reranker selected {len(selected_docs)} documents:")
            for i, doc in enumerate(selected_docs):
                print(f"Document {i+1} relevance score: {doc['relevance_score']:.4f}")
            
            # Объединяем содержимое всех выбранных документов
            combined_content = "\n\n".join([doc['content'] for doc in selected_docs])
            print(combined_content)
            system_prompt = "Use the following context as your learned knowledge. When answering the user: - If you don't know the answer, simply state that you don't know. - If you're unsure, seek clarification. - Avoid mentioning that the information was sourced from the context. - Respond in accordance with the language of the user's question."
            
            user_content = f"Documents:\n{combined_content}\n\nQuestion: {question}"
            
            generation_tokens = estimate_tokens(system_prompt + user_content)
            result_data['generation_tokens'] = generation_tokens
            print(f"Estimated tokens for generation: {generation_tokens}")
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            
            # Этап 3: Генерация ответа с температурой
            generation_start_time = time.time()
            answer = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: answer_generation(messages, mode, temperature)
            )
            generation_time = time.time() - generation_start_time
            result_data['generation_time'] = generation_time
            print(f"Generation time: {generation_time:.2f}s")
            
            result_data['answer_text'] = answer
            print(f"Answer: {answer}")
        else:
            print("Reranker returned no documents (below threshold or error)")
            # Fallback обработка
            system_prompt = """**Generate Response to User Query**
**Step 1: Parse Context Information**
Extract and utilize relevant knowledge from the provided context.
**Step 2: Analyze User Query**
Carefully read and comprehend the user's query, pinpointing the key concepts, entities, and intent behind the question.
**Step 3: Determine Response**
If the answer to the user's query can be directly inferred from the context information, provide a concise and accurate response in the same language as the user's query.
**Step 4: Handle Uncertainty**
If the answer is not clear, ask the user for clarification to ensure an accurate response.
**Step 5: Avoid Context Attribution**
When formulating your response, do not indicate that the information was derived from the context.
**Step 6: Respond in User's Language**
Maintain consistency by ensuring the response is in the same language as the user's query.
**Step 7: Provide Response**
Generate a clear, concise, and informative response to the user's query."""
            
            generation_tokens = estimate_tokens(system_prompt + question)
            result_data['generation_tokens'] = generation_tokens
            
            generation_start_time = time.time()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
            answer = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: answer_generation(messages, mode, temperature)
            )
            print(f"Answer: {answer}")
            generation_time = time.time() - generation_start_time
            result_data['generation_time'] = generation_time
            
            result_data['answer_text'] = answer
    else:
        print("No documents found by retriever (all below threshold)")
        # Fallback обработка
        system_prompt = """
        **Generate Response to User Query**
**Step 1: Analyze User Query**
Carefully read and comprehend the user's query, pinpointing the key concepts, entities, and intent behind the question.
**Step 2: Determine if the user query relates directly or indirectly to finance**
Determine if the user query relates directly or indirectly to finance. If yes, provide a concise and accurate response. If the question does not relate directly or indirectly to finance, refuse to answer the question. Answer in the same language as the user's query."
**Step 3: Handle Uncertainty**
If the answer is not clear, ask the user for clarification to ensure an accurate response.
**Step 4: Avoid Context Attribution**
When formulating your response, do not indicate that the information was derived from the context.
**Step 5: Respond in User's Language**
Maintain consistency by ensuring the response is in the same language as the user's query.
**Step 6: Provide Response**
Generate a clear, concise, and informative response to the user's query."""
        
        generation_tokens = estimate_tokens(system_prompt + question)
        result_data['generation_tokens'] = generation_tokens
        
        generation_start_time = time.time()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        answer = await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: answer_generation(messages, mode, temperature)
        )
        generation_time = time.time() - generation_start_time
        result_data['generation_time'] = generation_time
        
        result_data['answer_text'] = answer
    
    total_time = time.time() - total_start_time
    result_data['total_time'] = total_time
    print(f"Total processing time: {total_time:.2f}s")
    
    return result_data

async def main():
    parser = argparse.ArgumentParser(description='Financial QA System')
    parser.add_argument('--retriever_k', type=int, default=5, 
                       help='Number of documents for retriever to send to reranker')
    parser.add_argument('--reranker_k', type=int, default=1, 
                       help='Number of documents for reranker to return')
    parser.add_argument('--reranker_threshold', type=float, default=0.0, 
                       help='Minimum relevance score for reranker documents')
    parser.add_argument('--retriever_threshold', type=float, default=0.0, 
                       help='Minimum similarity score for retriever documents')
    parser.add_argument('--truncation_limit', type=int, default=1500, 
                       help='Maximum document length for reranker')
    parser.add_argument('--n_questions', type=int, default=1, 
                       help='Number of questions to process')
    parser.add_argument('--mode', type=str, default='prod', choices=['prod', 'test'], 
                       help='Mode: prod (hackathon API) or test (OpenRouter)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature for LLM generation (0.0 to 1.0)')
    parser.add_argument('--use_bm25', action='store_true',
                       help='Use BM25 instead of embedding-based retrieval')
    parser.add_argument('--question_id', type=int, default=None,
                       help='Process specific question ID (overrides n_questions)')
    
    args = parser.parse_args()
    
    # Если указан конкретный ID вопроса, игнорируем n_questions
    if args.question_id is not None:
        args.n_questions = 1
        print(f"Processing specific question ID: {args.question_id} (n_questions ignored)")
    
    print(f"Parameters: retriever_k={args.retriever_k}, reranker_k={args.reranker_k}, "
          f"reranker_threshold={args.reranker_threshold}, retriever_threshold={args.retriever_threshold}, "
          f"truncation_limit={args.truncation_limit}, n_questions={args.n_questions}, mode={args.mode}, "
          f"temperature={args.temperature}, use_bm25={args.use_bm25}, question_id={args.question_id}")
    
    questions_df = pd.read_csv('questions.csv')
    print(f"Loaded {len(questions_df)} questions")
    
    processed_questions = set()
    if os.path.exists('train_results.csv'):
        existing_results = pd.read_csv('train_results.csv')
        print(f"Found existing results with {len(existing_results)} records")
        
        for _, row in existing_results.iterrows():
            if (row['retriever_k'] == args.retriever_k and
                row['reranker_k'] == args.reranker_k and
                row['reranker_threshold'] == args.reranker_threshold and
                row['retriever_threshold'] == args.retriever_threshold and
                row['truncation_limit'] == args.truncation_limit and
                row['mode'] == args.mode and
                row.get('temperature', 0.7) == args.temperature and
                row.get('use_bm25', False) == args.use_bm25):
                processed_questions.add(row['question_id'])
    
    # Фильтрация вопросов для обработки
    if args.question_id is not None:
        # Обрабатываем конкретный ID вопроса
        if args.question_id in questions_df.index:
            sample_questions = questions_df.loc[[args.question_id]]
            print(f"Processing specific question ID: {args.question_id}")
        else:
            print(f"Question ID {args.question_id} not found in dataset")
            return
    else:
        # Обычная логика - фильтруем уже обработанные вопросы
        available_questions = questions_df[~questions_df.index.isin(processed_questions)]
        print(f"Available questions after filtering: {len(available_questions)}")
        
        if len(available_questions) == 0:
            print("No new questions to process with current parameters")
            return
        
        sample_questions = available_questions.sample(n=min(args.n_questions, len(available_questions)))
    
    try:
        embeddings = CustomEmbeddings(args.mode)
        vectorstore = InMemoryVectorStore.load('my_vdb.db', embeddings)
        print("Vector store loaded")
        
        # Инициализация BM25 индекса если нужно
        bm25_index = None
        if args.use_bm25:
            print("Building BM25 index...")
            # Извлекаем все документы из векторного хранилища
            all_documents = []
            for doc_id, doc_data in vectorstore.store.items():
                all_documents.append(doc_data['text'])
            
            bm25_index = SimpleBM25()
            bm25_index.fit(all_documents)
            print(f"BM25 index built with {len(all_documents)} documents")
            
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return
    
    all_results = []
    for i, (idx, row) in enumerate(sample_questions.iterrows()):
        question = row['Вопрос']
        question_id = idx
        
        print(f"\nProcessing {i+1}/{len(sample_questions)}")
        
        result = await process_question(vectorstore, embeddings, question, question_id, args.retriever_k, 
                                      args.reranker_k, args.reranker_threshold, args.retriever_threshold,
                                      args.truncation_limit, args.mode, args.temperature, args.use_bm25, bm25_index)
        all_results.append(result)
        
        await asyncio.sleep(0.2)
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        if os.path.exists('train_results.csv'):
            existing_df = pd.read_csv('train_results.csv')
            final_df = pd.concat([existing_df, results_df], ignore_index=True)
        else:
            final_df = results_df
        
        final_df.to_csv('train_results.csv', index=False)
        print(f"\nSaved {len(results_df)} results to train_results.csv")
    
    print(f"Processing completed. Processed {len(all_results)} questions")

if __name__ == "__main__":
    asyncio.run(main())