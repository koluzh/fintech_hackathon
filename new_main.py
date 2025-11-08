import pandas as pd
from openai import OpenAI
import numpy as np
import faiss
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
import time
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
LLM_API_KEY = os.getenv("LLM_API_KEY")
EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")
# Configuration
LLM_BASE_URL = "https://ai-for-finance-hack.up.railway.app/"
EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "openrouter/x-ai/grok-3-mini"
TOP_K_RETRIEVAL = 20
TOP_K_DOCS = 2  # Number of documents to use for generation
MAX_WORKERS = 10
MAX_RETRIES = 20
RETRY_DELAY = 5
REQUEST_TIMEOUT = 90
BM25_WEIGHT = 0.3
DENSE_WEIGHT = 0.7


def sanitize_input(text: str) -> str:
    """Sanitize input to prevent prompt injection"""
    if not text or not isinstance(text, str):
        return ""

    text = text.strip()
    text = re.sub(r'\n{3,}', '\n\n', text)

    suspicious_patterns = [
        r'(?i)(ignore|forget|disregard|skip)\s+(previous|above|all|prior)\s+(instructions?|prompts?|context)',
        r'(?i)system\s*:',
        r'(?i)assistant\s*:',
        r'(?i)user\s*:',
        r'(?i)(you\s+are|ты\s+теперь|ты\s+должен)\s+(now|теперь)',
    ]

    for pattern in suspicious_patterns:
        text = re.sub(pattern, '', text)

    max_length = 500
    if len(text) > max_length:
        text = text[:max_length]

    return text


class RAGSystem:
    def __init__(self):
        self.embedding_client = OpenAI(
            base_url=LLM_BASE_URL,
            api_key=EMBEDDER_API_KEY,
            timeout=REQUEST_TIMEOUT
        )
        self.generation_client = OpenAI(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            timeout=REQUEST_TIMEOUT
        )
        self.documents = []
        self.document_embeddings = []
        self.index = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

    def load_documents(self, filepath: str):
        """Load documents from CSV file"""
        print(f"Loading documents from {filepath}...")
        df = pd.read_csv(filepath)
        self.documents = []

        for _, row in df.iterrows():
            tags = str(row['tags']) if 'tags' in row else ''
            annotation = str(row['annotation']) if pd.notna(row['annotation']) else ''
            searchable_text = f"{annotation}\n\nТеги: {tags}"
            full_text = str(row['text'])

            self.documents.append({
                'id': row['id'],
                'searchable_text': searchable_text,
                'full_text': full_text,
                'annotation': annotation
            })

        print(f"Loaded {len(self.documents)} documents")
        return self.documents

    def get_embedding(self, text: str, retries=MAX_RETRIES) -> List[float]:
        """Get embedding for a single text with retry logic"""
        for attempt in range(retries):
            try:
                response = self.embedding_client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=text
                )
                return response.data[0].embedding
            except Exception as e:
                if attempt < retries - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    time.sleep(wait_time)
                else:
                    return None
        return None

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Get embeddings for multiple texts with batching"""
        embeddings = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                for text in batch:
                    futures.append(executor.submit(self.get_embedding, text))

            for future in tqdm(futures, desc="Getting embeddings"):
                result = future.result()
                if result is not None:
                    embeddings.append(result)

        return embeddings

    def build_index(self):
        """Build FAISS index and TF-IDF for hybrid search"""
        print("Building indexes...")
        doc_texts = [doc['searchable_text'] for doc in self.documents]

        # Get embeddings
        print("Generating embeddings...")
        self.document_embeddings = self.get_embeddings_batch(doc_texts)

        # Build FAISS index
        print("Building FAISS index...")
        embeddings_array = np.array(self.document_embeddings).astype('float32')
        dimension = embeddings_array.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings_array)
        self.index.add(embeddings_array)
        print(f"FAISS index built with {self.index.ntotal} vectors")

        # Build TF-IDF index
        print("Building TF-IDF index...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9,
            sublinear_tf=True
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(doc_texts)
        print(f"TF-IDF index built")

    def retrieve_documents(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict]:
        """Hybrid retrieval: Combine BM25 + Dense search"""
        query_embedding = self.get_embedding(query)
        if query_embedding is None:
            return []

        # Dense retrieval
        query_vector = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_vector)
        dense_scores, dense_indices = self.index.search(query_vector, top_k * 2)

        # BM25 retrieval
        query_tfidf = self.tfidf_vectorizer.transform([query])
        tfidf_scores = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()

        # Combine scores
        combined_scores = {}

        for idx, score in zip(dense_indices[0], dense_scores[0]):
            if idx < len(self.documents):
                combined_scores[idx] = DENSE_WEIGHT * float(score)

        for idx, score in enumerate(tfidf_scores):
            if idx in combined_scores:
                combined_scores[idx] += BM25_WEIGHT * float(score)
            else:
                combined_scores[idx] = BM25_WEIGHT * float(score)

        # Sort and get top-k
        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        retrieved_docs = []
        for idx, score in sorted_indices:
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['score'] = float(score)
                retrieved_docs.append(doc)

        return retrieved_docs

    def generate_answer(self, question: str, context_docs: List[Dict], retries=MAX_RETRIES) -> str:
        """Generate answer using retrieved context"""
        sanitized_question = sanitize_input(question)
        context = "\n\n---\n\n".join([doc['full_text'] for doc in context_docs])

        prompt = f"""Используя следующую информацию, дай структурированный ответ на вопрос.

Информация:
{context}

Вопрос: {sanitized_question}

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

        for attempt in range(retries):
            try:
                response = self.generation_client.chat.completions.create(
                    model=GENERATION_MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ],
                    temperature=0.3,
                    max_tokens=3200
                )

                answer = response.choices[0].message.content

                if not answer or len(answer.strip()) == 0:
                    raise ValueError("Generation returned empty answer")

                return answer

            except Exception as e:
                if attempt < retries - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    time.sleep(wait_time)
                else:
                    return "### Ошибка генерации ответа\n\nИзвините, не удалось сгенерировать ответ из-за технических проблем. Пожалуйста, попробуйте позже."

        return "### Ошибка генерации ответа\n\nИзвините, не удалось сгенерировать ответ из-за технических проблем. Пожалуйста, попробуйте позже."

    def process_question(self, question: str) -> str:
        """Complete RAG pipeline for a single question"""
        try:
            retrieved_docs = self.retrieve_documents(question, TOP_K_RETRIEVAL)

            if not retrieved_docs:
                return self.generate_answer(question, [])

            # Use top 2 documents
            top_docs = retrieved_docs[:TOP_K_DOCS]
            answer = self.generate_answer(question, top_docs)

            if not answer or not isinstance(answer, str):
                return "### Ошибка генерации\n\nНе удалось сгенерировать ответ на данный вопрос."

            return answer
        except Exception as e:
            return "### Критическая ошибка\n\nПроизошла критическая ошибка при обработке вопроса."

    def process_questions_parallel(self, questions: List[str]) -> List[str]:
        """Process multiple questions with parallel execution"""
        answers = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(self.process_question, q) for q in questions]

            for i, future in enumerate(futures):
                try:
                    answer = future.result()
                    if answer is None or not isinstance(answer, str) or len(answer.strip()) == 0:
                        answer = "### Ошибка генерации ответа\n\nИзвините, не удалось сгенерировать ответ. Пожалуйста, попробуйте переформулировать вопрос."
                    answers.append(answer)
                    if i < len(futures) - 1:
                        time.sleep(0.5)
                except Exception as e:
                    answers.append("### Ошибка обработки вопроса\n\nПроизошла ошибка при обработке вопроса. Пожалуйста, попробуйте позже.")

        return answers


def main():
    rag = RAGSystem()
    rag.load_documents('./train_data.csv')
    rag.build_index()
    print("\nLoading questions...")
    questions_df = pd.read_csv('./questions.csv')
    print(f"Loaded {len(questions_df)} questions")

    test_questions = questions_df
    questions = test_questions['Вопрос'].tolist()

    print(f"\nProcessing {len(questions)} question(s)...")
    answers = rag.process_questions_parallel(questions)

    result_df = test_questions.copy()
    result_df['Ответы на вопрос'] = answers
    result_df.to_csv('submission.csv', index=False)

    print(f"\nDone! Processed {len(answers)} questions")
    print(f"Results saved to submission.csv")


if __name__ == "__main__":
    main()
