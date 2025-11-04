# Функция реранкера
1) Векторная база подгружается из my_vdb.db локально, ембеддер используется только для векторизации user query.
2) Ретривер из ноутбука Коли, ничего не трогал. Разве что использовал similarity_search_with_score, чтобы получить оценку в виде метрики.
3) Реранкер используется deepinfra/Qwen/Qwen3-Reranker-4B из ai-for-finance-hack (т.к. на openrouter такой же модели нет, а из hf локально пока решил не поднимать, т.к. есть ключ)

## Запуск
```python
python main.py --retriever_k 8 --reranker_k 1 --reranker_threshold 0.3 --retriever_threshold 0.6 --truncation_limit 1200 --n_questions 4 --mode prod --use_bm25 --question_id 299 --temperature 0.2
```
1. **--retriever_k** Количество документов для ретривера
2. **--reranker_k** Количество документов для реранкера (пофиксил reranker_k > 1)
3. **--reranker_threshold** Порог релевантности для реранкера
4. **--retriever_threshold** Порог сходства для ретривера
5. **--truncation_limit** Максимальная длина документа для реранкера
6. **--n_questions** Количество вопросов для обработки (от 1 до 500)
7. **--mode** prod - будут использованы ключи LLM_API_KEY и EMBEDDER_API_KEY из ai-for-finance-hack, test - будут использованы ключи OPENROUTER_API_KEY (для всего, кроме реранкера) и EMBEDDER_API_KEY (только для реранкера)
8. **--question_id** ID конкретного вопроса, если указан, то n_questions будет проигнорирован (всегда = 1)
9. **--use_bm25** использовать альтернативный ретривер bm25 вместо базового ретривера (реализован без сторонней либы)
10. **--temperature** задаем температуру для генеративной модели

```python
load_dotenv()
LLM_API_KEY = os.getenv("LLM_API_KEY")
EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
```
### Доп инфа
https://medium.com/@kelvincampelo/how-ive-optimized-document-interactions-with-open-webui-and-rag-a-comprehensive-guide-65d1221729eb
https://www.reddit.com/r/OpenWebUI/comments/1krzvdm/best_system_and_rag_prompts/

Результаты сохраняются в train_results.csv, при запуске чекает файл и если там уже есть вопросы (по ID вопроса в questions.csv) с такими же параметрами argparse, то скипает его. 
Каждый результат обработки вопроса содержит:
1) все аргументы argparse
2) количество токенов исходя из 4 символа ~ 1 токен (плюс сумма токенов)
3) время обработки каждого этапа (плюс суммарное время)
4) текст вопроса, текст системного промпта, текст ответа
5) Пример вывода в консоль:
```python
Processing 108/500
Question ID: 242
Question: Почему повторяются финансовые пузыри?
Retriever time: 3.65s (embedding: 1.17s, search: 2.49s)
Estimated tokens for embedding: 9
Retriever found 5 documents (after threshold filtering)
Document 1 similarity score: 0.4586
Document 2 similarity score: 0.3876
Document 3 similarity score: 0.3857
Document 4 similarity score: 0.3849
Document 5 similarity score: 0.3810
{'id': 'Raq10INy090J3Inp5JlSG78L', 'results': [{'index': 0, 'relevance_score': 0.0002959570847451687}, {'index': 1, 'relevance_score': 1.67014204635052e-05}, {'index': 2, 'relevance_score': 0.0020507434383034706}, {'index': 3, 'relevance_score': 0.000626334105618298}, {'index': 4, 'relevance_score': 8.481103577651083e-05}], 'meta': {'billed_units': {'total_tokens': 1685}, 'tokens': {'input_tokens': 1685, 'output_tokens': 0}}}
Reranker time: 1.24s
Estimated tokens for reranker: 1031
Reranker returned no documents (below threshold or error)
Estimated tokens for generation: 255
System prompt: **Generate Response to User Query**
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
Generate a clear, concise, and informative response to the user's query.
Generation time: 11.55s
Answer: Финансовые пузыри повторяются из-за сочетания нескольких факторов, которые являются неотъемлемой частью экономических систем. Вот основные причины:

1. **Психология инвесторов**: Жадность, страх и коллективное поведение (эффект толпы) заставляют людей игнорировать риски и следовать за растущими ценами.

2. **Экономическая политика**: Либеральные меры, такие как низкие процентные ставки, поощряют чрезмерное заимствование и спекуляции.

3. **Технологические инновации**: Новые технологии часто создают ажиотаж, что приводит к переоценке активов, как это было в случае с дотком-бума.

4. **Провал регуляции**: Недостаточный надзор позволяет пузырям развиваться без контроля.

5. **Рыночные неэффективности**: Асимметрия информации и иррациональное поведение участников рынка приводят к неверной оценке активов.

Эти факторы сохраняются со временем, что делает повторение пузырей неизбежным, несмотря на меры предосторожности.
Total processing time: 16.45s
```
