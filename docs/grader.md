## Запуск
```python
python grader.py --input_path=submission.csv --output_path=grade_result.csv --grading_model=qwen/qwen3-vl-8b-instruct --baseline_path=ethalon_submission.csv --grade_threshold=1
```
1. **--input_path** Путь/название ответов для оценки
2. **--output_path** Путь/название ответов для результата
3. **--baseline_path** Путь/название ответов для сравнения (baseline)
4. **--grading_threshold** Пороговая оценка (не используется при сравнении)
5. **--grading_model** Модель для оценки (из списка доступных в Openrouter)

При сравнении важно, чтобы количество ответов в input_path и baseline_path совпадали
Для работы скрипта используется ключ от Openrouter:

```python
load_dotenv()
LLM_API_KEY = os.getenv("LLM_API_KEY")
```
Пример вывода в консоль:
```python
Результаты:
Оценивающая модель: qwen/qwen3-vl-8b-instruct
Всего вопросов: 10
Количество ответов с оценкой ниже порога 1.0: 0
   • Ответы: []
time taken:  17.45370221000121
```