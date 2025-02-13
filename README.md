# Text2SQL + RAG для работы с инцидентами

Система, которая конвертирует запросы на естественном языке (включая русский) в SQL-запросы, используя RAG (Retrieval Augmented Generation) для предоставления релевантного контекста из базы инцидентов.

## Особенности

- ✨ Поддержка русского и английского языков
- 🔍 RAG (Retrieval Augmented Generation) для улучшения точности запросов с динамическими порогами релевантности
- 🛡️ Встроенная защита от SQL-инъекций
- 📊 Умная обработка сложных запросов:
  - Фильтрация по риску, дате, категории, статусу
  - Сортировка по различным параметрам
  - Динамические лимиты результатов
  - Поддержка временных периодов (неделя, месяц, квартал)
- 🚀 Оптимизированный FastAPI веб-сервис
- 💾 Эффективное индексирование с использованием FAISS
- 📈 Расширенный мониторинг производительности
- 🧪 Комплексное тестовое покрытие

## Установка

1. Клонируйте репозиторий:
```bash
git clone <your-repo-url>
cd text2SQL-RAG
```

2. Создайте виртуальное окружение:
```bash
python -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Использование API

### Запуск сервера
```bash
uvicorn src.api.main:app --reload
```

### API Endpoints

#### 1. Запрос к базе данных (POST /query)
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"text": "Покажи критичные инциденты за последний месяц"}'
```

Пример ответа:
```json
{
    "sql_query": "SELECT i.*, c.name as category_name FROM incidents i LEFT JOIN categories c ON i.category_id = c.id WHERE risk_level >= 0.8 AND date_occurred >= date('now', '-1 month') ORDER BY risk_level DESC LIMIT 10",
    "results": [...],
    "context": [...]
}
```

#### 2. Проверка статуса (GET /health)
```bash
curl http://localhost:8000/health
```

## Примеры поддерживаемых запросов

### Базовые запросы
- "Покажи все инциденты с высоким риском"
- "Show network incidents from last week"
- "Отобрази 5 последних инцидентов"

### Сложные запросы
- "Покажи критичные инциденты категории server crash за последний месяц с риском выше 0.8"
- "Find all incidents in Q2 2024 with resolved status"
- "Отобрази инциденты с риском выше 0.7, сортируй по дате"

### Временные периоды
- "Покажи инциденты за последнюю неделю"
- "Show incidents from Q1 2024"
- "Инциденты за последний месяц со статусом resolved"

## Мониторинг и производительность

Система включает расширенный модуль мониторинга (RAGMonitor), который отслеживает:

- Латентность запросов
- Качество результатов поиска
- Производительность кэша
- Метрики по часам
- Динамические пороги релевантности
- Статистику ошибок

## Оптимизации

### RAG Retriever
- Динамическое определение порогов релевантности
- Кэширование эмбеддингов
- Оптимизированный IVF индекс FAISS
- Умная обработка чанков документов

### Text2SQL
- Улучшенная обработка русского языка
- Оптимизированная генерация SQL
- Предотвращение SQL-инъекций
- Поддержка сложных условий и сортировок

## Тестирование

### Запуск тестов
```bash
# Все тесты
pytest tests/ -v

# Конкретные модули
pytest tests/api/test_main.py
pytest tests/models/test_text2sql.py
pytest tests/rag/test_retriever.py
```

### Покрытие тестами
- API endpoints и маршрутизация
- Генерация SQL-запросов
- RAG функциональность
- Обработка сложных условий
- Защита от SQL-инъекций
- Мониторинг производительности

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## Лицензия

MIT