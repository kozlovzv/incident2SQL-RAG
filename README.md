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
git clone git@github.com:kozlovzv/incident2SQL-RAG.git
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
     -d '{"text": "Show high risk incidents"}'
```

Пример ответа:
```json
{
    "status": "success",
    "data": {
        "query": "Show high risk incidents",
        "sql": "SELECT i.*, c.name as category_name, c.description as category_description FROM incidents i LEFT JOIN categories c ON i.category_id = c.id WHERE risk_level >= 0.8 ORDER BY risk_level DESC LIMIT 10",
        "results": [...],
        "context": []
    },
    "metadata": {
        "total_results": 10,
        "execution_time": 0.001
    }
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
# Все тесты (27 тестов)
pytest

# Конкретные модули
pytest tests/api/test_main.py          # API endpoint тесты
pytest tests/models/test_text2sql.py   # Text2SQL генератор
pytest tests/models/test_complex_queries.py  # Сложные SQL запросы
pytest tests/rag/test_retriever.py     # RAG функциональность
```

### Покрытие тестами
- API endpoints и маршрутизация:
  - Health check endpoint
  - Query endpoint с валидацией структуры ответа
  - Обработка ошибок и edge cases
- Генерация SQL-запросов:
  - Базовые запросы
  - Сложные условия фильтрации
  - Сортировка и лимиты
- RAG функциональность:
  - Индексация документов
  - Поиск релевантного контекста
  - Оптимизация результатов
- Мониторинг производительности
  - Замер времени выполнения
  - Подсчет результатов
  - Метаданные запросов

## Локальная разработка

### 1. Подготовка окружения
```bash
# Клонируем репозиторий
git clone git@github.com:kozlovzv/incident2SQL-RAG.git
cd text2SQL-RAG

# Создаем и активируем виртуальное окружение
python -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate

# Устанавливаем зависимости
pip install -r requirements.txt
```

### 2. Запуск сервера
```bash
# Запускаем сервер с автоперезагрузкой
uvicorn src.api.main:app --reload

# Сервер запустится на http://localhost:8000
```

### 3. Проверка работоспособности

#### Проверка статуса сервера
```bash
curl http://localhost:8000/health
```

Ожидаемый ответ:
```json
{
    "status": "healthy",
    "components": {
        "database": "ready",
        "text2sql": "ready",
        "retriever": "ready"
    }
}
```

#### Тестовые запросы

1. Базовый запрос инцидентов с высоким риском:
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"text": "Show high risk incidents"}' \
     -s | python -m json.tool
```

2. Запрос инцидентов за последнюю неделю:
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"text": "Show incidents from last week"}' \
     -s | python -m json.tool
```

3. Запрос критических инцидентов определенной категории:
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"text": "Show critical server crash incidents"}' \
     -s | python -m json.tool
```

### 4. Проверка результатов

Каждый запрос возвращает JSON со следующей структурой:
```json
{
    "status": "success",
    "data": {
        "query": "исходный запрос",
        "sql": "сгенерированный SQL-запрос",
        "results": [
            {
                "id": "номер инцидента",
                "title": "название",
                "description": "описание",
                "date_occurred": "дата",
                "risk_level": "уровень риска",
                "status": "статус",
                "category_name": "название категории",
                "category_description": "описание категории"
            }
        ],
        "context": []
    },
    "metadata": {
        "total_results": "количество результатов",
        "execution_time": "время выполнения в секундах"
    }
}
```

### 5. Запуск тестов
```bash
# Все тесты (27 тестов)
pytest

# Конкретные модули
pytest tests/api/test_main.py          # API endpoint тесты
pytest tests/models/test_text2sql.py   # Text2SQL генератор
pytest tests/models/test_complex_queries.py  # Сложные SQL запросы
pytest tests/rag/test_retriever.py     # RAG функциональность
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## Лицензия

MIT