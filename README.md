# Text2SQL + RAG для работы с инцидентами

Система, которая конвертирует запросы на естественном языке (включая русский) в SQL-запросы, используя RAG (Retrieval Augmented Generation) для предоставления релевантного контекста из базы инцидентов.

## Особенности

- ✨ Поддержка русского и английского языков
- 🔍 RAG (Retrieval Augmented Generation) для улучшения точности запросов
- 🛡️ Встроенная защита от SQL-инъекций
- 🚀 FastAPI веб-сервис
- 📊 Работа с базой данных инцидентов (SQLite)
- 💡 Умная обработка сложных запросов (фильтрация по риску, дате, категории)

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

## Структура проекта

```
├── api/              # API определения и роуты
├── data/            # База данных и образцы данных
├── docs/            # Дополнительная документация
├── models/          # Модели Text2SQL и работа с БД
├── src/             # Исходный код
│   ├── api/         # FastAPI приложение
│   ├── models/      # Модели и работа с базой данных
│   ├── rag/         # Компоненты RAG
│   └── run_demo.py  # Демо-скрипт
└── tests/           # Тесты
```

## Использование

1. Запустите API сервер:
```bash
uvicorn src.api.main:app --reload
```

2. API будет доступно по адресу `http://localhost:8000`

3. Swagger документация доступна по адресу: `http://localhost:8000/docs`

### Примеры запросов

```bash
# Получить все инциденты высокого риска
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"text": "Покажи все инциденты с высоким риском"}'

# Показать сетевые инциденты за последнюю неделю
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"text": "Show network incidents from last week"}'
```

## Схема базы данных

### Таблица `incidents`
- `id` (PK)
- `title` - Заголовок инцидента
- `description` - Описание инцидента
- `date_occurred` - Дата возникновения
- `risk_level` - Уровень риска (0.0-1.0)
- `status` - Статус инцидента
- `category_id` (FK) - Связь с категорией

### Таблица `categories`
- `id` (PK)
- `name` - Название категории
- `description` - Описание категории

## Разработка

### Запуск тестов
```bash
pytest tests/
```

### Проверка стиля кода
```bash
black src/ tests/
isort src/ tests/
```

## Возможности Text2SQL

- Поддержка сложных условий фильтрации
- Сортировка по различным параметрам
- Обработка временных периодов (последняя неделя, месяц, квартал)
- Фильтрация по уровню риска
- Лимитирование результатов
- Поддержка русского и английского языков

## Примеры поддерживаемых запросов

- "Покажи все инциденты с высоким риском"
- "Show network incidents from last week"
- "Отобрази 5 последних инцидентов"
- "Find critical incidents in Q2 2024"
- "Покажи инциденты со статусом resolved, сортируй по риску"
- "Show incidents with risk level above 0.8"

## Безопасность

- Встроенная защита от SQL-инъекций
- Валидация входных данных
- Безопасная обработка запросов к базе данных

## Лицензия

MIT

## Тестирование системы

### Автоматические тесты
```bash
# Запуск всех тестов с подробным выводом
pytest tests/ -v

# Запуск конкретных тестов
pytest tests/api/test_main.py
pytest tests/models/test_text2sql.py
pytest tests/models/test_complex_queries.py
pytest tests/rag/test_retriever.py
```

### Тестирование через консоль

1. Запустите demo-скрипт для проверки основной функциональности:
```bash
python src/run_demo.py
```

2. Тестирование через API:

Сначала запустите сервер:
```bash
uvicorn src.api.main:app --reload
```

Затем выполните тестовые запросы:
```bash
# Проверка работоспособности сервера
curl http://localhost:8000/health

# Тестовые запросы (с форматированным выводом)
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"text": "Show critical incidents from last month with high risk"}' \
     | python -m json.tool

# Запрос на русском языке
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"text": "Покажи критичные инциденты за последний месяц"}' \
     | python -m json.tool
```

### Покрытие тестами

Тесты охватывают следующие аспекты:
- API endpoints (/health, /query)
- Генерация SQL запросов
- Обработка сложных условий (фильтрация, сортировка, лимиты)
- Поддержка русского языка
- Защита от SQL-инъекций
- RAG функциональность (индексация и поиск документов)
- Работу с базой данных