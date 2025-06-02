# 🚀 Быстрый запуск бота

## Что это?
Telegram-бот для суммаризации диалогов с помощью GPT-4.1-mini от OpenAI.

## 📋 Что нужно?
1. **Токен Telegram бота** (от @BotFather)
2. **API ключ OpenAI** (с platform.openai.com)
3. **Python 3.8+**

## ⚡ Быстрый старт

### 1. Настройка
```bash
# Переименуйте файл env_example.txt в .env
cp env_example.txt .env

# Отредактируйте .env и добавьте ваши токены:
# TELEGRAM_BOT_TOKEN=ваш_токен_от_botfather
# OPENAI_API_KEY=ваш_ключ_от_openai
```

### 2. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 3. Запуск

**Вариант A: Python скрипт (рекомендуется)**
```bash
python3 telegram_bot.py
```

**Вариант B: Jupyter Notebook**
```bash
jupyter notebook telegram_bot_summarizer.ipynb
```

## 🎯 Команды бота
- `/start` - Начать работу
- `/summarize` - Создать пересказ диалога  
- `/clear` - Очистить сообщения
- `/count` - Показать количество сообщений

## 💡 Как использовать
1. Запустите бота
2. Найдите его в Telegram 
3. Напишите `/start`
4. Отправьте несколько сообщений
5. Используйте `/summarize` для получения пересказа

## 💰 Стоимость
Примерно $0.001-0.005 за одну суммаризацию (очень дешево!)

## 🔧 Проблемы?
- Проверьте токены в `.env` файле
- Убедитесь что есть средства на OpenAI аккаунте
- Перезапустите бота

---
📖 Подробная документация в файле README.md 