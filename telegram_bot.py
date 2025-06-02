#!/usr/bin/env python3
"""
Telegram Bot для суммаризации диалогов
Использует GPT-4.1-mini от OpenAI для создания пересказов диалогов
"""

import os
import json
import logging
from datetime import datetime
from collections import defaultdict
from typing import Dict, List
import asyncio

from dotenv import load_dotenv
from openai import OpenAI
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Конфигурация
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not TELEGRAM_BOT_TOKEN or not OPENAI_API_KEY:
    print("❌ Необходимо установить TELEGRAM_BOT_TOKEN и OPENAI_API_KEY")
    exit(1)

# Инициализация OpenAI клиента
client = OpenAI(api_key=OPENAI_API_KEY)

# Хранилище сообщений для каждого чата
chat_messages: Dict[int, List[Dict]] = defaultdict(list)

# Максимальное количество сообщений для хранения
MAX_MESSAGES_PER_CHAT = 100


class DialogueSummarizerBot:
    def __init__(self):
        self.client = client
        self.chat_messages = chat_messages
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик команды /start"""
        welcome_message = (
            "🤖 Привет! Я бот для суммаризации диалогов.\n\n"
            "📝 Я буду сохранять ваши сообщения и могу создать их краткий пересказ.\n\n"
            "📋 Доступные команды:\n"
            "/start - Показать это сообщение\n"
            "/summarize - Создать пересказ всех сообщений\n"
            "/clear - Очистить сохранённые сообщения\n"
            "/count - Показать количество сохранённых сообщений\n\n"
            "💬 Просто начните писать сообщения, и я буду их сохранять!"
        )
        await update.message.reply_text(welcome_message)
    
    async def save_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Сохранение обычных сообщений"""
        chat_id = update.effective_chat.id
        user = update.effective_user
        message_text = update.message.text
        
        # Создаём запись о сообщении
        message_data = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user.id,
            'username': user.username or user.first_name,
            'text': message_text
        }
        
        # Добавляем сообщение в хранилище
        self.chat_messages[chat_id].append(message_data)
        
        # Ограничиваем количество сохранённых сообщений
        if len(self.chat_messages[chat_id]) > MAX_MESSAGES_PER_CHAT:
            self.chat_messages[chat_id] = self.chat_messages[chat_id][-MAX_MESSAGES_PER_CHAT:]
        
        logger.info(f"Сохранено сообщение от {user.username} в чате {chat_id}")
    
    async def summarize_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик команды /summarize"""
        chat_id = update.effective_chat.id
        
        if not self.chat_messages[chat_id]:
            await update.message.reply_text(
                "📭 Нет сохранённых сообщений для суммаризации.\n"
                "Напишите несколько сообщений, а затем используйте /summarize"
            )
            return
        
        # Показываем индикатор "печатает"
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        
        try:
            # Формируем текст для суммаризации
            messages_text = self._format_messages_for_summary(self.chat_messages[chat_id])
            
            # Получаем суммаризацию от GPT-4.1-mini
            summary = await self._get_summary(messages_text)
            
            # Отправляем результат
            response_message = (
                f"📄 **Пересказ диалога** ({len(self.chat_messages[chat_id])} сообщений)\n\n"
                f"{summary}\n\n"
                f"🤖 *Создано с помощью GPT-4.1-mini*"
            )
            
            await update.message.reply_text(response_message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Ошибка при создании суммаризации: {e}")
            await update.message.reply_text(
                "❌ Произошла ошибка при создании пересказа. Попробуйте позже."
            )
    
    def _format_messages_for_summary(self, messages: List[Dict]) -> str:
        """Форматирование сообщений для отправки в GPT"""
        formatted_messages = []
        
        for msg in messages:
            timestamp = datetime.fromisoformat(msg['timestamp']).strftime('%H:%M')
            username = msg['username']
            text = msg['text']
            formatted_messages.append(f"[{timestamp}] {username}: {text}")
        
        return "\n".join(formatted_messages)
    
    async def _get_summary(self, messages_text: str) -> str:
        """Получение суммаризации от OpenAI GPT-4.1-mini"""
        
        system_prompt = """
Ты - эксперт по суммаризации текстов. Твоя задача - создать краткий, но содержательный пересказ диалога.

Инструкции:
1. Выдели основные темы и ключевые моменты обсуждения
2. Сохрани важные детали и решения
3. Укажи, кто из участников что предлагал или решал
4. Структурируй ответ с использованием списков или абзацев
5. Отвечай на том же языке, что и исходные сообщения
6. Если в диалоге есть вопросы без ответов, обязательно их упомяни
7. Будь кратким, но не упускай важное

Формат ответа:
🔸 **Основные темы:** [перечисли главные темы]
🔸 **Ключевые моменты:** [важные детали и решения]
🔸 **Участники:** [кто что делал/предлагал]
🔸 **Нерешённые вопросы:** [если есть]
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Создай пересказ следующего диалога:\n\n{messages_text}"}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Ошибка при обращении к OpenAI API: {e}")
            raise
    
    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик команды /clear"""
        chat_id = update.effective_chat.id
        
        if not self.chat_messages[chat_id]:
            await update.message.reply_text("📭 Нет сообщений для удаления.")
            return
        
        message_count = len(self.chat_messages[chat_id])
        self.chat_messages[chat_id].clear()
        
        await update.message.reply_text(
            f"🗑️ Удалено {message_count} сообщений.\n"
            "Можете начинать новый диалог!"
        )
    
    async def count_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик команды /count"""
        chat_id = update.effective_chat.id
        count = len(self.chat_messages[chat_id])
        
        if count == 0:
            await update.message.reply_text("📭 Нет сохранённых сообщений.")
        else:
            await update.message.reply_text(
                f"📊 Сохранено сообщений: {count}\n"
                f"📝 Максимум: {MAX_MESSAGES_PER_CHAT}\n\n"
                "Используйте /summarize для создания пересказа."
            )
    
    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик ошибок"""
        logger.error(f"Ошибка: {context.error}")
        
        if isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text(
                "❌ Произошла непредвиденная ошибка. Попробуйте позже."
            )


async def main():
    """Основная функция запуска бота"""
    print("🚀 Запуск Telegram бота для суммаризации диалогов...")
    print("💡 Для остановки используйте Ctrl+C")
    print("📱 Найдите своего бота в Telegram и начните диалог командой /start")
    print("-" * 50)
    
    # Создаём экземпляр бота
    bot = DialogueSummarizerBot()
    
    # Создаём приложение
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Регистрируем обработчики команд
    application.add_handler(CommandHandler("start", bot.start_command))
    application.add_handler(CommandHandler("summarize", bot.summarize_command))
    application.add_handler(CommandHandler("clear", bot.clear_command))
    application.add_handler(CommandHandler("count", bot.count_command))
    
    # Регистрируем обработчик обычных сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.save_message))
    
    # Регистрируем обработчик ошибок
    application.add_error_handler(bot.error_handler)
    
    logger.info("Бот запущен и готов к работе!")
    
    # Запускаем бота
    await application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Бот остановлен пользователем")
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        logger.error(f"Критическая ошибка: {e}") 