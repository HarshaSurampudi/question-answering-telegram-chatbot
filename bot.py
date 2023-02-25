from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline

model = AutoModelForSequenceClassification.from_pretrained("harshasurampudi/autotrain-intent-classification-roberta-3647697496")
tokenizer = AutoTokenizer.from_pretrained("harshasurampudi/autotrain-intent-classification-roberta-3647697496")

TELEGRAM_BOT_TOKEN='your_telegram_bot_token'


my_pipeline = pipeline(task="text-classification", model=model, tokenizer=tokenizer)
 

# get qa.json as dictionary
import json
with open('qa.json') as f:
    qa = json.load(f)


async def hello(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f'Hello {update.effective_user.first_name}. I am a taxbot. Ask me a question about taxes and I will try to answer it. (e.g. What is Income Tax?)')

async def message(update: Update, context):
    query = update.message.text
    my_score = my_pipeline(query)
    question_tag = my_score[0]['label']
    answer = qa[question_tag]['answer']
    await update.message.reply_text(answer)


app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

app.add_handler(CommandHandler("hello", hello))
app.add_handler(MessageHandler(None, message))
print("Starting bot...")
app.run_polling(print("Bot started!"))
