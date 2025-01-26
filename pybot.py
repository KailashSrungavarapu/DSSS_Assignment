import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

logging.basicConfig(
    format='%(asctime=s - %(levelname)s - %(message)s',
    level=logging.INFO
)

application = ApplicationBuilder().token("8017384907:AAEDNyoBDYtJInVDPJYChgfnuGqrYr33W7A").build()

model_name = "EleutherAI/pythia-1.4b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Hello AI Assistant!')

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text
    messages = [
        {"role": "user", "content": user_message},
    ]
    prompt = " ".join([msg["content"] for msg in messages])
    response = pipe(prompt, max_new_tokens=150, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    await update.message.reply_text(response[0]['generated_text'])

application.add_handler(CommandHandler("start", start))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

print("Starting the bot...")
application.run_polling()
print("Bot is running")
