import pinecone
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler

# connect to pinecone environment
pinecone.init(
    api_key="305100e5-8a5c-4e22-bd61-7fc99335626c",
    environment="us-east1-gcp"  # find next to API key in console
)

index = pinecone.Index('abstractive-question-answering')

import torch
from sentence_transformers import SentenceTransformer

# set device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# load the retriever model from huggingface model hub
retriever = SentenceTransformer("flax-sentence-embeddings/all_datasets_v3_mpnet-base", device=device)

from transformers import BartTokenizer, BartForConditionalGeneration

# load bart tokenizer and model from huggingface
tokenizer = BartTokenizer.from_pretrained('vblagoje/bart_lfqa')
generator = BartForConditionalGeneration.from_pretrained('vblagoje/bart_lfqa').to(device)

def query_pinecone(query, top_k):
    # generate embeddings for the query
    xq = retriever.encode([query]).tolist()
    # search pinecone index for context passage with the answer
    xc = index.query(xq, top_k=top_k, include_metadata=True)
    return xc

def format_query(query, context):
    # extract passage_text from Pinecone search result and add the <P> tag
    context = [f"<P> {m['metadata']['paragraph']}" for m in context]
    # concatinate all context passages
    context = " ".join(context)
    # contcatinate the query and context passages
    query = f"question: {query} context: {context}"
    return query

def generate_answer(query):
    # tokenize the query to get input_ids
    inputs = tokenizer([query], max_length=1024, return_tensors="pt")
    # use generator to predict output ids
    ids = generator.generate(input_ids=inputs["input_ids"].to(device),
                                           attention_mask=inputs["attention_mask"].to(device),
                                           min_length=26,
                                           max_length=256,
                                           do_sample=False, 
                                           early_stopping=True,
                                           num_beams=8,
                                           temperature=1.0,
                                           top_k=None,
                                           top_p=None,
                                           eos_token_id=tokenizer.eos_token_id,
                                           no_repeat_ngram_size=3,
                                           num_return_sequences=1)
    # use tokenizer to decode the output ids
    answer = tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return answer

async def hello(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f'Hello {update.effective_user.first_name}. I am a taxbot. Ask me a question about taxes and I will try to answer it. (e.g. What is Income Tax). I am still learning so please be patient.')

async def message(update: Update, context):
    query = update.message.text
    result = query_pinecone(query, top_k=3)
    matches = result["matches"]
    #filter matches with score less than 0.5
    matches = [m for m in matches if m["score"] >= 0.5]
    if len(matches) == 0:
        await update.message.reply_text("Sorry, I don't know the answer to that question. Please try again.")
        return
    query = format_query(query, matches)
    await update.message.reply_text(generate_answer(query))


app = ApplicationBuilder().token("6020966392:AAEykpo7mcFM9pjeCp5bRfA5tuFJmrCpVoY").build()

app.add_handler(CommandHandler("hello", hello))
app.add_handler(MessageHandler(None, message))
print("Starting bot...")
app.run_polling(print("Bot started!"))
