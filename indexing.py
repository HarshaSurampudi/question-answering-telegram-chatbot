import pandas as pd
import pinecone
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm  

# Load the file into a list of strings
with open('documents.txt', 'r') as f:
    data = f.read().split('\n\n')

# Create a DataFrame with the paragraphs
df = pd.DataFrame({'paragraph': data})


# connect to pinecone environment
pinecone.init(
    api_key="305100e5-8a5c-4e22-bd61-7fc99335626c",
    environment="us-east1-gcp"  # find next to API key in console
)


index_name = "abstractive-question-answering"
#check if the index with your name already exists. pinecone will let you create only one index in the free tier
if index_name not in pinecone.list_indexes():
    # create the index if it does not exist
    pinecone.create_index(
        index_name,
        dimension=768,
        metric="cosine"
    )

# connect to abstractive-question-answering index we created
index = pinecone.Index(index_name)

# set device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# load the retriever model from huggingface model hub
retriever = SentenceTransformer("flax-sentence-embeddings/all_datasets_v3_mpnet-base")
retriever


# we will use batches of 64
batch_size = 64

for i in tqdm(range(0, len(df), batch_size)):
    # find end of batch
    i_end = min(i+batch_size, len(df))
    # extract batch
    batch = df.iloc[i:i_end]
    # generate embeddings for batch
    emb = retriever.encode(batch["paragraph"].tolist()).tolist()
    # get metadata
    meta = batch.to_dict(orient="records")
    # create unique IDs
    ids = [f"{idx}" for idx in range(i, i_end)]
    # add all to upsert list
    to_upsert = list(zip(ids, emb, meta))
    # upsert/insert these records to pinecone
    _ = index.upsert(vectors=to_upsert)

print(index.describe_index_stats())

