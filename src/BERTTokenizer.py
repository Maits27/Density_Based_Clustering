#https://www.analyticsvidhya.com/blog/2023/08/bert-embeddings/#:~:text=Using%20the%20BERT%20tokenizer%2C%20creating,word%20in%20the%20input%20text.
#https://medium.com/mlearning-ai/getting-contextualized-word-embeddings-with-bert-20798d8b43a4
#https://medium.com/@priyatoshanand/handle-long-text-corpus-for-bert-model-3c85248214aa#:~:text=If%20the%20tokens%20in%20a%20sequence%20are%20longer%20than%20512,in%20each%20of%20the%20tokens.

import torch # PyTorch
from transformers import AutoTokenizer, RobertaModel # library by HuggingFace
import pandas as pd
from tqdm import tqdm
import numpy as np
from loadSaveData import saveEmbeddings, loadEmbeddings

model = RobertaModel.from_pretrained('cardiffnlp/twitter-xlm-roberta-base').eval() # BERT base, which is a BERT model consists of 12 layers of Transformer encoder, 12 attention heads, 768 hidden size, and 110M parameters.
tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-xlm-roberta-base', use_fast=True)

data = pd.read_csv('../Datasets/Suicide_Detection10000.csv')
texts = []

for instancia in tqdm(data.values, desc='Loading csv'):
    texts.append(instancia[1])

# Método 1 (solo tokenizar)
"""
sentence = "Kaixo, egun on!"
print(tokenizer.encode(sentence))
"""

# Método 2 (solo tokenizar)
"""
tokens = tokenizer.tokenize(sentence)
print(tokens)

tokensWithSpecialChars = ['[CLS]'] + tokens + ['[SEP]']

token_ids = tokenizer.convert_tokens_to_ids(tokensWithSpecialChars)
print(token_ids)
"""

# Método 3
embeddingList = []
for text in tqdm(texts, desc='Generando embeddings'):

    tokens = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, add_special_tokens=True) 
    # Returns a tensor, https://huggingface.co/docs/transformers/v4.31.0/model_doc/bert#transformers.BertTokenizer
    # Truncates the text if the text has more than 512 tokens
    #print('Number of tokens:', tokens['input_ids'].shape[1])

    with torch.no_grad():
        output = model(**tokens)[0]

    embedding = output[0][0].numpy()
    embeddingList.append(embedding)

print('Each embedding has:', len(embeddingList[0]), 'dimensions')
print('Embedding list length (before save):', len(embeddingList))
npEmbeddingList = np.array(embeddingList)
saveEmbeddings(npEmbeddingList, len(embeddingList[0]), type='bert')

# Check
loadedEmbeddings = loadEmbeddings(len(embeddingList), len(embeddingList[0]), type='bert')
print('Embedding list length (after save)', len(loadedEmbeddings))


