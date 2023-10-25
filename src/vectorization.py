from loadSaveData import saveEmbeddings, loadEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from gensim.test.utils import get_tmpfile
from transformers import AutoTokenizer, RobertaModel # library by HuggingFace
from tqdm import tqdm
import torch # PyTorch
import numpy as np


def tfidf(textosToken, dimensiones):
    vectorizer = TfidfVectorizer()
    documentVectors = vectorizer.fit_transform(textosToken)
    saveEmbeddings(documentVectors, dimensiones)


def doc2vec(textosToken, dimensiones):

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(textosToken)]
    model = Doc2Vec(documents, vector_size=dimensiones, window=2, dm=1, epochs=100, workers=4)

    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

    model.save(get_tmpfile("my_doc2vec_model"))

    documentVectors = [model.infer_vector(doc) for doc in textosToken]

    saveEmbeddings(documentVectors, dimensiones)


def bertTransformer(rawData):
    model = RobertaModel.from_pretrained('cardiffnlp/twitter-xlm-roberta-base').eval() # BERT base, which is a BERT model consists of 12 layers of Transformer encoder, 12 attention heads, 768 hidden size, and 110M parameters.
    tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-xlm-roberta-base', use_fast=True)

    embeddingList = []
    for text in tqdm(rawData, desc='Generando embeddings'):

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
    
    return loadedEmbeddings
