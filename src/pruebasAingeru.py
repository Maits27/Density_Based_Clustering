#from loadSaveData import loadEmbeddings, loadRAW
from loadSaveData import loadEmbeddings, loadRAW, loadRAWwithClass, loadSinLimpiarTokens
from clustering import DBScanOriginal, DensityAlgorithmUrruela
from tokenization import tokenizarSinLimpiar
from evaluation import wordCloud, classToCluster
from transformers import AutoTokenizer, RobertaModel # library by HuggingFace
import torch # PyTorch


if __name__ == '__main__':
    # Probar modelo
    vectors = loadEmbeddings(length=10000, dimension=768, type='bert')
    algoritmo = DBScanOriginal(vectors=vectors, epsilon=0.8082, minPt=4) # dim=768
    algoritmo.ejecutarAlgoritmo()

    rawData = loadRAWwithClass('../Datasets/Suicide_Detection10000.csv')
    tokenTexts = loadSinLimpiarTokens(10000)
    #tokenTexts = tokenizarSinLimpiar(rawData['text'])

    wordCloud(algoritmo.clusters, textos_tokenizados=tokenTexts)
    classToCluster(data=rawData, clusters=algoritmo.clusters)

    # Tokenización
    """tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-xlm-roberta-base', use_fast=True)
    model = RobertaModel.from_pretrained('cardiffnlp/twitter-xlm-roberta-base').eval() # BERT base, which is a BERT model consists of 12 layers of Transformer encoder, 12 attention heads, 768 hidden size, and 110M parameters.

    text = 'Hello! This is an example of BERT tokenization'
    tokens = tokenizer.tokenize(text, max_length=512, truncation=True, add_special_tokens=True) 
    tokensDeUna = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, add_special_tokens=True)
    encoded = tokenizer.encode_plus(text)

    with torch.no_grad():
        output = model(**tokensDeUna)[0]
    embedding = output[0][0].numpy()

    print(tokens)
    print(encoded['input_ids'])
    print(tokensDeUna['input_ids'])
    print(embedding)"""
