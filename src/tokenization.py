import spacy
from loadSaveData import saveTokens
from tqdm import tqdm
import emoji


def tokenize(textos):
    textos_token = []

    nlp = spacy.load("en_core_web_sm")  # Cargar modelo
    nlp.add_pipe("emoji", first=True)

    for texto in tqdm(textos, desc="Procesando textos"):
        texto = emoji.demojize(texto)  # Emojis a texto
        texto = texto.replace(':', ' ').replace('filler', ' ').replace('filer', ' ').replace('_', ' ')
        doc = nlp(texto)
        lexical_tokens = [token.lemma_.lower() for token in doc if len(token.text) > 3 and token.is_alpha and not token.is_stop]
        textos_token.append(lexical_tokens)

    saveTokens(textos_token)

    return textos_token