import spacy
from loadSaveData import saveTokens, saveSinLimpiarTokens, loadSinLimpiarTokens, loadTokens
from tqdm import tqdm
import emoji


def tokenize(textos):
    """
    Tokeniza los textos recibidos como parametro de la siguiente manera:
    - Quita los emojis
    - Quita las palabras 'filler' y 'filer'
    - Quita las palabras de más de 3 tokens
    - Quita los stopwords
    - Quita todos los carácteres especiales
    - Lematiza los tokens
    - Pasa a minúscula todos los tokens
    """
    textos_token = loadTokens(len(textos))

    if isinstance(textos_token, bool):

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


def tokenizarSinLimpiar(rawText):
    """
    Prueba a cargar los tokens. Si no existen, los genera.
    """
    textos_token = loadSinLimpiarTokens(length=len(rawText)) 

    if isinstance(textos_token, bool):

        textos_token = []

        nlp = spacy.load("en_core_web_sm")  # Cargar modelo
        for texto in tqdm(rawText, desc="Procesando textos"):
            doc = nlp(texto)
            tokens_palabras = [token.text for token in doc if len(token.text) > 3 and token.is_alpha]
            textos_token.append(tokens_palabras)
        saveSinLimpiarTokens(textos_token)

    return textos_token
