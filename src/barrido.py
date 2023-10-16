from sklearn.cluster import DBSCAN
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.test.utils import get_tmpfile

def barridoDBSCAN(espilonList, minPtsList):
    document_vectors = [model.infer_vector(doc) for doc in textos_tokenizados]

    # Aplicar DBSCAN a los vectores de documentos
    dbscan = DBSCAN(eps=2, min_samples=2, leaf_size=5)  # Ajusta los parámetros según tu caso
    labels = dbscan.fit_predict(np.array(document_vectors))

    # Los resultados del clustering están en 'labels'
    print("Etiquetas de clusters:", labels)


documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(textos_tokenizados)]
model = Doc2Vec(documents, vector_size=150, window=2, dm=1, epochs=100, workers=4)

model.build_vocab(documents)
model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

model.save(get_tmpfile("my_doc2vec_model"))