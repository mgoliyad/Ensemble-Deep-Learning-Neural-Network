import pandas as pd 
import numpy as np 
import pathlib
from keras.utils import to_categorical
from spacy.lang.en.stop_words import STOP_WORDS

class DataLoader():
    def __init__(self, nlp, configs):
        self.nlp = nlp
        self.configs = configs

    def cleanup_text(self, docs, logging=False):
        docs = docs.str.strip().replace("\n", " ").replace("\r", " ")
        texts = []
        counter = 1
        for doc in docs:
            if counter % 1000 == 0 and logging:
                print("Processed %d out of %d documents." % (counter, len(docs)))
            counter += 1
            doc = self.nlp(doc, disable=['parser', 'ner'])
            tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-' and tok.pos_ !='NUM' and tok.pos_ !='PUNCT' and tok not in STOP_WORDS]
            tokens = ' '.join(tokens)
            texts.append(tokens)
        return pd.Series(texts)
    
    def get_vectors(self, docs):
        max_length = self.configs['training']['max_length']
        docs = list(docs)
        Xs = np.zeros((len(docs), max_length), dtype="int32")
        for i, doc in enumerate(docs):
            j = 0
            for token in doc:
                vector_id = token.vocab.vectors.find(key=token.orth)
                if vector_id >= 0:
                    Xs[i, j] = vector_id
                else:
                    Xs[i, j] = 0
                j += 1
                if j >= max_length:
                    break
        return Xs

    def read_data(self, configs, data_dir):
        texts = pd.DataFrame()
        for filename in pathlib.Path(data_dir).iterdir():
            with filename.open(encoding='latin-1') as file_:
                if not file_.name.endswith('DS_Store'):
                    text = pd.read_csv(file_, usecols=[1, 2], encoding='latin-1')
                    texts = texts.append(text, ignore_index=True)
        texts = texts.sample(frac=1)
        text_cln = self.cleanup_text(texts.iloc[:, 1], logging=True)
        sentiments = np.asarray(texts.iloc[:, 0].unique())
        for i in range(len(sentiments)):
            texts.iloc[:, 0].replace(sentiments[i], i, inplace=True)

        train_size = int(len(texts) * configs['training']['training_portion'])
        
        train_texts, train_labels = text_cln[:train_size], texts.iloc[:train_size, 0]
        val_texts, val_labels = text_cln[train_size:], texts.iloc[train_size:, 0]
        train_labels = to_categorical(train_labels,2)
        val_labels = to_categorical(val_labels,2)
        
        return train_texts, train_labels, val_texts, val_labels
