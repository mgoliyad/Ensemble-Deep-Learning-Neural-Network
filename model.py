from ensemble import SnapshotEnsemble
from keras.layers import LSTM, Dense, Embedding, Bidirectional, TimeDistributed
from keras.models import Sequential
from sklearn.metrics import accuracy_score
import numpy as np 
import matplotlib.pyplot as plt


class Model(SnapshotEnsemble):
    def __init__(self, nlp, configs, predictions, val_vec):
        
        self.nb_epochs = configs['training']['nb_epochs']
        self.batch_size = configs['training']['batch_size']
        self.nb_cycles = configs['training']['nb_cycles']
        self.lr_max = configs['training']['lrate_max']
        self.lrates = list()
        self.nlp = nlp
        self.model = Sequential()
        self.vocab = self.nlp.vocab
        self.lrates = list()
        self.configs = configs
        self.predictions = predictions
        self.val_vec = val_vec
        self.ca = SnapshotEnsemble(self.configs, self.val_vec, self.predictions )
        
        
    def get_embeddings(self, vocab):
        return vocab.vectors.data
 
    def plot_series(self, series_1, series_2, format="-", title=None, legend=None):
        plt.plot(series_1)
        plt.plot(series_2)
        plt.title(title)
        plt.legend(legend, loc='upper left')
        plt.show()  
        
    def get_labelled_sentences(self, docs, doc_labels):
        labels = []
        sentences = []
        for doc, y in zip(docs, doc_labels):
            for sent in doc.sents:
                sentences.append(sent)
                labels.append(y)
        return sentences, np.asarray(labels, dtype="int32")

    def train_model(self,train_vecs,train_labels,val_vecs,val_labels):
        
        print("Loading spaCy")
        self.nlp.add_pipe(self.nlp.create_pipe("sentencizer"))
        embeddings = self.get_embeddings(self.vocab)
        model = self.compile_model(embeddings)

        estimator = model.fit(
            train_vecs,
            train_labels,
            validation_data=(val_vecs, val_labels),
            epochs=self.nb_epochs,
            batch_size=self.batch_size, 
            callbacks=[self.ca]
        )

        self.plot_series(estimator.history['acc'], estimator.history['val_acc'], title='Model accuracy for Chatbot dataset', legend=['train', 'valid'])
        self.plot_series(estimator.history['loss'], estimator.history['val_loss'], title='Model loss for Chatbot dataset', legend=['train', 'valid'])

    def compile_model(self, embeddings):
        
        metrics = [self.configs['training']['metrics']]
        nb_hiddens = self.configs['training']['nb_hiddens']
        dropout = self.configs['training']['dropout']
        nb_class = self.configs['training']['nb_class']
        activation = self.configs['training']['activation']
        max_length = self.configs['training']['max_length']
        optimizer = self.configs['training']['optimizer']
        loss = self.configs['training']['loss']
        
        self.model.add(
            Embedding(
                embeddings.shape[0],
                embeddings.shape[1],
                input_length=max_length,
                trainable=False,
                weights=[embeddings],
                mask_zero=True,
            )
        )
        self.model.add(TimeDistributed(Dense(nb_hiddens, use_bias=False)))
        self.model.add(
            Bidirectional(
                LSTM(
                    nb_hiddens,
                    recurrent_dropout=dropout,
                    dropout=dropout, 
                    return_sequences=True
                )
            )
        )
        self.model.add(
            Bidirectional(
                LSTM(
                    nb_hiddens,
                    recurrent_dropout=dropout,
                    dropout=dropout,
                    return_sequences=True,
                )
            )
        )
        self.model.add(
            Bidirectional(
                LSTM(
                    nb_hiddens,
                    recurrent_dropout=dropout,
                    dropout=dropout,
                    return_sequences=True,
                )
            )
        )
        self.model.add(
            Bidirectional(
                LSTM(
                    nb_hiddens,
                    recurrent_dropout=dropout,
                    dropout=dropout,
                )
            )
        )
        self.model.add(Dense(nb_class, activation=activation))

        self.model.compile(optimizer=optimizer, loss = loss, metrics=metrics)
        
        return self.model

    # make an ensemble prediction
    def model_evaluation(self, val_labels):
        summed = np.sum(self.predictions, axis=0)
        result = np.argmax(summed, axis=1)
        acc_score = accuracy_score(np.argmax(val_labels, axis=1), result)
        print('Ensemble score: ', acc_score)
        return result
    