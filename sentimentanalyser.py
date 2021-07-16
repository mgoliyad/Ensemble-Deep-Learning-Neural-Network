import plac
import pathlib
#from spacy.compat import pickle
from model import Model
from data_processing import DataLoader
import spacy
#import os
import json
import numpy as np 
#import tensorflow as tf

nlp = spacy.load("en_vectors_web_lg")

def main():  
    configs = json.load(open('config.json', 'r'))
    model_dir = configs['data']['model_dir']
    train_dir = configs['data']['train_dir']
    
    if model_dir is not None:
        model_dir = pathlib.Path(model_dir)
    if train_dir is None:
        print('Please provide training directory!')
    else:
        train_dir = pathlib.Path(train_dir)
        
    data = DataLoader(nlp, configs)
    train_texts, train_labels, val_texts, val_labels = data.read_data(configs, train_dir)
    
    print("Parsing texts...")
        
    train_docs = list(nlp.pipe(train_texts))
    val_docs = list(nlp.pipe(val_texts))
    if configs['training']['by_sentence']:
        train_docs, train_labels = data.get_labelled_sentences(train_docs, train_labels)
        val_docs, val_labels = data.get_labelled_sentences(val_docs, val_labels)
            
    train_vec= data.get_vectors(train_docs)
    val_vec = data.get_vectors(val_docs)
    predictions = []
    
    model = Model(nlp, configs, predictions, val_vec)
    model.train_model(train_vec,train_labels,val_vec,val_labels)
    
    predictions = np.array(predictions)
    ensemble_prediction = model.model_evaluation(val_labels)
    val_labels = np.argmax(val_labels, axis=1)
    
    print('We got ', np.sum(ensemble_prediction != val_labels), 'out of ', val_labels.shape[0], 'misclassified texts')
    print('Here is the list of misclassified texts:\n')
    
    val_texts = np.array(val_texts).reshape(-1)
    
    print(val_texts[np.array(np.where(ensemble_prediction != val_labels))][:])

if __name__ == "__main__":
    plac.call(main)
