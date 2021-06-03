from functools import partial
import os
import pickle
import re
from typing import Dict
from typing import List
from typing import Optional

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB

from base import Model


class TfIdfModel(Model):
    def __init__(self, 
                config: Dict, 
                train_datapoints: pd.DataFrame,
                val_datapoints: pd.DataFrame,
                test_datapoints: pd.DataFrame,
    ):
        self.config = config
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.saved_model_path = os.path.join(self.base_dir, self.config.get("saved_model_path", ""))

        self.lemmatizer = WordNetLemmatizer()
        self.tfidfVectorizer = TfidfVectorizer()
        self.classifier = MultinomialNB()

        if bool(self.config.get("train", False)):
            self.model = None
        else:
            try:
                self.model = pickle.load(self.saved_model_path)
            except TypeError as e:
                print("Impossible to load existing model")
                self.model = None
        #self.raw_model_path = os.path.join(self.base_dir, self.config["raw_model_path"])

        preprocess = partial(self.text_preprocess, self.lemmatizer)

        self.train_data = train_datapoints
        self.train_data[self.config['train_on']] = self.train_data[self.config['train_on']].apply(preprocess)

        self.val_data = val_datapoints
        self.val_data[self.config['train_on']] = val_datapoints[self.config['train_on']].apply(preprocess)

        self.test_data = test_datapoints
        self.test_data[self.config['train_on']] = test_datapoints[self.config['train_on']].apply(preprocess)

          
    def train(self) -> None:
        train_transformed = self.tfidfVectorizer.fit_transform(self.train_data[self.config['train_on']]).toarray()
        self.classifier.fit(train_transformed, self.train_data.label)

        if self.saved_model_path:
            try:
                pickle.dump(self.classifier, self.saved_model_path) 
                print("Model saved")
            except:
                pass


    def predict(self, test_data: List) -> List:
        test_transformed = self.tfidfVectorizer.transform(test_data)
        return self.classifier.predict(test_transformed)
        
    def compute_metrics(self) -> None:
        expected_labels = self.test_data.label
        predicted_labels = self.predict(self.test_data[self.config['train_on']])
       
        accuracy = accuracy_score(expected_labels, predicted_labels)
        f1 = f1_score(expected_labels, predicted_labels)
        auc = roc_auc_score(expected_labels, predicted_labels)
        conf_mat = confusion_matrix(expected_labels, predicted_labels)
        tn, fp, fn, tp = conf_mat.ravel()
        print(f"Accuracy: {accuracy}, F1: {f1}, AUC: {auc}")

        print( {"true negative": tn,
                "false negative": fn,
                "false positive": fp,
                "true positive": tp,
        })

    @staticmethod
    def text_preprocess(lemmatizer, text: str) -> str:
        text = re.sub('[^a-zA-Z]', ' ', text) # Retain only alphabets
        text = text.lower() # Lower case string
        text = [w for w in text.split() if not w in set(stopwords.words('english'))]  # Remove stopwords
        
        text = [lemmatizer.lemmatize(w) for w in text if len(w) > 1] # Lemmatize
        
        text = ' '.join(text)

        return text


if __name__ == "__main__":
    config = {
        'train_on': 'text'
    }

    #generated = table = pd.read_csv('../../data/generated_news.csv')
    #generated['label'] = 1
    #train = pd.concat([pd.read_csv('../../data/train_berkeley_1.csv'), generated])
    train = pd.read_csv('../../data/train_berkeley_1.csv')
    test = pd.read_csv('../../data/test_berkeley_1.csv')
    val = pd.read_csv('../../data/eval_berkeley_1.csv')

    model = TfIdfModel(config, train, val, test)
    model.train()
    model.compute_metrics()
