from typing import Dict
from typing import List
from typing import Optional

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
import re

from base import Model


class TfIdfModel(Model):
    def __init__(self, 
                config: Dict, 
                train_datapoints: pd.DataFrame,
                val_datapoints: pd.DataFrame,
                test_datapoints: pd.DataFrame,
    ):
        self.testtt()
        self.config = config
        """
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.saved_model_path = os.path.join(self.base_dir, self.config["saved_model_path"])
        self.raw_model_path = os.path.join(self.base_dir, self.config["raw_model_path"])

        if self.config["train"] == "False":           
            self.model = BertForSequenceClassification.from_pretrained(self.saved_model_path)
        
        elif self.config["download_model"] == "True":
            self.model = BertForSequenceClassification.from_pretrained(self.config['model_type'], num_labels = 2)
        
        else:
            self.model = BertForSequenceClassification.from_pretrained(self.raw_model_path, num_labels = 2)

        self.training_args = TrainingArguments(**config["training_args"])
        """
        self.lemmatizer = WordNetLemmatizer()
        self.tfidfVectorizer = TfidfVectorizer()
        self.classifier = MultinomialNB()

        #self.inputs_train = train_datapoints[self.config['train_on']].values.tolist()
        #self.labels_train = train_datapoints['label'].values.tolist()

        self.train_data = train_datapoints
        self.train_data[self.config['train_on']] = self.train_data[self.config['train_on']].apply(self.__text_preprocess)

        #self.inputs_eval = val_datapoints[self.config['train_on']].values.tolist()
        #self.labels_eval = val_datapoints['label'].values.tolist()

        self.val_data = val_datapoints
        self.val_data[self.config['train_on']] = val_datapoints[self.config['train_on']].apply(self.__text_preprocess)

        #self.inputs_test = test_datapoints[self.config['train_on']].values.tolist()
        #self.labels_test = test_datapoints['label'].values.tolist()

        self.test_data = test_datapoints
        self.test_data[self.config['train_on']] = test_datapoints[self.config['train_on']].apply(self.__text_preprocess)



        """
        self.train_data = BertDataset(self.config, self.inputs_train, self.labels_train)
        self.eval_data = BertDataset(self.config, self.inputs_eval, self.labels_eval)
        self.test_data = BertDataset(self.config, self.inputs_test, self.labels_test)
       
        self.trainer = Trainer(model=self.model, args=self.training_args, train_dataset=self.train_data, eval_dataset=self.eval_data)
        """

        self.testtt()
        self.train()
    def testtt(self):
        print('testtt')
          
    def train(self):
        print('Training...')
        train_transformed = self.tfidfVectorizer.fit_transform(self.train_data[self.config['train_on']]).toarray()
        test_transformed = self.tfidfVectorizer.transform(self.test_data[self.config['train_on']])

        self.classifier.fit(train_transformed, self.train_data.label)
        #self.trainer.save_model(self.saved_model_path)
        y_pred = self.classifier.predict(test_transformed)
        classification_report_ = classification_report(self.test_data.label, y_pred)

        print('\n Accuracy: ', accuracy_score(self.test_data.label, y_pred))
        print('\nClassification Report')
        print('======================================================')
        print('\n', classification_report_)
    
    def eval(self):
        print(self.trainer.evaluate())

    def predict(self, test_data):
        return self.trainer.predict(test_data)
        
    def compute_metrics(self):
        expected_labels = self.labels_test
        predicted_proba = self.predict(self.test_data)
        predicted_labels = np.argmax(predicted_proba[0], axis=1)
       
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

    def __text_preprocess(self, text: str) -> str:
        text = re.sub('[^a-zA-Z]', ' ', text) # Retain only alphabets
        text = text.lower() # Lower case string
        text = [w for w in text.split() if not w in set(stopwords.words('english'))]  # Remove stopwords
        
        text = [self.lemmatizer.lemmatize(w) for w in text if len(w) > 1] # Lemmatize
        
        text = ' '.join(text)

        return text


if __name__ == "__main__":
    config = {
        'train_on': 'text'
    }

    train = pd.read_csv('../../data/train_berkeley_1.csv')
    test = pd.read_csv('../../data/test_berkeley_1.csv')
    val = pd.read_csv('../../data/eval_berkeley_1.csv')

    model = TfIdfModel(config, train, val, test)

    #print(model)
    #print(type(model))
    #model.train()

