# Fake news Detection and Applications

This project is compose of two distincts parts : 

## 1 - ML pipeline for Fake News detection 

This first part is a standart ML pipeline to train and evaluate three models for fake news detection : TF_IDF, FastText and BERT. 

### Preprocessing et dataset generation:

The raw dataset can be put in folder data/raw and should be in .csv

The Dataset class will load the dataset and preprocess it. There are two simple preprocessing steps. First it will remove empty articles and then it will remove duplicates. It's also possible to  combine several datasets together. Duplicated removal is done by computing ngram TfIdfVectors of articles and then using cosine distance for time complexity matters.

The new dataset will be stored and every time the class is initialized the class will first try to locate the file. If the file doesn't exist, the preprocessing will be launched.

Preprocess dataset will be saved in folder data/processed


### Dataset analyses :

If you want to analyse your dataset you can use the notebook ??????? wich contains saveral type of analyses like ??????

blabla

### Launch pipeline to train & evaluate :

Each model has been trained on titles and texts. Config files for each model can be found in the config folder. You can choose in config file if you want to train & evaluate your model or just evaluate it ( ie. used an already trained model saved before in the folder fakenewsdetec/model/saved_model/YOUR_ALREADY_TRAINED_MODEL) .

To launch the pipeline with the config file of your choice use the command below :

`python main.py --config-file config/YOUR_CONFIG_FILE.json`

## 2 - Adversarial approach

The objective of this part is to better understand our models, trained in the previous part, and to find what they didn't learn well.
Newt we want to create a fake news generator that will generate mislearned articles in order to train our models on them.
This way our models should be improved by those new trainings.

### Models analyses with LIME

We used LIME to understand model behaviour and explain the model predictions. lime() method (for tfidf model) will run the prediction for the test dataset and then the analysis on false negatifs. Word weights are printed and are also saved in a json file.


### Retrieve Real Headlines 

????

### Fake news generation

We used GROVER to build a fake news generator that takes in input real headlines with keywords ( found with LIME ) on it. You can find the generator code in the notebook located inside the folder notebook/generator. Fakes news generated wiil be stored in the data/generator/generated folder in a .csv file, you can use them to train again your models
