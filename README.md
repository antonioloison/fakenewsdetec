# Fake news Detection and Applications

This project is composed of two distincts parts : 

## 1 - ML pipeline for Fake News detection 

This first part is a standard ML pipeline to train and evaluate three models for fake news detection : TF_IDF, FastText and BERT. 

### Preprocessing and dataset generation:

The raw dataset can be put in folder `data/raw` and should be in `csv` format.

The `Dataset` class will load the dataset and preprocess it. There are two simple preprocessing steps. First it removes empty articles and then it removes duplicates. It's also possible to  combine several datasets together. Duplicated removal is done by computing ngram TfIdfVectors of articles and then using cosine distance for time complexity matters.

The new dataset will be stored and every time the class is initialized, the class will first try to locate the file. If the file doesn't exist, the preprocessing is launched.

Preprocess dataset will be saved in folder `data/processed` 


### Dataset analyses :

If you want to analyse your dataset you can use the notebooks in the folder `notebook/dataset_exploration` wich contains several types of analysis like:
- Label distribution
- Metadata distribution
- Text length distribution
- Vocabulary distribution
- Sentiment Analysis

### Launch pipeline to train & evaluate :

Each model has been trained on titles and texts. Config files for each model can be found in the `config` folder. You can choose in the config file if you want to train & evaluate your model or just evaluate it ( ie. used an already trained model saved before in the folder fakenewsdetec/model/saved_model/YOUR_ALREADY_TRAINED_MODEL) .

To launch the pipeline with the config file of your choice use the command below :

```python main.py --config-file config/YOUR_CONFIG_FILE.json```

There are three examples of config files depending on the model you want to use. To configure your own config file, please replace the `.dist.json` with `.json` and work on that file.


## 2 - Adversarial approach

The objective of this part is to better understand our models, trained in the previous part, and to find what they didn't learn well.
We want to create a fake news generator that will generate mislearned articles in order to train our models on them.
This way our models should be improved by those new trainings.

### Models analyses with LIME

We used LIME to understand model behaviour and explain the model predictions. lime() method (for tfidf model) will run the prediction for the test dataset and then the analysis on false negatifs. Word weights are printed and are also saved in a json file.


### Retrieve Real Headlines 

To retrieve real headlines, you can execute this command:

``` python fakenewsdetec/fakenewsdetec/google_news_scrapping.py ```

You can change the words that you want in the headlines by changing the variable `words`.

### Fake news generation

We used GROVER to build a fake news generator that takes as input real headlines with keywords ( found with LIME ) on it. You can find the generator code in the notebook located inside the folder `notebook/generator`. Generated fake news will be stored in the `data/generator/generated` folder in a .csv file, you can use them to train again your models.
