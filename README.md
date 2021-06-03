# Fake news Detection and Applications

This project is compose of two distincts parts : 

## 1 - ML pipeline for Fake News detection 

This first part is a standart ML pipeline to train and evaluate three models for fake news detection : TF_IDF, FastText and BERT. 

### Preprocessing et dataset generation:

The raw dataset can be put in folder data/raw and should be in .csv
To launch preprocessing use the command below, it will compute a ?????? :

blabla

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

?????

### Retrieve Real Headlines 

????

### Fake news generation

We used GROVER to build a fake news generator that takes in input real headlines with keywords ( found with LIME ) on it. You can found the generator code in the notebook located inside the folder notebook/generator
