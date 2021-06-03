# Fake news Detection and Applications

This project is compose of two parts : 

the first part is a standart ML pipeline to train and evaluate three models of fakes news detection : TF_IDF, FastText and BERT. 

Preprocessing et génération du dataset à partir des raw data dans data/raw:

blabla

Analyses du dataset :

blabla

Pipeline train + evaluate :

Each model has been trained on titles and texts. Config files for each model can be found in the config folder. You can choose in config file if you want to train & evaluate your model or just evaluate and used an already train model saved before in the folder fakenewsdetec/model/saved_model.

To launch the pipeline with configuration choose in the config file use the command :

python main.py --config-file config/YOUR_CONFIG_FILE.json
