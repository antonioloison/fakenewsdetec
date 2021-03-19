from pathlib import Path

import pandas as pd

def load_isot_dataset(folder: Path):
    """Load isot dataset.
    
    :param folder: folder to look for isot files
    :type folder: pathlib.Path
    
    :rtype: pd.Dataframe
    :return: isot dataset"""
    isot_df = []
    
    for label, csv_file in enumerate(['True.csv', 'Fake.csv']):
        isot_df.append(pd.read_csv(folder / Path(csv_file), header=0))
        isot_df[-1]['label'] = label
        
    isot_df_concat = pd.concat(isot_df)
    isot_df_concat['dataset'] = 'isot'
    
    isot_df_concat = isot_df_concat.sample(frac=1).reset_index(drop=True)
    
    return isot_df_concat


def load_kaggle_dataset(folder: Path):
    """Load kaggle dataset.
    
    :param folder: folder to look for kaggle files
    :type folder: pathlib.Path
    
    :rtype: pd.Dataframe
    :return: kaggle dataset"""
    kaggle_df = pd.read_csv(folder / Path('train.csv'), header=0, index_col='id')
    kaggle_df['dataset'] = 'kaggle'
        
    return kaggle_df

def load_berkeley_dataset(folder: Path):
    berkeley_df = pd.read_csv(folder / 'fake_or_real_news.csv', index_col=0)
    berkeley_df['dataset'] = 'berkeley'
    berkeley_df['label'] = berkeley_df['label'].map({'FAKE': 1, 'REAL': 0})
    return berkeley_df

load_dataset = {
    'berkeley': load_berkeley_dataset,
    'isot': load_isot_dataset,
    'kaggle': load_kaggle_dataset
}
