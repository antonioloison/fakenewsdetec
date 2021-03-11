from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from typing import Tuple

class Dataset:
    def __init__(self, data_folder):
        self.data_folder = Path(data_folder)
        self.isot_folder = Path('isot/')
        self.kaggle_folder = Path('kaggle/')

    def load_dataset(self, dataset: int = 0):
        """Load the specified dataset.
        
        :param dataset: 0 loads isot + kaggle; 1 loads isot; 2 loads kaggle
        :type dataset: int, optional
        
        :rtype: pd.Dataframe
        :return: dataset
        """
        if dataset==0:
            dataset_name = 'combined_dataset.csv'
        elif dataset==1:
            dataset_name = 'isot.csv'
        elif dataset==2:
            dataset_name = 'kaggle.csv'

        data_path = self.data_folder / 'processed/' / dataset_name

        if data_path.is_file():
            df = pd.read_csv(data_path, header=0)
        else:
            df = self.__build_dataset(data_path, dataset)
        
        return df

    def __build_dataset(self, path: Path, dataset: int = 0):
        """Load the specified dataset.
        
        :param dataset: 0 loads isot + kaggle; 1 loads isot; 2 loads kaggle
        :type dataset: int, optional
        
        :rtype: pd.Dataframe
        :return: dataset
        """
        if dataset==0:
            df = pd.concat([self.__load_isot_dataset(self.data_folder / self.isot_folder), self.__load_kaggle_dataset(self.data_folder / self.kaggle_folder)])
            df = df.sample(frac=1).reset_index(drop=True)
        elif dataset==1:
            df = self.__load_isot_dataset(self.data_folder / self.isot_folder)
        elif dataset==2:
            df = self.__load_kaggle_dataset(self.data_folder / self.kaggle_folder)
            
        df['text'] = df['text'].astype(str)
        df['title'] = df['title'].astype(str)

        df = self.__preprocess_dataset(df)

        path.parents[0].mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
        
        return df

    def __load_isot_dataset(self, folder: Path):
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

    def __load_kaggle_dataset(self, folder: Path):
        """Load kaggle dataset.
        
        :param folder: folder to look for kaggle files
        :type folder: pathlib.Path
        
        :rtype: pd.Dataframe
        :return: kaggle dataset"""
        kaggle_df = pd.read_csv(folder / Path('train.csv'), header=0, index_col='id')
        kaggle_df['dataset'] = 'kaggle'
            
        return kaggle_df

    def __preprocess_dataset(self, df: pd.DataFrame):
        """Apply preprocessing operations to dataset.
        
        :param df: dataset
        :type dataset: pd.DataFrame"""
        # Remove break lines
        df['text'].str.replace('\n', ' ')
        
        # Filter articles too short
        text_length = df['text'].apply(len)
        df = df[text_length > 10]

        # Remove duplicates
        df = self.__remove_duplicates(df, 'text')
        
        return df

    def __build_duplicated_df(self, df: pd.DataFrame, field: str):
        vectorizer_tfidf = TfidfVectorizer(ngram_range = (1,3))
        X = vectorizer_tfidf.fit_transform(df[field])

        nbrs = NearestNeighbors(n_neighbors = 2, metric = 'cosine').fit(X)
        distances, indexes = nbrs.kneighbors(X, n_neighbors = 2)

        nearest_values = np.array(df.text)[indexes]

        d_ = {'text': nearest_values[:,0], 'distance': distances[:,1], 'best_match_index': indexes[:,1]}
        return pd.DataFrame(data=d_)

    def __remove_duplicates(self, df: pd.DataFrame, field: str, threshold: float = 0.3):
        df_no_duplicates = df.reset_index()
        similar_df = self.__build_duplicated_df(df_no_duplicates, field)
        similar_df = similar_df[similar_df['distance'] < threshold]

        groups = []
        seen = {}
        for row in similar_df.itertuples():
            if row.Index in seen:
                groups[seen[row.Index]].add(row.best_match_index)
                seen[row.best_match_index] = seen[row.Index]
            elif row.best_match_index in seen:
                groups[seen[row.best_match_index]].add(row.Index)
                seen[row.Index] = seen[row.best_match_index]
            else:
                groups.append(set([row.Index, row.best_match_index]))
                seen[row.Index] = len(groups) - 1
                seen[row.best_match_index] = len(groups) - 1


        indexes_to_remove = list(similar_df.index)
        for group in groups:
            indexes_to_remove.remove(group.pop())
        preprocessed_df = df_no_duplicates.reset_index().drop(index=indexes_to_remove)

        return preprocessed_df

if __name__ == '__main__':
    dataset = Dataset('../../data')

    print(dataset.load_dataset())
