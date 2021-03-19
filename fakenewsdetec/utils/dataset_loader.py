import functools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from typing import List, Tuple

from dataset_functions import load_dataset

class Dataset:
    def __init__(self, data_folder):
        self.data_folder = Path(data_folder)

    def load_dataset(self, datasets: List[str] = ['berkeley']):
        """Load the specified dataset.
        
        :param datasets: list of datasets to load
        :type datasets: List, optional
        
        :rtype: pd.Dataframe
        :return: dataset
        """
        df = pd.DataFrame()

        combined_name = functools.reduce(lambda a,b: f"{a}_{b}", sorted(datasets))
        data_path_c = self.data_folder / 'processed/' / f"{combined_name}.csv"

        if data_path_c.is_file():
            df = pd.read_csv(data_path_c)
        else:
            for dataset in datasets:
                data_path = self.data_folder / 'processed/' / f"{dataset}.csv"

                if data_path.is_file():
                    df_ = pd.read_csv(data_path)
                    print(f"{dataset} already exists")
                else:
                    df_ = self.__build_dataset(data_path, dataset)

                df = pd.concat([df, df_])
            df = df.sample(frac=1).reset_index(drop=True)
            df = self.__preprocess_dataset(df)
            df.to_csv(data_path_c, index=False)

        train, rest = train_test_split(df, test_size=0.3)
        test, evaluation = train_test_split(rest, test_size=0.5)

        for type_ in ['train', 'test', 'eval']:
            path = self.data_folder / 'processed/' / f"{combined_name}/" / f"{type_}.csv"
            path.parents[0].mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
    
        return df

    def __build_dataset(self, path: Path, dataset: str):
        """Load the specified dataset.
        
        :param dataset: 0 loads isot + kaggle; 1 loads isot; 2 loads kaggle
        :type dataset: int, optional
        
        :rtype: pd.Dataframe
        :return: dataset
        """
        print(f"Build {dataset}")
        df = load_dataset[dataset](self.data_folder / f"{dataset}/")
            
        df['text'] = df['text'].astype(str)
        df['title'] = df['title'].astype(str)

        df = self.__preprocess_dataset(df)

        path.parents[0].mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        
        return df

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

    def __remove_duplicates(self, df: pd.DataFrame, field: str, threshold: float = 0.2):
        df_no_duplicates = df.reset_index(drop=True)
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
            index_to_keep = group.pop()
            if index_to_keep in indexes_to_remove:
                indexes_to_remove.remove(index_to_keep)
        preprocessed_df = df_no_duplicates.reset_index(drop=True).drop(index=indexes_to_remove)

        return preprocessed_df



if __name__ == '__main__':
    dataset = Dataset('../../data')

    print(dataset.load_dataset(['berkeley']))
