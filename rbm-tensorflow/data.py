import pandas as pd
import tensorflow as tf


class Dataset:
    """
    Common dataset structure for collaborative filtering recommendations on MovieLens dataset.
    Structure: userId, movieId, rating
    """
    def __init__(self, dataset_csv_path: str, n_most_users: int, m_most_items: int):
        self.dataset = pd.read_csv(dataset_csv_path)
        most_common_users = self.dataset.userId.value_counts().head(n_most_users)
        most_common_movies = self.dataset.movieId.value_counts().head(m_most_items)
        self.dataset = self.dataset[(self.dataset.userId.isin(most_common_users.index)) &
                                    (self.dataset.movieId.isin(most_common_movies.index))]
        self.dataset['userIdOrdered'] = self.dataset['userId'].astype('category').cat.codes
        self.dataset['movieIdOrdered'] = self.dataset['movieId'].astype('category').cat.codes
        self.dataset = self.dataset[["userIdOrdered", "movieIdOrdered", "rating"]]
        self.n_users = self.dataset['userIdOrdered'].max() + 1
        self.n_items = self.dataset['movieIdOrdered'].max() + 1

    def sparse_dataset(self, test_ratio: float, batch_size: int) -> tuple:
        """
        Creates a shuffled sparse dataset split into train and test based on the test_ratio arg
        :param test_ratio: Ratio of test samples, between 0 - 1
        :param batch_size: Batch size
        :return: tuple of train_dataset, test_dataset
        """
        ratings_map = {0.5: 0, 1: 1, 1.5: 2, 2: 3, 2.5: 4, 3: 5, 3.5: 6, 4: 7, 4.5: 8, 5: 9}
        indices = list(zip(self.dataset['userIdOrdered'], self.dataset['movieIdOrdered'], self.dataset['rating'].map(ratings_map)))
        values = tf.ones(len(indices))
        sparse_dataset = tf.SparseTensor(indices=indices, values=values, dense_shape=(self.n_users, self.n_items, 10))
        sparse_ds = tf.data.Dataset.from_tensor_slices((sparse_dataset, sparse_dataset)).batch(batch_size)
        train_size = int((1 - test_ratio) * sparse_ds.cardinality().numpy())
        val_size = sparse_ds.cardinality().numpy() - train_size
        sparse_train = sparse_ds.take(train_size)
        sparse_test = sparse_ds.skip(train_size).take(val_size)
        return sparse_train, sparse_test
