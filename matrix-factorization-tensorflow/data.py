import pandas as pd


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

    def split_dataset(self, test_ratio: float) -> tuple:
        """
        Splits dataset randomly into train, test.
        :param test_ratio: Ratio of test samples, between 0 - 1
        :return: tuple of train_dataset, test_dataset
        """
        dataset_train = self.dataset.sample(frac=(1 - test_ratio), random_state=42)
        dataset_test = self.dataset.drop(dataset_train.index)
        return dataset_train, dataset_test
