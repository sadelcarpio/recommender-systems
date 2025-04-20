from src.item_item.item_item import ItemItemModel
from src.data import Dataset

if __name__ == '__main__':
    movielens_subset = Dataset("movielens-20m-dataset/rating.csv", n_most_users=1000, m_most_items=200)
    train_ds, test_ds = movielens_subset.split_dataset(test_ratio=0.2)
    print(f"Number of instances on train: {len(train_ds)}")
    print(f"Number of instances on test: {len(test_ds)}")
    model = ItemItemModel(min_common_users=5, max_common_weights=25)
    model.fit(train_ds)
    train_predictions = model.predict(train_ds)
    print(f"Number of valid predictions on train: {train_predictions.count()}")
    train_mse = model.evaluate_mse(train_ds["rating"], train_predictions)
    test_predictions = model.predict(test_ds)
    print(f"Number of valid predictions on test: {test_predictions.count()}")
    test_mse = model.evaluate_mse(test_ds["rating"], test_predictions)
    print(f"MSE on train set: {train_mse}")
    print(f"MSE on test set: {test_mse}")
