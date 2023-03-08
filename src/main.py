import dataset_extractor as ds_extractor
import encoder

USERS_SIZE = 943
ITEMS_SIZE = 1682

def main():
    print('Welcome to the best Recommender System.\nEver.')
    dataset_extractor = ds_extractor.DatasetExtractor(USERS_SIZE, ITEMS_SIZE)
    urm = dataset_extractor.get_urm()

    onehot_users = encoder.get_onehot_encoding(USERS_SIZE)
    onehot_items = encoder.get_onehot_encoding(ITEMS_SIZE)


if __name__ == '__main__':
    main()