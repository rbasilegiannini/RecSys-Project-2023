import dataset_extractor as ds_extractor
import NNCF

USERS_SIZE = 943
ITEMS_SIZE = 1682

hyperparams = {
    "res": 0.01,
    "k": 10,
    "hidden layers": 1,
    "neurons": 5,
    "activation": 'sigmoid'
}


def main():
    print('Welcome to the best Recommender System.\nEver.')
    print()

    # Retrieve URM and Test items list
    print("URM extraction...", end="")
    dataset_extractor = ds_extractor.DatasetExtractor(USERS_SIZE, ITEMS_SIZE)
    urm = dataset_extractor.get_urm()
    # test_items = dataset_extractor.get_test_items()
    print(" Complete.")

    net = NNCF.NNCF(urm,
                    hyperparams['res'],
                    hyperparams['k'],
                    hyperparams['hidden layers'],
                    hyperparams['neurons'],
                    hyperparams['activation'])

    # DEBUG
    items_not_interacted = [50, 123, 43, 65, 89, 23, 22, 19, 10]
    recommendations = net.get_recommendations(10, items_not_interacted, 5)

    print(recommendations)


if __name__ == '__main__':
    main()
