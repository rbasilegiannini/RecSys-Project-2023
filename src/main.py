import dataset_extractor as ds_extractor
import MLP_training_set_builder as ds_builder


import NNCF

USERS_SIZE = 943
ITEMS_SIZE = 1682


hyperparams = {
    "res": 0.5,
    "k": 3,
    "hidden layers": 1,
    "neurons": 1,
    "activation": 'sigmoid',
    "max epochs": 1
}


def main():
    print('Welcome to the best Recommender System.\nEver.')
    print()

    # Retrieve URM and Test items list
    print("URM extraction...", end="")
    dataset_extractor = ds_extractor.DatasetExtractor(USERS_SIZE, ITEMS_SIZE)
    urm = dataset_extractor.get_urm()
    test_items = dataset_extractor.get_test_items()
    print(" Complete.")

    # Building and learning NNCF
    net = NNCF.NNCF(urm,
                    hyperparams['res'],
                    hyperparams['k'],
                    hyperparams['hidden layers'],
                    hyperparams['neurons'],
                    hyperparams['activation']
                    )

    user_item_concatenated_embeddings = net.run_integration_component()
    training_set = ds_builder.get_training_set(urm, user_item_concatenated_embeddings, test_items)
    net.learning_MLP(training_set, hyperparams['max epochs'])

    # Testing
    top_k = 150
    evaluate_recsys(dataset_extractor, net, top_k, test_items)


def evaluate_recsys(dataset_extractor, net, top_k, test_items):
    hit = 0
    for user in range(USERS_SIZE):
        items_not_interacted = dataset_extractor.get_not_interacted_items(user)
        recommendations = net.get_recommendations(user, items_not_interacted, top_k)

        if test_items[user] in recommendations:
            hit += 1

    hrk = round((hit / USERS_SIZE) * 100, 2)

    print("HR@" + str(top_k) + ": " + str(hrk) + "%")

    return hrk


if __name__ == '__main__':
    main()
