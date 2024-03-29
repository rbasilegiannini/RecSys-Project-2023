import numpy as np
import time
import dataset_extractor as ds_extractor
import URM_manager
import MLP_training_set_builder as ts_builder
import NNCF

USERS_SIZE = 943
ITEMS_SIZE = 1682


hyperparams = {
    "res": 0.5,
    "k": 10,
    "hidden layers": 1,
    "neurons": 5,
    "activation": 'leaky_relu',
    "max epochs": 100,
    "kernels": 16
}


def build_recsys(urm, items_to_avoid):
    # Building and learning NNCF
    net = NNCF.NNCF(urm,
                    hyperparams['res'],
                    hyperparams['k'],
                    hyperparams['hidden layers'],
                    hyperparams['neurons'],
                    hyperparams['activation']
                    )

    run_int_comp_time_start = time.time()
    user_item_concatenated_embeddings = net.run_integration_component(hyperparams['kernels'])
    run_int_comp_time_end = time.time()
    run_int_comp_time = run_int_comp_time_end - run_int_comp_time_start
    print("Integration component time: " + str(round(run_int_comp_time, 2)) + "s")

    training_set = ts_builder.get_training_set(urm, user_item_concatenated_embeddings, items_to_avoid)
    net.learning_MLP(training_set, hyperparams['max epochs'])

    return net


def evaluate_recsys(net, top_k, test_items, not_interacted_items):
    hit20 = 0
    hit10 = 0
    hit5 = 0

    for user in range(USERS_SIZE):

        recommendations_20 = net.get_recommendations(user, not_interacted_items[user], top_k)
        recommendations_10 = recommendations_20[:10]
        recommendations_5 = recommendations_20[:5]

        if test_items[user] in recommendations_20:
            hit20 += 1

        if test_items[user] in recommendations_10:
            hit10 += 1

        if test_items[user] in recommendations_5:
            hit5 += 1

    hr20 = round((hit20 / USERS_SIZE) * 100, 2)
    hr10 = round((hit10 / USERS_SIZE) * 100, 2)
    hr5 = round((hit5 / USERS_SIZE) * 100, 2)

    print("HR@" + str(20) + ": " + str(hr20) + "%")
    print("HR@" + str(10) + ": " + str(hr10) + "%")
    print("HR@" + str(5) + ": " + str(hr5) + "%")

    return [hr5, hr10, hr20]


def main():
    hr5_list = []
    hr10_list = []
    hr20_list = []

    top_k = 20

    # Retrieve URM and Test items list
    print("URM extraction...", end="")
    urm_manager = URM_manager.URMManager(USERS_SIZE, ITEMS_SIZE)
    urm = urm_manager.get_urm()
    test_items = urm_manager.get_test_items()

    # Retrieve the items to avoid in the learning task (they will be used for recommendation)
    not_interacted_items = np.ndarray([USERS_SIZE, 101], dtype=int)
    for user in range(USERS_SIZE):
        not_interacted_items[user] = urm_manager.get_not_interacted_items_for_recommendation(user)

    print(" Complete.")

    for attempt in range(2):
        recsys = build_recsys(urm, not_interacted_items)
        hr_values = evaluate_recsys(recsys, top_k, test_items, not_interacted_items)
        hr5_list.append(hr_values[0])
        hr10_list.append(hr_values[1])
        hr20_list.append(hr_values[2])

    hr5_values = np.array(hr5_list)
    hr10_values = np.array(hr10_list)
    hr20_values = np.array(hr20_list)

    print("HR@20 mean: " + str(hr20_values.mean()) + "%")
    print("HR@10 mean: " + str(hr10_values.mean()) + "%")
    print("HR@5 mean: " + str(hr5_values.mean()) + "%")


if __name__ == '__main__':
    # tf.config.set_visible_devices([], 'GPU')  # For Apple Silicon
    main()

