import matplotlib.pyplot as plt
import numpy as np

import dataset_extractor as ds_extractor
import neighborhood_builder as nb_builder
import embedding_builder as emb_builder
import learning as learn
import encoder
import NNFF
import MLP_dataset_builder as mlp_builder


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

    # Retrieve URM and one-hot encoding
    print("URM extraction...", end="")
    dataset_extractor = ds_extractor.DatasetExtractor(USERS_SIZE, ITEMS_SIZE)
    urm = dataset_extractor.get_urm()

    onehot_users = encoder.get_onehot_encoding(USERS_SIZE)
    onehot_items = encoder.get_onehot_encoding(ITEMS_SIZE)
    print(" Complete.")

    # Retrieve neighborhoods and binary encoding
    print("Neighborhoods extraction...", end="")
    [users_neighborhood, items_neighborhood] = nb_builder.extract_neighborhood(urm, hyperparams['res'])
    binary_users_neighborhood = encoder.get_neighborhoods_encoding(users_neighborhood, ITEMS_SIZE)
    binary_items_neighborhood = encoder.get_neighborhoods_encoding(items_neighborhood, USERS_SIZE)
    print(" Complete.")

    # Compute latent vectors
    print("Compute latent vectors (users, items)...", end="")
    [users_latent_vector, items_latent_vector] = emb_builder.get_latent_vectors(urm,  hyperparams['k'])
    print(" Complete.")

    print("Run Integration Component's phase...", end="")
    # Compute the interaction function for pair <user,item>
    interaction_functions = emb_builder.get_interaction_functions(
        users_latent_vector, items_latent_vector, hyperparams['k'])

    # Retrieve the neighborhood embedding
    users_neighborhood_embedding = emb_builder.get_neighborhoods_embedding(binary_users_neighborhood, items_latent_vector)
    items_neighborhood_embedding = emb_builder.get_neighborhoods_embedding(binary_items_neighborhood, users_latent_vector)

    # Concatenate the three embeddings
    user_item_concatenated_embeddings = emb_builder.get_concatenated_embeddings(
        interaction_functions, users_neighborhood_embedding, items_neighborhood_embedding)
    print(" Complete.")

    # Get the training set formatted for the MLP
    training_set = mlp_builder.get_training_set(urm, user_item_concatenated_embeddings)

    # Test of integration component and prediction component combination
    test_neural_network(training_set)


def test_neural_network(training_set):

    num_test = 3
    accuracies = []
    for i in range(num_test):
        print("\nRUN TEST NUMBER: " + str(i+1) + "\n")

        np.random.shuffle(training_set)
        training_set = training_set[:200000, :]
        samples = training_set[:, :-1]
        labels = training_set[:, -1]

        print("Samples normalization...", end="")
        # samples = learn.normalize_samples(samples, -0.5, 0.5)
        print(" Complete.")

        test_set_size = round(0.3 * samples.shape[0])
        training_set_size = samples.shape[0] - test_set_size

        training_set_samples = samples[:training_set_size, :]
        training_set_labels = labels[:training_set_size]

        test_set_samples = samples[training_set_size:, :]
        test_set_labels = labels[training_set_size:]

        print("Training samples: " + str(len(training_set_samples)))
        print("Test samples: " + str(len(test_set_samples)))

        # Convert labels in one-hot encoding
        training_labels_one_hot = encoder.get_binary_one_hot_labels(training_set_labels)
        test_labels_one_hot = encoder.get_binary_one_hot_labels(test_set_labels)

        # Learning
        NN = NNFF.NeuralNetworkFF(samples.shape[1],
                                  5,
                                  [5],
                                  2,
                                  'sigmoid',
                                  bias=0)

        NN = learn.learning(NN, 200, training_set_samples, training_labels_one_hot)
        acc = learn.accuracy(NN, test_set_samples, test_labels_one_hot)

        print('\nAccuracy: ' + str(acc) + '%')
        accuracies.append(acc)

    x = range(num_test)
    plt.plot(x, accuracies)
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

    accurray = np.array(accuracies)
    print(str(round(accurray.mean(), 2)) + '%')


if __name__ == '__main__':
    main()
