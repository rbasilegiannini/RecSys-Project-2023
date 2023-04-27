import numpy as np

import dataset_extractor as ds_extractor
import neighborhood_builder as nb_builder
import embedding_builder as emb_builder
import learning as learn
import encoder
import NNFF

from sklearn.datasets import load_breast_cancer
import sklearn.utils


USERS_SIZE = 943
ITEMS_SIZE = 1682

hyperparams = {
    "resolution": 1,
    "k": 10
}


def main():
    print('Welcome to the best Recommender System.\nEver.')

    # Retrieve URM and one-hot encoding
    dataset_extractor = ds_extractor.DatasetExtractor(USERS_SIZE, ITEMS_SIZE)
    urm = dataset_extractor.get_urm()

    onehot_users = encoder.get_onehot_encoding(USERS_SIZE)
    onehot_items = encoder.get_onehot_encoding(ITEMS_SIZE)

    # Retrieve neighborhoods and binary encoding
    [users_neighborhood, items_neighborhood] = nb_builder.extract_neighborhood(urm, hyperparams['resolution'])
    binary_users_neighborhood = encoder.get_neighborhoods_encoding(users_neighborhood, ITEMS_SIZE)
    binary_items_neighborhood = encoder.get_neighborhoods_encoding(items_neighborhood, USERS_SIZE)

    # Compute latent vectors
    [users_latent_vector, items_latent_vector] = emb_builder.get_latent_vectors(urm,  hyperparams['k'])

    # Compute the interaction function for pair <user,item>
    interaction_functions = emb_builder.get_interaction_functions(
        users_latent_vector, items_latent_vector, hyperparams['k'])

    # Retrieve the neighborhood embedding
    users_neighborhood_embedding = emb_builder.get_neighborhoods_embedding(binary_users_neighborhood, items_latent_vector)
    items_neighborhood_embedding = emb_builder.get_neighborhoods_embedding(binary_items_neighborhood, users_latent_vector)

    # Concatenate the three embeddings
    user_item_concatenated_embeddings = emb_builder.get_concatenated_embeddings(
        interaction_functions, users_neighborhood_embedding, items_neighborhood_embedding)


    training_set = get_training_set(urm, user_item_concatenated_embeddings)



def test_neural_network():

    for i in range(10):

        # Compose dataset
        [samples, labels] = load_breast_cancer(return_X_y=True)
        [samples, labels] = sklearn.utils.shuffle(samples, labels)

        samples = learn.normalize_samples(samples, -0.5, 0.5)

        test_set_size = round(0.3 * samples.shape[0])
        training_set_size = samples.shape[0] - test_set_size

        training_set_samples = samples[:training_set_size, :]
        training_set_labels = labels[:training_set_size]

        test_set_samples = samples[(training_set_size+1):, :]
        test_set_labels = labels[(training_set_size+1):]

        # Convert labels in one-hot encoding
        training_labels_one_hot = encoder.get_binary_one_hot_labels(training_set_labels)
        test_labels_one_hot = encoder.get_binary_one_hot_labels(test_set_labels)

        # Learning
        NN = NNFF.NeuralNetworkFF(30,
                                  15,
                                  [5],
                                  2,
                                  'sigmoid',
                                  bias=0)

        NN = learn.learning(NN, 150, training_set_samples, training_labels_one_hot)
        acc = learn.accuracy(NN, test_set_samples, test_labels_one_hot)

        print('Accuracy: ' + str(acc) + '%')


if __name__ == '__main__':
    main()
    # test_neural_network()
