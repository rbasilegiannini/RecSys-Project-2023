import neighborhood_builder as nb_builder
import embedding_builder as emb_builder
import MLP_dataset_builder as mlp_builder
import NNFF
import learning as learn
import encoder
import numpy as np

USERS_SIZE = 943
ITEMS_SIZE = 1682


class NNCF:

    def __init__(self, urm, res, k, hidden_layers, neurons, activation):
        self.__MLP = None
        self.__user_item_concatenated_embeddings = None
        self.__urm = urm
        self.__res = res
        self.__latent_factor = k
        self.__number_hidden_layers = hidden_layers
        self.__neurons = neurons
        self.__activation = activation

        self.__learning_NNCF()

    def __run_integration_component(self):

        print("Run Integration Component's phase...", end="")

        # Retrieve neighborhoods and binary encoding
        [users_neighborhood, items_neighborhood] = nb_builder.extract_neighborhood(self.__urm, self.__res)
        binary_users_neighborhood = encoder.get_neighborhoods_encoding(users_neighborhood, ITEMS_SIZE)
        binary_items_neighborhood = encoder.get_neighborhoods_encoding(items_neighborhood, USERS_SIZE)

        # Compute latent vectors
        [users_latent_vector, items_latent_vector] = emb_builder.get_latent_vectors(self.__urm, self.__latent_factor)

        # Compute the interaction function for pair <user,item>
        interaction_functions = emb_builder.get_interaction_functions(
            users_latent_vector, items_latent_vector, self.__latent_factor)

        # Retrieve the neighborhood embedding
        users_neighborhood_embedding = emb_builder.get_neighborhoods_embedding(binary_users_neighborhood,
                                                                               items_latent_vector)
        items_neighborhood_embedding = emb_builder.get_neighborhoods_embedding(binary_items_neighborhood,
                                                                               users_latent_vector)

        # Concatenate the three embeddings
        self.__user_item_concatenated_embeddings = emb_builder.get_concatenated_embeddings(
            interaction_functions, users_neighborhood_embedding, items_neighborhood_embedding)

        print(" Complete.")

    def __learning_MLP(self, training_set, max_epoch):

        np.random.shuffle(training_set)
        training_set_samples = training_set[:, :-1]
        training_set_labels = training_set[:, -1]

        print("Samples normalization...", end="")
        training_set_samples = learn.normalize_samples(training_set_samples, -0.5, 0.5)
        print(" Complete.")

        # test_set_size = round(0.3 * samples.shape[0])
        # training_set_size = samples.shape[0] - test_set_size

        # training_set_samples = samples[:training_set_size, :]
        # training_set_labels = labels[:training_set_size]

        # test_set_samples = samples[training_set_size:, :]

        print("Training samples: " + str(len(training_set_samples)))
        # print("Test samples: " + str(len(test_set_samples)))

        # Convert labels in one-hot encoding
        training_labels_one_hot = encoder.get_binary_one_hot_labels(training_set_labels)

        # Learning
        hidden_layers = []
        for layer in range(self.__number_hidden_layers):
            hidden_layers.append(self.__neurons)
        input_dim = training_set_samples.shape[1]

        self.__MLP = NNFF.NeuralNetworkFF(input_dim,
                                          5,
                                          hidden_layers,
                                          2,
                                          self.__activation,
                                          bias=0)

        self.__MLP = learn.learning(self.__MLP, max_epoch, training_set_samples, training_labels_one_hot)

    def __learning_NNCF(self):

        self.__run_integration_component()
        training_set = mlp_builder.get_training_set(self.__urm, self.__user_item_concatenated_embeddings)
        self.__learning_MLP(training_set, 200)

    def get_recommendations(self, user, items_not_interacted, k):

        # For each non-interacting item build (u, i) interaction
        user_item_concatenated_embeddings = []
        for item in items_not_interacted:
            user_item_concatenated_embeddings.append(self.__user_item_concatenated_embeddings[user][item])

        # Retrieve K most probability items
        probability_items_list = []
        for user_item_concatenated_embedding in user_item_concatenated_embeddings:
            probability_item = self.__MLP.compute_network(user_item_concatenated_embedding)
            probability_items_list.append(probability_item)

        # Sort in decreasing order. "argsort" because we want the items' id
        recommendations = np.argsort(probability_items_list)[::-1]

        return recommendations[:k]





