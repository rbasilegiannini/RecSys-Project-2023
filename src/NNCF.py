import neighborhood_builder as nb_builder
import embedding_builder as emb_builder
import MLP_dataset_builder as mlp_builder
import NNFF
import learning as learn
import encoder
import numpy as np
import error_functions as ef


class NNCF:
    """
    This class implements the Neighborhood-based Neural Collaborative Filtering model.
    """

    def __init__(self, urm, res, k, hidden_layers, neurons, activation, max_epoch):
        self.__MLP = None
        self.__user_item_concatenated_embeddings = None
        self.__urm = urm
        self.__res = res
        self.__latent_factor = k
        self.__number_hidden_layers = hidden_layers
        self.__neurons = neurons
        self.__activation = activation
        self.__max_epoch = max_epoch

        self.__learning_NNCF()

    def __run_integration_component(self):

        print("Run Integration Component's phase...", end="")

        # Retrieve neighborhoods and binary encoding
        num_of_users = self.__urm.shape[0]
        num_of_items = self.__urm.shape[1]

        [users_neighborhood, items_neighborhood] = nb_builder.extract_neighborhood(self.__urm, self.__res)
        binary_users_neighborhood = encoder.get_neighborhoods_encoding(users_neighborhood, num_of_items)
        binary_items_neighborhood = encoder.get_neighborhoods_encoding(items_neighborhood, num_of_users)

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
        # training_set = training_set[:1000, :]
        training_set_samples = training_set[:, :-1]
        training_set_labels = training_set[:, -1]

        print("Samples normalization...", end="")
        training_set_samples = learn.normalize_samples(training_set_samples, -0.5, 0.5)
        print(" Complete.")

        print("Training samples: " + str(len(training_set_samples)))

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
        self.__learning_MLP(training_set, self.__max_epoch)

    # Interface
    def get_recommendations(self, user, items_not_interacted, k):
        """
        This function returns the first k recommended items.

        :param user:
            The user who wants the recommendations.
        :param items_not_interacted:
            A list of items that the user has not interacted with.
        :param k:
            The number of recommendations to return.
        :return:
            A list with the first k recommended items sorted by best.
        """

        # For each non-interacting item build (u, i) interaction
        user_item_concatenated_embeddings = []
        for item in items_not_interacted:
            user_item_concatenated_embeddings.append(self.__user_item_concatenated_embeddings[user][item])

        # Retrieve K most probability items
        probability_items_list = []
        for user_item_concatenated_embedding in user_item_concatenated_embeddings:
            output_lines = self.__MLP.compute_network(user_item_concatenated_embedding)[1][-1]
            probability_items_list.append((ef.softmax(output_lines)).max())

        # Sort in decreasing order
        best_items = np.argsort(probability_items_list)[::-1]
        if k < best_items.size:
            best_items = best_items[:k]
        recommendations = []

        # Retrieve recommendations
        for best_item in best_items:
            recommendations.append(items_not_interacted[best_item])

        return recommendations





