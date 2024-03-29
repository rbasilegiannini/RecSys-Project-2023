import neighborhood_builder as nb_builder
import embedding_builder as emb_builder
import NNFF
import learning as learn
import encoder
import numpy as np
import error_functions as ef


class NNCF:
    """
    This class implements the Neighborhood-based Neural Collaborative Filtering model.
    """

    def __init__(self, urm, res, k, hidden_layers, neurons, activation):
        self.__MLP = None
        self.__user_item_concatenated_embeddings = None
        self.__urm = urm
        self.__res = res
        self.__latent_factor = k
        self.__number_hidden_layers = hidden_layers
        self.__neurons = neurons
        self.__activation = activation

    def run_integration_component(self, kernels=8):
        """
        This method run the integration component phase.

        :return:
            The user-item concatenated embeddings.
        """

        print("Run Integration Component's phase...", end="")

        # Retrieve neighborhoods and binary encoding
        num_of_users = self.__urm.shape[0]
        num_of_items = self.__urm.shape[1]

        [users_neighborhood, items_neighborhood] = nb_builder.extract_urm_neighborhoods(self.__urm, self.__res)
        binary_users_neighborhood = encoder.get_neighborhoods_encoding(users_neighborhood, num_of_items)
        binary_items_neighborhood = encoder.get_neighborhoods_encoding(items_neighborhood, num_of_users)

        # Compute latent vectors
        [users_latent_vector, items_latent_vector] = emb_builder.get_latent_vectors(self.__urm, self.__latent_factor)

        # Compute the interaction function for pair <user,item>
        interaction_functions = emb_builder.get_interaction_functions(
            users_latent_vector, items_latent_vector, self.__latent_factor)

        # Retrieve the neighborhood embedding
        users_neighborhood_embedding = emb_builder.get_neighborhoods_embedding(binary_users_neighborhood,
                                                                               items_latent_vector, kernels)
        items_neighborhood_embedding = emb_builder.get_neighborhoods_embedding(binary_items_neighborhood,
                                                                               users_latent_vector, kernels)

        # Concatenate the three embeddings
        self.__user_item_concatenated_embeddings = emb_builder.get_concatenated_embeddings(
            interaction_functions, users_neighborhood_embedding, items_neighborhood_embedding)

        print(" Complete.")

        return self.__user_item_concatenated_embeddings

    def learning_MLP(self, training_set, max_epochs=10):
        """
        This method performs the learning of the Multi-Layer Perceptron.
        It's mandatory to run the integration component phase first.

        :param training_set:
            The training set.
        :param max_epochs:
            The max number of epochs (default = 10)

        """

        # Check integration component phase
        if self.__user_item_concatenated_embeddings is None:
            print("[NNCF] Run integration component phase is mandatory!")
            return

        np.random.shuffle(training_set)
        training_set_samples = training_set[:, :-1]
        training_set_labels = training_set[:, -1]

        print("Training samples: " + str(len(training_set_samples)))

        # Learning
        hidden_layers = []
        for layer in range(self.__number_hidden_layers):
            hidden_layers.append(self.__neurons)
        input_dim = training_set_samples.shape[1]

        self.__MLP = NNFF.NeuralNetworkFF(input_dim,
                                          hidden_layers,
                                          1,
                                          self.__activation,
                                          bias=0)

        self.__MLP = learn.learning(self.__MLP, max_epochs, training_set_samples, training_set_labels)

    def get_recommendations(self, user, not_interacted_items, k):
        """
        This function returns the first k recommended items.

        :param user:
            The user who wants the recommendations.
        :param not_interacted_items:
            A list of items that the user has not interacted with.
        :param k:
            The number of recommendations to return.
        :return:
            A list with the first k recommended items sorted by best.
        """

        # Check MLP
        if self.__MLP is None:
            print("[NNCF] Learning MLP is mandatory!")
            return

        # For each non-interacted item build (u, i) interaction
        user_item_concatenated_embeddings = []
        for item in not_interacted_items:
            user_item_concatenated_embeddings.append(self.__user_item_concatenated_embeddings[user][item])

        # Retrieve K most probability items
        items_interaction_probabilities = []
        for user_item_concatenated_embedding in user_item_concatenated_embeddings:
            output = self.__MLP.compute_network(user_item_concatenated_embedding)[1][-1][0]
            items_interaction_probabilities.append(output)

        # Sort in decreasing order
        best_items = np.argsort(items_interaction_probabilities)[::-1]
        if k < best_items.size:
            best_items = best_items[:k]
        recommendations = []

        # Retrieve recommendations
        for best_item in best_items:
            recommendations.append(not_interacted_items[best_item])

        return recommendations





