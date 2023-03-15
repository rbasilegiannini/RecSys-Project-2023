import dataset_extractor as ds_extractor
import neighborhood_builder as nb_builder
import embedding_builder as emb_builder

import encoder

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

    binary_users_neighborhood = []
    for neighborhood in users_neighborhood:
        binary_users_neighborhood.append(
            encoder.get_binary_encoding(ITEMS_SIZE, neighborhood)
        )

    binary_items_neighborhood = []
    for neighborhood in items_neighborhood:
        binary_items_neighborhood.append(
            encoder.get_binary_encoding(USERS_SIZE, neighborhood)
        )

    # Compute latent vectors
    embedding = emb_builder.Embedding(urm, hyperparams['k'])
    user_latent_vectors = embedding.get_user_embeddings()
    item_latent_vectors = embedding.get_item_embeddings()

    del embedding   # Optimization


if __name__ == '__main__':
    main()
