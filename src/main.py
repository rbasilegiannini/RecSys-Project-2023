import numpy as np

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
    binary_users_neighborhood = encoder.get_neighborhoods_encoding(users_neighborhood, ITEMS_SIZE)
    binary_items_neighborhood = encoder.get_neighborhoods_encoding(items_neighborhood, USERS_SIZE)

    # Compute latent vectors
    embedding = emb_builder.Embedding(urm, hyperparams['k'])
    users_latent_vector = embedding.get_user_embeddings()
    items_latent_vector = embedding.get_item_embeddings()

    del embedding   # Optimization


if __name__ == '__main__':
    main()
