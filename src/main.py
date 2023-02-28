import dataset_extractor as ds_extractor


def main():
    print('Welcome to the best Recommender System.\nEver.')
    dataset_extractor = ds_extractor.DatasetExtractor()
    urm = dataset_extractor.get_urm()


if __name__ == '__main__':
    main()
