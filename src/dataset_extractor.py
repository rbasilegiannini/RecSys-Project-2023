class DatasetExtractor:
    def __init__(self):
        pass

    def get_urm(self):
        dataset_file = open('/res/u.data', 'r')
        dataset_text = dataset_file.read()
