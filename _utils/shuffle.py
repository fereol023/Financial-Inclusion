import pandas as pd


def shuffle(csv_file: str, k: int = None):
    """To shuffle a pd.DataFrame data and split into requested numbers and save."""
    test_data = pd.read_csv(csv_file)
    assert isinstance(test_data, pd.DataFrame), f'Erreur while loading data.'

    test_data = test_data.sample(frac=1, ignore_index=True)

    if k is not None:
        assert isinstance(k, int), f'k must be an integer'
        print(f'Data shape : {test_data.shape}')
        each = test_data.shape[0] // k
        print(f'data will be split in {k} periods of {each} rows each.')

        for i in range(k):
            # print(each*i, each*(i+1))
            p = test_data.iloc[each*i:each*(i+1), :]

            p.to_csv(f'drift/Period_{i+1}.csv')

    print('Done !')


if __name__ == '__main__':

    PATH = '../_data/drift/Period0.csv'
    shuffle(PATH, k=3)
