import pandas as pd
import os


def load_data(args):
    path = os.path.join('data', args.data)
    dataset = pd.read_csv(path + '.csv')
    data = dataset.values
    X = data[:, 1:]
    y = data[:, 0]
    return X, y
