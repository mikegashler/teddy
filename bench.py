from argparse import ArgumentParser
import numpy as np
import teddy as td
import pandas as pd
import time

def pd_time_row_access(n_slices: int) -> float:
    x = pd.DataFrame({
        'c0': pd.Categorical(list('ababababab'), categories=['a', 'b'], ordered=True),
        'c1': [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9],
        'c2': pd.Categorical(list('ababababab'), categories=['a', 'b'], ordered=True),
        'c3': [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9],
        'c4': pd.Categorical(list('ababababab'), categories=['a', 'b'], ordered=True),
        'c5': [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9],
        'c6': pd.Categorical(list('ababababab'), categories=['a', 'b'], ordered=True),
        'c7': [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9],
        'c8': pd.Categorical(list('ababababab'), categories=['a', 'b'], ordered=True),
        'c9': [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9],
    })
    t = time.time()
    for i in range(n_slices):
        y = x.iloc[i % 10]
    return time.time() - t

def pd_time_col_access(n_slices: int) -> float:
    x = pd.DataFrame({
        'c0': pd.Categorical(list('ababababab'), categories=['a', 'b'], ordered=True),
        'c1': [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9],
        'c2': pd.Categorical(list('ababababab'), categories=['a', 'b'], ordered=True),
        'c3': [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9],
        'c4': pd.Categorical(list('ababababab'), categories=['a', 'b'], ordered=True),
        'c5': [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9],
        'c6': pd.Categorical(list('ababababab'), categories=['a', 'b'], ordered=True),
        'c7': [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9],
        'c8': pd.Categorical(list('ababababab'), categories=['a', 'b'], ordered=True),
        'c9': [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9],
    })
    c = [ str(c) for c in x.columns ]
    t = time.time()
    for i in range(n_slices):
        y = x[c[i % 10]]
    return time.time() - t

def np_time_row_access(n_slices: int) -> float:
    x = np.array([
        ('a', 0.0, 'a', 0.0, 'a', 0.0, 'a', 0.0, 'a', 0.0),
        ('b', 1.1, 'a', 1.1, 'a', 1.1, 'a', 1.1, 'a', 1.1),
        ('a', 0.0, 'a', 0.0, 'a', 0.0, 'a', 0.0, 'a', 0.0),
        ('b', 1.1, 'a', 1.1, 'a', 1.1, 'a', 1.1, 'a', 1.1),
        ('a', 0.0, 'a', 0.0, 'a', 0.0, 'a', 0.0, 'a', 0.0),
        ('b', 1.1, 'a', 1.1, 'a', 1.1, 'a', 1.1, 'a', 1.1),
        ('a', 0.0, 'a', 0.0, 'a', 0.0, 'a', 0.0, 'a', 0.0),
        ('b', 1.1, 'a', 1.1, 'a', 1.1, 'a', 1.1, 'a', 1.1),
        ('a', 0.0, 'a', 0.0, 'a', 0.0, 'a', 0.0, 'a', 0.0),
        ('b', 1.1, 'a', 1.1, 'a', 1.1, 'a', 1.1, 'a', 1.1),
    ], dtype = [
        ('c0', 'S10'),
        ('c1', float),
        ('c2', 'S10'),
        ('c3', float),
        ('c4', 'S10'),
        ('c5', float),
        ('c6', 'S10'),
        ('c7', float),
        ('c8', 'S10'),
        ('c9', float)
    ])
    t = time.time()
    for i in range(n_slices):
        y = x[i % 10]
    return time.time() - t

def np_time_col_access(n_slices: int) -> float:
    x = np.array([
        ('a', 0.0, 'a', 0.0, 'a', 0.0, 'a', 0.0, 'a', 0.0),
        ('b', 1.1, 'a', 1.1, 'a', 1.1, 'a', 1.1, 'a', 1.1),
        ('a', 0.0, 'a', 0.0, 'a', 0.0, 'a', 0.0, 'a', 0.0),
        ('b', 1.1, 'a', 1.1, 'a', 1.1, 'a', 1.1, 'a', 1.1),
        ('a', 0.0, 'a', 0.0, 'a', 0.0, 'a', 0.0, 'a', 0.0),
        ('b', 1.1, 'a', 1.1, 'a', 1.1, 'a', 1.1, 'a', 1.1),
        ('a', 0.0, 'a', 0.0, 'a', 0.0, 'a', 0.0, 'a', 0.0),
        ('b', 1.1, 'a', 1.1, 'a', 1.1, 'a', 1.1, 'a', 1.1),
        ('a', 0.0, 'a', 0.0, 'a', 0.0, 'a', 0.0, 'a', 0.0),
        ('b', 1.1, 'a', 1.1, 'a', 1.1, 'a', 1.1, 'a', 1.1),
    ], dtype = [
        ('c0', 'S10'),
        ('c1', float),
        ('c2', 'S10'),
        ('c3', float),
        ('c4', 'S10'),
        ('c5', float),
        ('c6', 'S10'),
        ('c7', float),
        ('c8', 'S10'),
        ('c9', float)
    ])
    c = [ str(c) for c in x.dtype.names ]
    t = time.time()
    for i in range(n_slices):
        y = x[c[i % 10]]
    return time.time() - t

def td_time_row_access(n_slices: int) -> float:
    x = td.init_2d([
        ('c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'),
        ('a',  0.0,  'a',  0.0,  'a',  0.0,  'a',  0.0,  'a',  0.0),
        ('b',  1.1,  'a',  1.1,  'a',  1.1,  'a',  1.1,  'a',  1.1),
        ('a',  0.0,  'a',  0.0,  'a',  0.0,  'a',  0.0,  'a',  0.0),
        ('b',  1.1,  'a',  1.1,  'a',  1.1,  'a',  1.1,  'a',  1.1),
        ('a',  0.0,  'a',  0.0,  'a',  0.0,  'a',  0.0,  'a',  0.0),
        ('b',  1.1,  'a',  1.1,  'a',  1.1,  'a',  1.1,  'a',  1.1),
        ('a',  0.0,  'a',  0.0,  'a',  0.0,  'a',  0.0,  'a',  0.0),
        ('b',  1.1,  'a',  1.1,  'a',  1.1,  'a',  1.1,  'a',  1.1),
        ('a',  0.0,  'a',  0.0,  'a',  0.0,  'a',  0.0,  'a',  0.0),
        ('b',  1.1,  'a',  1.1,  'a',  1.1,  'a',  1.1,  'a',  1.1),
    ]).data
    t = time.time()
    for i in range(n_slices):
        y = x[i % 10]
    return time.time() - t

def td_time_col_access(n_slices: int) -> float:
    x = td.init_2d([
        ('c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'),
        ('a',  0.0,  'a',  0.0,  'a',  0.0,  'a',  0.0,  'a',  0.0),
        ('b',  1.1,  'a',  1.1,  'a',  1.1,  'a',  1.1,  'a',  1.1),
        ('a',  0.0,  'a',  0.0,  'a',  0.0,  'a',  0.0,  'a',  0.0),
        ('b',  1.1,  'a',  1.1,  'a',  1.1,  'a',  1.1,  'a',  1.1),
        ('a',  0.0,  'a',  0.0,  'a',  0.0,  'a',  0.0,  'a',  0.0),
        ('b',  1.1,  'a',  1.1,  'a',  1.1,  'a',  1.1,  'a',  1.1),
        ('a',  0.0,  'a',  0.0,  'a',  0.0,  'a',  0.0,  'a',  0.0),
        ('b',  1.1,  'a',  1.1,  'a',  1.1,  'a',  1.1,  'a',  1.1),
        ('a',  0.0,  'a',  0.0,  'a',  0.0,  'a',  0.0,  'a',  0.0),
        ('b',  1.1,  'a',  1.1,  'a',  1.1,  'a',  1.1,  'a',  1.1),
    ]).data
    t = time.time()
    for i in range(n_slices):
        y = x[:, i % 10]
    return time.time() - t

if __name__ == '__main__':
    argp = ArgumentParser()
    argp.add_argument('-n', '--num_slices', type=int, default=10000)
    argp.add_argument('--no_pd', dest='pd', action='store_false')
    argp.add_argument('--no_np', dest='np', action='store_false')
    argp.add_argument('--no_td', dest='td', action='store_false')
    args = argp.parse_args()

    if args.pd:
        t_row = pd_time_row_access(args.num_slices)
        t_col = pd_time_col_access(args.num_slices)
        print(f'Pandas           = {t_row:.3f}s, {t_col:.3f}s')

    if args.np:
        t_row = np_time_row_access(args.num_slices)
        t_col = np_time_col_access(args.num_slices)
        print(f'Structured Numpy = {t_row:.3f}s, {t_col:.3f}s')

    if args.td:
        t_row = td_time_row_access(args.num_slices)
        t_col = td_time_col_access(args.num_slices)
        print(f'Teddy            = {t_row:.3f}s, {t_col:.3f}s')
