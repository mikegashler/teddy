import numpy as np
import teddy as td
import pandas as pd
import time

# Make a Pandas DataFrame with mixed types
p = pd.DataFrame({
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

# Measure the cost of slicing it
p_before = time.time()
for i in range(10000):
    q = p.iloc[i % 10]
p_after = time.time()


# Make a structured Numpy array with mixed types
n = np.array([
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
	], dtype = [('c0', 'S10'), ('c1', float), ('c2', 'S10'), ('c3', float),
				('c4', 'S10'), ('c5', float), ('c6', 'S10'), ('c7', float),
				('c8', 'S10'), ('c9', float)])

# Measure the cost of slicing it
n_before = time.time()
for i in range(10000):
    z = n[i % 10]
n_after = time.time()



# Make a Teddy Tensor with mixed types
t = td.Tensor(np.array([
    [0, 0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0, 0.0],
    [1, 1.1, 1, 1.1, 1, 1.1, 1, 1.1, 1, 1.1],
    [0, 2.2, 0, 2.2, 0, 2.2, 0, 2.2, 0, 2.2],
    [1, 3.3, 1, 3.3, 1, 3.3, 1, 3.3, 1, 3.3],
    [0, 4.4, 0, 4.4, 0, 4.4, 0, 4.4, 0, 4.4],
    [1, 5.5, 1, 5.5, 1, 5.5, 1, 5.5, 1, 5.5],
    [0, 6.6, 0, 6.6, 0, 6.6, 0, 6.6, 0, 6.6],
    [1, 7.7, 1, 7.7, 1, 7.7, 1, 7.7, 1, 7.7],
    [0, 8.8, 0, 8.8, 0, 8.8, 0, 8.8, 0, 8.8],
    [1, 9.9, 1, 9.9, 1, 9.9, 1, 9.9, 1, 9.9],
    ]),
    td.MetaData([
        {0:'a', 1:'b'}, None, {0:'a', 1:'b'}, None, {0:'a', 1:'b'},
        None, {0:'a', 1:'b'}, None, {0:'a', 1:'b'}, None],
    1, ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']))

# Measure the cost of slicing it
t_before = time.time()
for i in range(10000):
    u = t[i % 10]
t_after = time.time()

# Report results
print("Pandas = " + str(p_after - p_before) + " seconds")
print("Structured Numpy = " + str(n_after - n_before) + " seconds")
print("Teddy = " + str(t_after - t_before) + " seconds")
