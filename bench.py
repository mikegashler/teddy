import numpy as np
import teddy as td
import pandas as pd
import time

a = time.time()

for i in range(1000):

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

    # Slice it
    q = p.iloc[1:9, 1:9]

b = time.time()

for i in range(1000):

    # Make a Teddy Tensor with mixed types
    t = td.Tensor(np.array([
        [0,1,0,1,0,1,0,1,0,1],
        [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9],
        [0,1,0,1,0,1,0,1,0,1],
        [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9],
        [0,1,0,1,0,1,0,1,0,1],
        [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9],
        [0,1,0,1,0,1,0,1,0,1],
        [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9],
        [0,1,0,1,0,1,0,1,0,1],
        [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9],
        ]),
        td.MetaData([
            {0:'a', 1:'b'}, None, {0:'a', 1:'b'}, None, {0:'a', 1:'b'},
            None, {0:'a', 1:'b'}, None, {0:'a', 1:'b'}, None],
        0, ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']))

    # Slice it
    u = t[1:9, 1:9]

c = time.time()

# Report results
print("Pandas = " + str(b - a) + " seconds")
print("Teddy = " + str(c - b) + " seconds")
