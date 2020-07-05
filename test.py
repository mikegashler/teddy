from typing import List
import teddy as td
import numpy as np
import tempfile

# Assert that 'a' and 'b' are equal.
# If not, print some helpful output to identify where they first differ
def helpful_assert(a: str, b: str) -> None:
    if a == b:
        return

    # Uh oh, they're not equal. Let's figure out where they first differ.
    n = min(len(a), len(b))
    got_it = False
    for i in range(n):
        if a[i] != b[i]:
            start = max(0, i - 8)
            finish = min(min(len(a), len(b)), i + 8)
            print("Strings do not match here:")
            print('\n' + '_' * (i - start) + 'V' + '_' * (finish - i - 1))
            for j in range(start, finish):
                if a[j] > ' ':
                    print(a[j], end = '')
                else:
                    print('~', end = '')
            print('')
            for j in range(start, finish):
                if b[j] > ' ':
                    print(b[j], end = '')
                else:
                    print('~', end = '')
            print('\n')
            got_it = True
            break
    if not got_it:
        if len(a) < len(b):
            print('The first string is shorter')
        elif len(a) > len(b):
            print('The first string is longer')
        else:
            print('huh?')
    print('a=' + a[:500] + '\n')
    print('b=' + b[:500] + '\n')

    # Trigger a test failure
    assert False


class TestTeddy():

    def test_rank0_tensors(self) -> None:
        t1 = td.Tensor(3.14, td.MetaData(None, ['pi']))
        helpful_assert(str(t1), 'pi:3.14')
        t2 = td.Tensor(0, td.MetaData(None, ['gender'], [['female', 'male']]))
        helpful_assert(str(t2), 'gender:female')

    def test_categorical_representations(self) -> None:
        t = td.init_2d([
            (      'date', 'color', 'units'),
            ('2016/02/26',  'blue',     2.2),
            ('2016/02/28',   'red',     4.4),
            ('2016/03/02',  'blue',     1.1),
            (  '2016/3/3', 'green',     8.8),
            ])
        assert t.data[0, 1] != t.data[1, 1]
        assert t.data[0, 1] == t.data[2, 1]

    def test_slicing(self) -> None:
        t = td.init_2d([
            (      'date', 'color', 'units'),
            ('2016/02/26',  'blue',     2.2),
            ('2016/02/28',   'red',     4.4),
            ('2016/03/02',  'blue',     1.1),
            (  '2016/3/3', 'green',     8.8),
            ])

        expected = '[date:2016/02/28, color:red, units:4.4]'
        helpful_assert(str(t[1]), expected)

        expected = 'color:[blue, red, blue, green]'
        helpful_assert(str(t[:, 1]), expected)
        helpful_assert(str(t[:, 'color']), expected)

        expected = ('   color \n'
                    '[[ blue]\n'
                    ' [  red]\n'
                    ' [ blue]\n'
                    ' [green]]\n')
        helpful_assert(str(t[:, 1:2]), expected)

        expected = ('         date unit \n'
                    '[[2016/02/26, 2.2]\n'
                    ' [2016/02/28, 4.4]\n'
                    ' [2016/03/02, 1.1]\n'
                    ' [  2016/3/3, 8.8]]\n')
        helpful_assert(str(t[:, [0, 2]]), expected)

        expected = ('         date unit \n'
                    '[[2016/02/26, 2.2]\n'
                    ' [2016/02/28, 4.4]\n'
                    ' [2016/03/02, 1.1]\n'
                    ' [  2016/3/3, 8.8]]\n')
        helpful_assert(str(t[:, ['date', 'units']]), expected)

        expected = ('         date \n'
                    '[[2016/02/26]\n'
                    ' [2016/02/28]\n'
                    ' [2016/03/02]\n'
                    ' [  2016/3/3]]\n')
        helpful_assert(str(t[:, ['date']]), expected)

        expected = ('[[]\n'
                    ' []\n'
                    ' []\n'
                    ' []]\n')
        helpful_assert(str(t[:, []]), expected)

        expected = ('         date  color unit \n'
                    '[[2016/02/28,   red, 4.4]\n'
                    ' [  2016/3/3, green, 8.8]]\n')
        helpful_assert(str(t[[1, 3], :]), expected)

        expected = ('         date  color \n'
                    '[[2016/02/28,   red]\n'
                    ' [2016/03/02,  blue]\n'
                    ' [  2016/3/3, green]]\n')
        helpful_assert(str(t[1:, :2]), expected)

    def test_to_list(self) -> None:
        t = td.init_2d([
            ('animal',  'num'),
            (   'cat',      1),
            (   'dog',      2),
            (   'bat',      3),
            (   'pig',      4),
        ])
        expected = ("[['cat', 1.0], ['dog', 2.0], ['bat', 3.0], ['pig', 4.0]]")
        helpful_assert(str(t.to_list()), expected)

    def test_normalizing(self) -> None:
        t = td.init_2d([
            ('attr1', 'attr2', 'attr3'),
            (0.,       'blue',     20),
            (2.,        'red',     30),
            (4.,       'blue',     20),
            (6.,      'green',     40),
            (8.,        'red',     30),
            (10.,     'green',     25),
            ])
        t.normalize_inplace()
        assert abs(0.0 - t.data[0, 0]) < 1e-10
        assert abs(0.2 - t.data[1, 0]) < 1e-10
        assert abs(0.4 - t.data[2, 0]) < 1e-10
        assert abs(0.6 - t.data[3, 0]) < 1e-10
        assert abs(0.8 - t.data[4, 0]) < 1e-10
        assert abs(1.0 - t.data[5, 0]) < 1e-10
        helpful_assert(str(t[:,'attr2']), "attr2:[blue, red, blue, green, red, green]")
        assert abs(0.0 - t.data[0, 2]) < 1e-10
        assert abs(0.5 - t.data[1, 2]) < 1e-10
        assert abs(0.0 - t.data[2, 2]) < 1e-10
        assert abs(1.0 - t.data[3, 2]) < 1e-10
        assert abs(0.5 - t.data[4, 2]) < 1e-10
        assert abs(0.25 - t.data[5, 2]) < 1e-10

    def test_one_hot(self) -> None:
        t = td.init_2d([
            ('attr1', 'attr2', 'attr3',  'attr4',  'attr5'),
            (0.0,       'blue',      20,  'true',  'apple'),
            (0.1,        'red',      30, 'false', 'carrot'),
            (0.2,      'green',      20,  'true', 'banana'),
            (0.3,        'red',      10,  'true',  'grape'),
            ])
        expected = ('  attr blue red_ gree attr3 attr appl carr bana grap \n'
                    '[[0.0, 1.0, 0.0, 0.0, 20.0, 0.0, 1.0, 0.0, 0.0, 0.0]\n'
                    ' [0.1, 0.0, 1.0, 0.0, 30.0, 1.0, 0.0, 1.0, 0.0, 0.0]\n'
                    ' [0.2, 0.0, 0.0, 1.0, 20.0, 0.0, 0.0, 0.0, 1.0, 0.0]\n'
                    ' [0.3, 0.0, 1.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0]]\n')
        actual = str(t.one_hot())
        helpful_assert(actual, expected)

        # Test that it can also handle NaNs correctly
        t.data[0, 3] = np.nan
        t.data[0, 4] = np.nan
        t.data[2, 0] = np.nan
        t.data[2, 1] = np.nan
        expecte2 = ('  attr blue red_ gree attr3 attr appl carr bana grap \n'
                    '[[0.0, 1.0, 0.0, 0.0, 20.0, 0.5, 0.0, 0.0, 0.0, 0.0]\n'
                    ' [0.1, 0.0, 1.0, 0.0, 30.0, 1.0, 0.0, 1.0, 0.0, 0.0]\n'
                    ' [nan, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 1.0, 0.0]\n'
                    ' [0.3, 0.0, 1.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0]]\n')
        actua2 = str(t.one_hot())
        helpful_assert(actua2, expecte2)


    def test_sorting(self) -> None:
        t = td.init_2d([
            ('num', 'color', 'val'),
            (    5,  'blue',  3.14),
            (    7,   'red',  1.01),
            (    4, 'green',  88.8),
            ])

        expected = ('   num  color   val \n'
                    '[[4.0, green, 88.8]\n'
                    ' [5.0,  blue, 3.14]\n'
                    ' [7.0,   red, 1.01]]\n')
        helpful_assert(str(t.sort((-1, 0))), expected)

        expected = ('   num  color   val \n'
                    '[[5.0,  blue, 3.14]\n'
                    ' [4.0, green, 88.8]\n'
                    ' [7.0,   red, 1.01]]\n')
        helpful_assert(str(t.sort((-1, 1))), expected)

        expected = ('   num  color   val \n'
                    '[[7.0,   red, 1.01]\n'
                    ' [5.0,  blue, 3.14]\n'
                    ' [4.0, green, 88.8]]\n')
        helpful_assert(str(t.sort((-1, 2))), expected)

        expected = ('   color   val  num \n'
                    '[[ blue, 3.14, 5.0]\n'
                    ' [  red, 1.01, 7.0]\n'
                    ' [green, 88.8, 4.0]]\n')
        helpful_assert(str(t.sort((0, -1))), expected)

    def test_expand_dims(self) -> None:
        t = td.Tensor(np.zeros(4), td.MetaData(0, ['a', 'b', 'c', 'd']))

        expected = ('     a    b    c    d \n'
                    '[[0.0, 0.0, 0.0, 0.0]]\n')
        helpful_assert(str(t.expand_dims(0)), expected)

        expected = ('[           a:[0.0]\n'
                    '            b:[0.0]\n'
                    '            c:[0.0]\n'
                    '            d:[0.0]]\n')
        helpful_assert(str(t.expand_dims(1)), expected)

    def test_pandas_conversion(self) -> None:
        t = td.init_2d([
            ('num', 'color', 'val'),
            (    5,  'blue',  3.14),
            (    7,   'red',  1.01),
            (    4, 'green',  88.8),
            ])

        expected = ('   num  color   val \n'
                    '[[5.0,  blue, 3.14]\n'
                    ' [7.0,   red, 1.01]\n'
                    ' [4.0, green, 88.8]]\n')
        helpful_assert(str(t), expected)
        helpful_assert(str(td.from_pandas(t.to_pandas())), expected)

    def test_from_column_mapping(self) -> None:
        cols = { 'numbers': [1,2,3,4], 'letters': ['a', 'b', 'c', 'd'] }
        t = td.from_column_mapping(cols)
        expected = ('  numb le \n'
                    '[[1.0, a]\n'
                    ' [2.0, b]\n'
                    ' [3.0, c]\n'
                    ' [4.0, d]]\n')
        helpful_assert(str(t), expected)

    def test_load_arff(self) -> None:
        arff = ('@RELATION some ARFF relation\n'
                '@ATTRIBUTE a1 real\n'
                '@ATTRIBUTE a2 {cat, dog, mouse}\n'
                '@DATA\n'
                '3.14, cat\n'
                '0.12, mouse\n'
                '0, dog')
        with tempfile.NamedTemporaryFile() as tf:
            tf.write(bytes(arff, encoding='utf8'))
            tf.flush()
            t = td.load_arff(tf.name)
            expected = ('     a1     a2 \n'
                        '[[3.14,   cat]\n'
                        ' [0.12, mouse]\n'
                        ' [ 0.0,   dog]]\n')
            helpful_assert(str(t), expected)

    def test_align(self) -> None:
        template = td.MetaData(1, ['c0'], [['apple', 'carrot']])
        t = td.init_2d([
            ('c0',),
            ('banana',),
            ('carrot',),
            ('apple',),
            ])
        aligned = td.align([t], template)
        if not np.isnan(aligned[0].data[0, 0]):
            raise ValueError('expected nan')
        if aligned[0].data[1, 0] != 1:
            raise ValueError('expected 1, got ' + str(aligned[0].data[1, 0]))
        if aligned[0].data[2, 0] != 0:
            raise ValueError('expected 0, got ' + str(aligned[0].data[2, 0]))
        expected = ('       c0 \n'
                    '[[   NaN]\n'
                    ' [carrot]\n'
                    ' [ apple]]\n')
        helpful_assert(str(aligned[0]), expected)

    def test_concat(self) -> None:
        a = td.init_2d([
            ('num', 'animal'),
            (    1,  'cat'),
            (    2,  'dog'),
            (    3,  'cat'),
            ])
        b = td.init_2d([
            ('num', 'animal'),
            (    4,  'cat'),
            (    5,  'mouse'),
            (    6,  'cat'),
            ])

        expected = ('  animal  num \n'
                    '[[  cat, 1.0]\n'
                    ' [  dog, 2.0]\n'
                    ' [  cat, 3.0]\n'
                    ' [  cat, 4.0]\n'
                    ' [mouse, 5.0]\n'
                    ' [  cat, 6.0]]\n')
        helpful_assert(str(td.concat([a, b], 0)), expected)

        c = td.init_2d([
            ('xxx', '_nimal'),
            (    4,  'cat'),
            (    5,  'mouse'),
            (    6,  'cat'),
            ])
        expected = ('   num anim  xxx _nimal \n'
                    '[[1.0, cat, 4.0,   cat]\n'
                    ' [2.0, dog, 5.0, mouse]\n'
                    ' [3.0, cat, 6.0,   cat]]\n')
        helpful_assert(str(td.concat([a, c], 1)), expected)

    def test_concat_many(self) -> None:
        a = td.init_2d([
            ('num', 'animal'),
            (    1,  'cat'),
            (    2,  'dog'),
            (    3,  'cat'),
            ])
        b = td.init_2d([
            ('animal', 'num'),
            ('mouse',  4),
            ('mouse',  5),
            (  'cat',  6),
            ])
        c = td.init_2d([
            ('num', 'animal'),
            (    7,  'giraffe'),
            (    8,  'bear'),
            (    9,  'ant'),
            ])
        d = td.init_2d([
            ('num', 'animal'),
            (   10,  'ant'),
            (   11,  'mouse'),
            (   12,  'cat'),
            ])

        expected = ('    animal   num \n'
                    '[[    cat,  1.0]\n'
                    ' [    dog,  2.0]\n'
                    ' [    cat,  3.0]\n'
                    ' [  mouse,  4.0]\n'
                    ' [  mouse,  5.0]\n'
                    ' [    cat,  6.0]\n'
                    ' [giraffe,  7.0]\n'
                    ' [   bear,  8.0]\n'
                    ' [    ant,  9.0]\n'
                    ' [    ant, 10.0]\n'
                    ' [  mouse, 11.0]\n'
                    ' [    cat, 12.0]]\n')
        all = td.concat([a, b, c, d], 0)
        helpful_assert(str(all), expected)
        assert len(all.meta.cats[0]) == 6 # six unique animals
        assert all.data[8, 0] == 0 # 'ant'
        assert all.data[3, 0] == 5 # 'mouse'

    def test_transpose(self) -> None:
        t = td.init_2d([
            ('num', 'color', 'val'),
            (    5,  'blue',  3.14),
            (    7,   'red',  1.01),
            (    4, 'green',  88.8),
            ])
        expected = ('[         num:[ 5.0,  7.0,   4.0]\n'
                    '        color:[blue,  red, green]\n'
                    '          val:[3.14, 1.01,  88.8]]\n')
        helpful_assert(str(t.transpose()), expected)

    def test_print_nans(self) -> None:
        a = td.init_2d([
            ('num', 'color', 'val'),
            (    5,  'blue',  3.14),
            (    4, 'green',  88.8),
            (    7,   'red',  1.01),
            ])
        a.data[2, 2] = np.nan
        a.data[1, 1] = np.nan
        expected = ('   num color   val \n'
                    '[[5.0, blue, 3.14]\n'
                    ' [4.0,  NaN, 88.8]\n'
                    ' [7.0,  red,  nan]]\n')
        helpful_assert(str(a), expected)

    def test_remap_cat_vals(self) -> None:
        a = td.init_2d([
            ('num', 'color', 'val'),
            (    4, 'green',  88.8),
            ])
        b = td.init_2d([
            ('num', 'color', 'val'),
            (    5,  'blue',  3.14),
            (    4, 'green',  88.8),
            (    7,   'red',  1.01),
            ])
        c = a.deepcopy()
        c.remap_cat_vals(b.meta, False)
        if a.meta.cat_to_enum[1]['green'] != 0:
            raise ValueError('Unexpected initial value')
        if c.meta.cat_to_enum[1]['green'] != 1:
            raise ValueError('Unexpected mapped value')
        helpful_assert(str(a), str(c))

    def test_remap_cat_vals_missing(self) -> None:
        a = td.init_2d([
            ('num', 'color', 'val'),
            (    4,  'pink',  88.8),
            (    3,  'pink',  44.4),
            (    2,   'red',  22.2),
            ])
        a.data[1, 1] = np.nan
        b = td.init_2d([
            ('num', 'color', 'val'),
            (    5,  'blue',  3.14),
            (    4, 'green',  88.8),
            (    7,   'red',  1.01),
            ])
        c = a.deepcopy()
        c.remap_cat_vals(b.meta, True)
        if a.meta.cat_to_enum[1]['pink'] != 0:
            raise ValueError('Unexpected initial value')
        if c.meta.cat_to_enum[1]['green'] != 1:
            raise ValueError('Unexpected mapped value')
        expected = ('   num colo   val \n'
                    '[[4.0, NaN, 88.8]\n'
                    ' [3.0, NaN, 44.4]\n'
                    ' [2.0, red, 22.2]]\n')
        helpful_assert(str(c), expected)

    def test_mean(self) -> None:
        a = td.init_2d([
            ('a','b','c'),
            (  4,  6,  8),
            (  2,  5,  5),
            (  3,  7,  5),
            ])
        expected1 = ('mean:5.0')
        helpful_assert(str(a.mean()), expected1)
        expected2 = ('[a:3.0, b:6.0, c:6.0]')
        helpful_assert(str(a.mean(axis=0)), expected2)
        expected3 = ('mean:[6.0, 4.0, 5.0]')
        helpful_assert(str(a.mean(axis=1)), expected3)

    def test_json(self) -> None:
        a = td.init_2d([
            ('num', 'color', 'val'),
            (    4,  'pink',  88.8),
            (    3,  'pink',  44.4),
            (    2,   'red',  22.2),
            ])
        a.save_json('tmp.json')
        b = td.load_json('tmp.json')
        helpful_assert(str(a), str(b))

    def run_all_tests(self) -> None:
        print('Testing rank 0 tensors...')
        self.test_rank0_tensors()

        print('Testing categorical representations...')
        self.test_categorical_representations()

        print('Testing slicing...')
        self.test_slicing()

        print('Testing to_list...')
        self.test_to_list()

        print('Testing normalizing...')
        self.test_normalizing()

        print('Testing one hot encoding...')
        self.test_one_hot()

        print('Testing expand_dims...')
        self.test_expand_dims()

        print('Testing sorting...')
        self.test_sorting()

        print('Testing Pandas conversion...')
        self.test_pandas_conversion()

        print('Testing from_column_mapping...')
        self.test_from_column_mapping()

        print('Testing load_arff...')
        self.test_load_arff()

        print('Testing align')
        self.test_align()

        print('Testing concat...')
        self.test_concat()
        self.test_concat_many()

        print('Testing transpose...')
        self.test_transpose()

        print('Testing print nans...')
        self.test_print_nans()

        print('Testing remap_cat_vals...')
        self.test_remap_cat_vals()
        self.test_remap_cat_vals_missing()

        print('Testing mean...')
        self.test_mean()

        print('Testing json...')
        self.test_json()

        print('Passed all tests!')


tt = TestTeddy()
tt.run_all_tests()
