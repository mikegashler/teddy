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
        t1 = td.Tensor(3.14, td.MetaData(-1, ['pi']))
        helpful_assert(str(t1), 'pi:3.14')
        t2 = td.Tensor(0, td.MetaData(-1, ['gender'], [['female', 'male']]))
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

        expected = ('         date  color unit \n'
                    '[[2016/02/28,   red, 4.4]\n'
                    ' [  2016/3/3, green, 8.8]]\n')
        helpful_assert(str(t[[1, 3], :]), expected)

        expected = ('         date  color \n'
                    '[[2016/02/28,   red]\n'
                    ' [2016/03/02,  blue]\n'
                    ' [  2016/3/3, green]]\n')
        helpful_assert(str(t[1:, :2]), expected)

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

        expected = ('   num animal \n'
                    '[[1.0,   cat]\n'
                    ' [2.0,   dog]\n'
                    ' [3.0,   cat]\n'
                    ' [4.0,   cat]\n'
                    ' [5.0, mouse]\n'
                    ' [6.0,   cat]]\n')
        helpful_assert(str(td.concat([a, b], 0)), expected)

        expected = ('   num anim  num animal \n'
                    '[[1.0, cat, 4.0,   cat]\n'
                    ' [2.0, dog, 5.0, mouse]\n'
                    ' [3.0, cat, 6.0,   cat]]\n')
        helpful_assert(str(td.concat([a, b], 1)), expected)

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

    def run_all_tests(self) -> None:
        print("Testing rank 0 tensors...")
        self.test_rank0_tensors()

        print("Testing categorical representations...")
        self.test_categorical_representations()

        print("Testing slicing...")
        self.test_slicing()

        print("Testing normalizing...")
        self.test_normalizing()

        print("Testing one hot encoding...")
        self.test_one_hot()

        print("Testing expand_dims...")
        self.test_expand_dims()

        print("Testing sorting...")
        self.test_sorting()

        print("Testing Pandas conversion...")
        self.test_pandas_conversion()

        print("Testing load_arff...")
        self.test_load_arff()

        print("Testing concat...")
        self.test_concat()

        print("Testing transpose...")
        self.test_transpose()

        print("Passed all tests!")


tt = TestTeddy()
tt.run_all_tests()
