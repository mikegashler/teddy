# teddy
Teddy lets you add meta-data to numpy arrays.
If you work with tables of data in Python, Teddy may be for you.
It is an alternative to Pandas.
Here is a comparison table:

|Teddy Tensor|Pandas DataFrame|
|---|---|
|Supports tensors of any rank|Only supports 2-dimensional tables|
|Stores data in numpy arrays|Stores data in its own structures|
|Attaches meta-data to any one axis|Adds meta-data to both axes|
|Simple interface|Many features|
|Newer. I wrote it.|Big community. Well-tested.|
|Not yet documented|Well-documented, yet still cumbersome to use|
|Teddy bears are soft and cuddly|Pandas appear cuddly, but they're really mean.|

# Benchmarks
To see a performance benchmark, run this command:
```
python3 bench.py
```
On my laptop, it gives this output:
```
Pandas = 5.522193670272827 seconds
Teddy = 0.018588542938232422 seconds
```
It looks like Teddy is more than two orders of magnitude faster at a nearly identical slicing task!
Please examine bench.py to decide for yourself how fair this comparison is.

# How to use Teddy
Sorry, I haven't written much documentation yet.
But you can get a pretty quick idea of how to use Teddy by examining test.py.
It contains numerous examples for how to use it. Run this command:
```
python3 test.py
```
to produce these results:

```
Let's make a single scalar (rank 0 tensor with a continuous value)...
3.14

Let's make a single class label (rank 0 tensor with a categorical value)...
taco

Let's make a vector of scalars (rank 1 tensor with continuous values)...
[1.1, 2.2, 3.3, 4.4, 5.5]

Let's make a vector of class labels (rank 1 tensor with categorical values)...
[alpha, beta, gamma, alpha, gamma]

Let's make a Rank 2 tensor full of scalar zeros...
[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]


Let's make a Rank 3 tensor of zeros...
[
[[0.0, 0.0, 0.0, 0.0]
 [0.0, 0.0, 0.0, 0.0]
 [0.0, 0.0, 0.0, 0.0]]


[[0.0, 0.0, 0.0, 0.0]
 [0.0, 0.0, 0.0, 0.0]
 [0.0, 0.0, 0.0, 0.0]]
]


Let's make a Rank 4 tensor of zeros...
[[
[[0.0, 0.0, 0.0, 0.0]
 [0.0, 0.0, 0.0, 0.0]
 [0.0, 0.0, 0.0, 0.0]]


[[0.0, 0.0, 0.0, 0.0]
 [0.0, 0.0, 0.0, 0.0]
 [0.0, 0.0, 0.0, 0.0]]
]
[
[[0.0, 0.0, 0.0, 0.0]
 [0.0, 0.0, 0.0, 0.0]
 [0.0, 0.0, 0.0, 0.0]]


[[0.0, 0.0, 0.0, 0.0]
 [0.0, 0.0, 0.0, 0.0]
 [0.0, 0.0, 0.0, 0.0]]
]]


Let's slice the Rank 2 tensor...
[[0.0, 0.0, 0.0]
 [0.0, 0.0, 0.0]]


Let's slice the rank 3 tensor into a rank 2 tensor...
[[0.0, 0.0, 0.0]
 [0.0, 0.0, 0.0]
 [0.0, 0.0, 0.0]]


Let's make a rank 2 tensor with mixed types...
   Gender Shoe  Favori   Age
[[  Male, 12.5, Green, 18.0]
 [Female,  7.0,  Blue, 27.0]
 [  Male,  8.5,   Red, 18.0]]


Let's slice it a few different ways...
(1) [Gender:Female, Shoe size:7.0, Favorite color:Blue, Age:27.0]
(2) Gender:Male
(3) Shoe size:8.5
(4) Favorite color:[Green, Blue, Red]
(5)
  Shoe Favor   Age
[[7.0, Blue, 27.0]
 [8.5,  Red, 18.0]]


Let's do one with labeled rows...
[    Children:[     0,     3,   2,     0]
         Hair:[Blonde, Brown, Red, Black]
           IQ:[   100,    85, 114,   148]]
```
