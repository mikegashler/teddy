# teddy
Teddy lets you add meta-data to unstructured numpy arrays.
If you work with tables of data in Python, Teddy may be for you.
It is a light-weight alternative to Pandas and structured numpy arrays.
Here is a comparison table:

|Teddy Tensor|Pandas DataFrame|Structured Numpy Arrays|
|---|---|---|
|Stores data in unstructured numpy arrays|Stores data in custom structures|Makes you think you are using numpy arrays, but really its a totally different structure that's much harder to work with|
|Stores categorical values as enumerations|Stores categorical values as strings|Stores categorical values as strings|
|Supports tensors of any rank|Only supports 2-dimensional tables|Supports hierarchically nested tables (why?)|
|Adds meta-data to any one axis|Adds meta-data to both axes|Immutable meta-data per column|
|Less than 1k lines of code|Over 200k lines of code|Don't know|
|Does only one thing|Does almost everything|Does some weird stuff|
|Fast|Slow|Very Fast|
|Not yet documented|Well-documented|Buggy. A nightmare to work with|
|Teddy bears are soft and cuddly|Pandas appear cuddly, but they are really dangerous|No mascot|

# Benchmarks
To see a performance benchmark, run this command:
```
python3 bench.py
```
On my laptop, it gives this output:
```
Pandas = 2.0256314277648926 seconds
Numpy = 0.0014584064483642578 seconds
Teddy = 0.0169677734375 seconds
```
Teddy is more than two orders of magnitude faster at a nearly identical slicing task!
Yes, structured numpy is yet another order of magnitude faster, but it's a trap!
Just try working with structured numpy arrays for a while, and I predict you will come running back.
By contrast, Teddy uses simple metadata that you can manipulate however you want.
Admittedly, there are not yet a lot of convenience methods for doing that,
but at least you won't be completely blocked by complexity as with Pandas or structured Numpy arrays.
My conclusion is that the competitor structures have just grown more complex than is really helpful, and it's time for a reboot.
Please examine bench.py to decide for yourself how fair this comparison really is.
I recommend tweaking this test to see how Teddy and Pandas compare at the operations you rely on most.


# How to use Teddy
Sorry, I haven't written much documentation yet.
But you can get a pretty quick idea of how to use Teddy by examining test.py.
It contains many examples for how to use it. Run this command:
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


Let's build a dataset dynamically by adding new rows one at-a-time...
  N A O


    Name   Age           Occupation
[[Alice, 23.0, Computer Programmer]]


    Name   Age           Occupation
[[Alice, 23.0, Computer Programmer]
 [  Bob, 32.0, Sanitation Engineer]]


      Name   Age           Occupation
[[  Alice, 23.0, Computer Programmer]
 [    Bob, 32.0, Sanitation Engineer]
 [Charlie,  4.5,             Student]]


      Name   Age           Occupation
[[  Alice, 23.0, Computer Programmer]
 [    Bob, 32.0, Sanitation Engineer]
 [Charlie,  4.5,             Student]
 [   Dave, 55.0, Sanitation Engineer]]


      Name   Age           Occupation
[[  Alice, 23.0, Computer Programmer]
 [    Bob, 32.0, Sanitation Engineer]
 [Charlie,  4.5,             Student]
 [   Dave, 55.0, Sanitation Engineer]
 [   Eric, 79.0,             Retired]]


Let's change Eric's job...
      Name   Age           Occupation
[[  Alice, 23.0, Computer Programmer]
 [    Bob, 32.0, Sanitation Engineer]
 [Charlie,  4.5,             Student]
 [   Dave, 55.0, Sanitation Engineer]
 [   Eric, 79.0, Sanitation Engineer]]


Let's print the internal data, to make sure the same categorical value is reused for 'Sanitation Engineer'...
[[ 0.  23.   0. ]
 [ 1.  32.   1. ]
 [ 2.   4.5  2. ]
 [ 3.  55.   1. ]
 [ 4.  79.   1. ]]

Let's print the metadata, because we can...
MetaData for axis 1
Name: {Alice, Bob, Charlie, Dave, Eric}
Age: Continuous
Occupation: {Computer Programmer, Sanitation Engineer, Student, Retired}

Let's convert a continuous attribute to a categorical one...
Before:
[[0.1, 1.0]
 [0.2, 0.0]
 [0.3, 1.0]
 [0.4, 0.0]]

After:
[[0.1,  Hot]
 [0.2, Cold]
 [0.3,  Hot]
 [0.4, Cold]]

And let's demonstrate that you can still perform numpy operations on the raw data...
[[0.2,  Hot]
 [0.4, Cold]
 [0.6,  Hot]
 [0.8, Cold]]

Let's demonstrate normalizing...
Before:
    Num  Color   Val
[[ 0.0, Green, 20.0]
 [ 2.0,  Blue, 30.0]
 [ 4.0,  Blue, 20.0]
 [ 6.0,  Blue, 40.0]
 [ 8.0,  Blue, 30.0]
 [10.0,   Red, 21.0]]

After:
                n_Num  Color n_Val
[[               0.0, Green,  0.0]
 [               0.2,  Blue,  0.5]
 [               0.4,  Blue,  0.0]
 [0.6000000000000001,  Blue,  1.0]
 [               0.8,  Blue,  0.5]
 [               1.0,   Red, 0.05]]

Let's demonstrate one-hot encoding...
Before:
   Gender Shoe  Favori   Age
[[  Male, 12.5, Green, 18.0]
 [Female,  7.0,  Blue, 27.0]
 [  Male,  8.5,   Red, 18.0]]

After:
  Male Fema Shoe  Red_ Gree Blue   Age
[[1.0, 0.0, 12.5, 0.0, 1.0, 0.0, 18.0]
 [0.0, 1.0,  7.0, 0.0, 0.0, 1.0, 27.0]
 [1.0, 0.0,  8.5, 1.0, 0.0, 0.0, 18.0]]

```

# FAQ
1. **Pandas can do XYZ. How do I do that in Teddy?** Chances are pretty good that the functionality you want has not yet been implemented in Teddy. If you want a library that is already complete, Pandas is probably for you. Teddy is for people who like to hack on the data structures they use. But the metadata in Teddy is simple and mutable, so you can probably implement the functionality you need without too much trouble.

2. **Why don't you store string values in the data table like everyone else?** String comparison is much slower than number comparison. Also, following string refs requires more page-flipping, which destroys cache coherency. Basically, that's just not a great way to store data. Teddy deliberately avoids doing that by storing the raw data as floats, and doing string look-ups only when the user is ready to see a string value.

3. **When do you plan on writing complete documentation?** I'm not really very invested in this project. I was just annoyed at Pandas one day, so I took a weekend to code up an alternative. Please feel free to take it over if you like.

4. **How do you handle missing values?** Good question. Someone needs to figure that out.

5. **How do you handle date/time attributes?** Good question. Someone needs to figure that out.

6. **This code is so awesome! Can I send you a bunch of money?** Keep your money. Pay it forward.
