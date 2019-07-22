import numpy as np
import teddy as td

print("Let's make a single scalar (rank 0 tensor with a continuous value)...")
rank0 = td.Tensor(3.14)
print(rank0)
print("")


print("Let's make a single class label (rank 0 tensor with a categorical value)...")
rank0cat = td.Tensor(0, td.MetaData([{0: "taco"}]))
print(rank0cat)
print("")


print("Let's make a vector of scalars (rank 1 tensor with continuous values)...")
rank1 = td.Tensor(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
print(rank1)
print("")

print("Let's make a vector of class labels (rank 1 tensor with categorical values)...")
rank1cat = td.Tensor(np.array([0, 1, 2, 0, 2]), td.MetaData([{0: "alpha", 1: "beta", 2: "gamma"}]))
print(rank1cat)
print("")

print("Let's make a Rank 2 tensor full of scalar zeros...")
rank2 = td.Tensor(np.zeros([4, 7]))
print(rank2)
print("")

print("Let's make a Rank 3 tensor of zeros...")
rank3 = td.Tensor(np.zeros([2, 3, 4]))
print(rank3)
print("")

print("Let's make a Rank 4 tensor of zeros...")
rank4 = td.Tensor(np.zeros([2, 2, 3, 4]))
print(rank4)
print("")

print("Let's slice the Rank 2 tensor...")
rank2_sliced = rank2[2 : 4, 2 : 5]
print(rank2_sliced)
print("")

print("Let's slice the rank 3 tensor into a rank 2 tensor...")
rank3_sliced = rank3[1,:,1:]
print(rank3_sliced)
print("")

print("Let's make a rank 2 tensor with mixed types...")
data = np.array([
    [0, 12.5, 1, 18],
    [1, 7.0, 2, 27],
    [0, 8.5, 0, 18]])
vals =  [
            {0: "Male", 1: "Female"},          # Binary (2-category) attribute
            None,                              # Continuous attribute
            {0: "Red", 1: "Green", 2: "Blue"}, # Ternary (3-category) attribute
            None,                              # Continuous attribute
        ]
names = ["Gender", "Shoe size", "Favorite color", "Age"]
mixed1 = td.Tensor(data, td.MetaData(vals, 1, names)) # The "1" indicates that the metadata is applied to the columns (axis=1).
print(mixed1)
print("")

print("Let's slice it a few different ways...")
print("(1) " + str(mixed1[1]))
print("(2) " + str(mixed1[0, 0]))
print("(3) " + str(mixed1[2, 1]))
print("(4) " + str(mixed1[:,2]))
print("(5)\n" + str(mixed1[1:, 1:]))
print("")

print("Let's do one with labeled rows...")
data = np.array([
    [0, 3, 2, 0],
    [0, 3, 2, 1],
    [100, 85, 114, 148]])
vals =  [
            None,                                            # Continuous attribute
            {0: "Blonde", 1: "Black", 2: "Red", 3: "Brown"}, # Quadrinary (4-category) attribute
            None,                                            # Continuous attribute
        ]
names = ["Children", "Hair", "IQ"]
mixed2 = td.Tensor(data, td.MetaData(vals, 0, names)) # The "0" indicates that the metadata is applied to the rows (axis=0).
print(mixed2)
print("")

print("Let's build a dataset dynamically by adding new rows one at-a-time...")
dyn = td.Tensor(np.zeros([0, 3]), td.MetaData([{}, None, {}], 1, ['Name', 'Age', 'Occupation']))
print(dyn)
print("")

dyn.data = np.append(dyn.data, [[0, 0, 0]], 0)
dyn.insert_string((0, 0), 'Alice')
dyn.set_float((0, 1), 23.)
dyn.insert_string((0, 2), 'Computer Programmer')
print(dyn)
print("")

dyn.data = np.append(dyn.data, [[0, 0, 0]], 0)
dyn.insert_string((1, 0), 'Bob')
dyn.set_float((1, 1), 32.)
dyn.insert_string((1, 2), 'Sanitation Engineer')
print(dyn)
print("")

dyn.data = np.append(dyn.data, [[0, 0, 0]], 0)
dyn.insert_string((2, 0), 'Charlie')
dyn.set_float((2, 1), 4.5)
dyn.insert_string((2, 2), 'Student')
print(dyn)
print("")

dyn.data = np.append(dyn.data, [[0, 0, 0]], 0)
dyn.insert_string((3, 0), 'Dave')
dyn.set_float((3, 1), 55)
dyn.insert_string((3, 2), 'Sanitation Engineer')
print(dyn)
print("")

dyn.data = np.append(dyn.data, [[0, 0, 0]], 0)
dyn.insert_string((4, 0), 'Eric')
dyn.set_float((4, 1), 79)
dyn.insert_string((4, 2), 'Retired')
print(dyn)
print("")

print("Let's change Eric's job...")
dyn.set_string((4, 2), 'Sanitation Engineer')
print(dyn)
print("")

print("Let's print the internal data, to make sure the same categorical value is reused for 'Sanitation Engineer'...")
print(dyn.data)
print("")

print("Let's print the metadata, because we can...")
print(dyn.meta)

print("Let's convert a continuous attribute to a categorical one...")
convert_me = td.Tensor(np.array([[0.1,1],[0.2,0],[0.3,1],[0.4,0]]), td.MetaData([None, None], 1))
print("Before:\n" + str(convert_me))
convert_me.meta.tostr[1] = {0:'Cold', 1:'Hot'}
convert_me.meta.toval = [] # Forces Teddy to regenerate toval
print("After:\n" + str(convert_me))

print("And let's demonstrate that you can still perform numpy operations on the raw data...")
convert_me.data[:,0] *= 2
print(convert_me)

print("Let's demonstrate normalizing...")
data = np.array([
    [0., 1, 20],
    [2., 2, 30],
    [4., 2, 20],
    [6, 2, 40],
    [8, 2, 30],
    [10., 0, 21]])
vals =  [
            None,                              # Continuous attribute
            {0: "Red", 1: "Green", 2: "Blue"}, # Ternary (3-category) attribute
            None,                              # Continuous attribute
        ]
names = ["Num", "Color", "Val"]
normalize_me = td.Tensor(data, td.MetaData(vals, 1, names))
print("Before:")
print(normalize_me)
normalize_me.normalize_inplace()
print("After:")
print(normalize_me)

print("Let's demonstrate one-hot encoding...")
print("Before:")
print(mixed1)
print("After:")
print(mixed1.one_hot())

# print("Let's load a dataset from ARFF format...")
# t = td.load_arff("/home/mike/data/class/mushroom.arff")
# print(t)
