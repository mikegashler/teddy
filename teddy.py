# The contents of this file are licensed at your option under any or all of the following:
# WTFPL, CC0, Apache 2.0, MIT, BSD 3-clause, MPL 2.0, GPL2.0, GPL3.0, LGPL, CDDL1.0, and EPL1.0
# So pick your favorite one, do whatever you want, and have fun!

from typing import Union, Dict, Optional, Tuple, Any, List, Iterable
import numpy as np
import scipy.io.arff as arff
import copy
import json


class MetaData():

    # tostr is a list of dictionaries that map sequential integers starting with 0 to string values.
    # axis specifies which axis this metadata is bound to.
    # names is a list of strings that describe each attribute.
    def __init__(self,
        tostr: List[Optional[Dict[int, str]]] = [],
        axis: int = -1,
        names: Optional[List[str]] = None
    ) -> None:
        self.tostr: List[Optional[Dict[int, str]]] = tostr
        self.toval: List[Optional[Dict[str, int]]] = []
        if axis < 0 and len(tostr) > 1: raise ValueError("Multiple string maps were provided, but no axis was specified.")
        self.axis = axis
        self.names = names


    # Returns a deep copy of this object
    def deepcopy(self) -> "MetaData":
        return MetaData(copy.deepcopy(self.tostr), self.axis, copy.deepcopy(self.names))


    # Lazily build the string-to-int dictionary
    def ensure_toval(self) -> None:
        if self.toval == []:
            for d in self.tostr:
                if d is None:
                    self.toval.append(None)
                else:
                    t: Dict[str, int] = {}
                    for k, v in d.items():
                        t[v] = k
                    self.toval.append(t)


    # Returns the numerical representation of the specified string given the specified axis
    def to_val(self, attr: int, s: str) -> int:
        self.ensure_toval()
        if self.axis < 0: attr = 0 # if axis is -1, the metadata applies to every axis
        return self.toval[attr][s] # type: ignore


    # Returns a string representation of the specified value
    def to_str(self, attr: int, val: float) -> str:
        if self.axis < 0: attr = 0 # if axis is -1, the metadata applies to every axis
        if attr >= len(self.tostr): return str(val) # Continuous
        d = self.tostr[attr]
        if d is None: return str(val) # Continuous
        else: return d[int(val)] # Categorical


    # Implements the [] operator to support slicing
    def __getitem__(self, args) -> 'MetaData': # type: ignore
        if isinstance(args, tuple): # Tuple
            newaxis = self.axis
            newtostr = self.tostr
            newnames = self.names
            i = 0
            for a in args:
                if i < self.axis:
                    if not isinstance(a, slice):
                        newaxis -= 1
                elif i == self.axis:
                    if isinstance(a, slice):
                        newtostr = newtostr[a] # slice the tostr array
                        if newnames: newnames = newnames[a]
                        else: newnames = None # slice the names array
                    else:
                        newtostr = [self.tostr[a]] # list of one element
                        if newnames and len(newnames) > a: newnames = [newnames[a]]
                        else: newnames = None # list of one element
                        newaxis = -1
                else:
                    break
                i += 1
            return MetaData(newtostr, newaxis, newnames)
        elif isinstance(args, slice): # Slice axis 0
            if self.axis == 0: return MetaData(self.tostr[args], self.axis, self.names[args] if self.names else None)
            else: return self # Nothing to change
        else: # Index axis 0
            if self.axis > 0: return MetaData(self.tostr, self.axis - 1, self.names) # Drop first axis
            elif self.axis == 0: return MetaData([self.tostr[args]], -1, self.names[args] if self.names and len(self.names) > args else None) # List of one element
            else: return self # Nothing to change


    def __str__(self) -> str:
        s = "MetaData for axis " + str(self.axis) + '\n'
        for i in range(len(self.tostr)):
            if self.names and len(self.names) > i: s += self.names[i]
            else: s += 'untitled'
            s += ': '
            d = self.tostr[i]
            if d is None: s += 'Continuous'
            else:
                s += '{'
                for j in range(len(d)):
                    if j > 0: s += ', '
                    s += d[j]
                s += '}'
            s += '\n'
        return s


    # Returns true iff the specified attribute is continuous (as opposed to categorical)
    def is_continuous(self, attr: int) -> bool:
        if self.axis < 0: attr = 0 # if axis is -1, the metadata applies to everything
        if attr >= len(self.tostr): return True
        else: return self.tostr[attr] == None


    # Returns the number of categories in a categorical attribute
    def categories(self, attr: int) -> int:
        if self.axis < 0: attr = 0 # if axis is -1, the metadata applies to everything
        return len(self.tostr[attr]) # type: ignore



# A numpy array with some meta data attached to one axis
class Tensor():

    # data is a float or numpy array
    # meta is a MetaData object that describes the data
    def __init__(self,
        data: Any,
        meta: Optional[MetaData] = None
    ) -> None:
        self.data: Any = data
        self.meta = meta or MetaData([], min(self.rank() - 1, 1))

        # Check that the data and metadata line up
        if self.meta.axis >= 0:
            if not isinstance(self.data, np.ndarray):
                raise ValueError("Data has no axes, but caller specified to attach meta data to axis " + str(self.meta.axis))
            if self.meta.axis >= len(self.data.shape):
                raise ValueError("Data has only " + str(len(self.data.shape)) + " axes, but caller specified to attach meta data to axis " + str(self.meta.axis))
        attr_count = 1 if self.meta.axis < 0 else self.data.shape[self.meta.axis]
        if len(self.meta.tostr) > attr_count:
            raise ValueError(str(len(self.meta.tostr)) + " string maps were provided for axis " + str(self.meta.axis) + ", which only has a size of " + str(attr_count))


    # Returns a deep copy of this tensor
    def deepcopy(self) -> "Tensor":
        return Tensor(np.copy(self.data), self.meta.deepcopy())


    # This overloads the [] operator
    def __getitem__(self, args) -> 'Tensor': # type: ignore
        return Tensor(self.data[args], self.meta[args])


    # Returns the number of axes in this tensor
    def rank(self) -> int:
        if isinstance(self.data, np.ndarray):
            return len(self.data.shape)
        else:
            return 0


    # Generates a string represention of the matrix made of the last two axes in a tensor.
    # coords should be a full-rank list of integers for the whole tensor.
    # The last 2 elements in coords will be used as an in-place buffer.
    def matrix_str(self, coords: List[int]) -> str:
        s = ""
        dims = len(self.data.shape)

        # Measure column widths
        colwidths = [0] * self.data.shape[dims - 1]
        for coords[dims - 2] in range(self.data.shape[dims - 2]):
            for coords[dims - 1] in range(self.data.shape[dims - 1]):
                v = self.meta.to_str(coords[self.meta.axis], self.data[tuple(coords)])
                colwidths[coords[dims - 1]] = max(colwidths[coords[dims - 1]], len(v))

        # Column names
        if self.meta.axis == dims - 1 and self.meta.names:
            s += "  "
            for i in range(self.data.shape[dims - 1]):
                colname = self.meta.names[i] if len(self.meta.names) > i else ''
                if len(colname) > colwidths[i] + 1: colname = colname[:colwidths[i] + 1]
                if len(colname) < colwidths[i] + 1: colname = ' ' * (colwidths[i] + 1 - len(colname)) + colname
                s += colname + ' '
            s += '\n'

        # Make the matrix
        for coords[dims - 2] in range(self.data.shape[dims - 2]):
            if coords[dims - 2] == 0: s += "["
            else: s += " "

            # Row names
            if self.meta.axis == dims - 2 and self.meta.names:
                max_row_name_len = 12
                rowname = self.meta.names[coords[dims - 2]] if len(self.meta.names) > coords[dims - 2] else ''
                if len(rowname) > max_row_name_len: rowname = colname[:max_row_name_len]
                if len(rowname) < max_row_name_len: rowname = ' ' * (max_row_name_len - len(rowname)) + rowname
                s += rowname + ':'

            # Row data
            s += "["
            for coords[dims - 1] in range(self.data.shape[dims - 1]):
                v = self.meta.to_str(coords[self.meta.axis], self.data[tuple(coords)])
                if coords[dims - 1] > 0:
                    s += ", "
                s += " " * (colwidths[coords[dims - 1]] - len(v)) + v # Pad on left side with spaces
            if coords[dims - 2] == self.data.shape[dims - 2] - 1: s += "]]\n";
            else: s += "]\n";
        return s


    # Make a human-readable string representation of this tensor
    def __str__(self) -> str:
        if isinstance(self.data, np.ndarray):
            if len(self.data.shape) == 0: # Single element
                # Actually, I don't think this case ever occurs because numpy doesn't allow zero-rank tensors
                # Oh well, doesn't hurt to support it just in case they ever decide to fix that.
                s = ''
                if self.meta.names and len(self.meta.names) > 0: s += self.meta.names[0] + ':'
                return s + self.meta.to_str(0, self.data)
            elif len(self.data.shape) == 1: # Vector
                if self.meta.axis < 0 and self.meta.names and len(self.meta.names) > 0: s = self.meta.names[0] + ':'
                else: s = ''
                s += '['
                if self.data.shape[0] > 0:
                    if self.meta.axis == 0 and self.meta.names and len(self.meta.names) > 0: s += self.meta.names[0] + ':'
                    s += self.meta.to_str(0, self.data[0])
                for i in range(1, self.data.shape[0]):
                    s += ", "
                    if self.meta.axis == 0 and self.meta.names and len(self.meta.names) > i: s += self.meta.names[i] + ':'
                    s += self.meta.to_str(i, self.data[i])
                return s + ']'
            elif len(self.data.shape) == 2: # Matrix
                return self.matrix_str([0, 0])
            else:
                coords = [0] * len(self.data.shape)
                keepgoing = True

                # Visit all coordinates (not including the last 2)
                s = ""
                attr = 0
                while keepgoing:

                    # Opening brackets
                    bracks = 0
                    for i in range(len(coords) - 2):
                        if coords[len(coords) - 3 - i] == 0:
                            s += '['
                            bracks += 1
                        else: break

                    # # Axis label
                    # if self.meta.axis == len(self.data.shape) - 3 - bracks:
                    #     s += self.meta.tostr[attr] + ':'
                    #     attr += 1

                    # Print the rank-2 slice
                    s += '\n'
                    s += self.matrix_str(coords)

                    # Closing brackets
                    for i in range(len(coords) - 2):
                        dim = len(coords) - 3 - i
                        if coords[dim] == self.data.shape[dim] - 1: s += ']'
                        else: break
                    s += '\n'

                    # Increment the coordinates
                    for revdim in range(len(coords) - 2):
                        dim = len(coords) - 3 - revdim
                        coords[dim] += 1
                        if coords[dim] >= self.data.shape[dim]:
                            coords[dim] = 0
                            if dim == 0:
                                keepgoing = False
                        else:
                            break
                return s
        else: # Single value
            s = ''
            if self.meta.names and len(self.meta.names) > 0: s += self.meta.names[0] + ':'
            return s + self.meta.to_str(0, self.data)


    # Use the same string representation even when it is part of a collection
    __repr__ = __str__


    # Gets the value at the specified coordinates.
    # If the value is categorical, returns its enumeration as a float.
    def get_float(self, coords: Tuple[int, ...]) -> Any:
        return self.data[coords]


    # Gets the value at the specified coordinates.
    # If the value is continuous, returns a string representation of its value.
    def get_string(self, coords: Tuple[int, ...]) -> str:
        return self.meta.to_str(coords[self.meta.axis], self.data[coords])


    # Returns the element at the specified coordinates.
    def get(self, coords: Tuple[int, ...]) -> Any:
        if self.meta.is_continuous(self.meta.axis): return self.get_float(coords)
        else: return self.get_string(coords)


    # Sets the specified continuous value at the specified coordinates
    # (Does not do any validity checking, even if the attribute is categorical.
    # If you know the right enumeration value,
    # calling this method is a bit faster than calling set_string.)
    def set_float(self, coords: Tuple[int, ...], val: Any) -> None:
        self.data[coords] = val


    # Sets a categorical value by string.
    # If the string does not match one of the categories specified in the meta data,
    # or if the attribute is not categorical, then this will raise an error.
    def set_string(self, coords: Tuple[int, ...], val: str) -> None:
        d = self.meta.to_val(coords[self.meta.axis], val)
        self.set_float(coords, d)


    # Sets the specified element to the specified value.
    def set(self, coords: Tuple[int, ...], val: Union[float, str]) -> None:
        if  isinstance(val, str): self.set_string(coords, val)
        else: self.set_float(coords, val)


    # Sets a value by string, adding a new category if necessary to accomodate it.
    # Raises an error if the specified attribute is not categorical.
    def insert_string(self, coords: Tuple[int, ...], val: str) -> None:
        self.meta.ensure_toval()
        if self.meta.axis < 0: i = 0
        else: i = coords[self.meta.axis]
        tv = self.meta.toval[i]
        if tv is None:
            raise ValueError("Attempted to add a string to a continuous attribute")
        else:
            if val in tv: # If this is already an existing category...
                self.set_float(coords, tv[val])
            else:
                # Add a new category
                ts = self.meta.tostr[i]
                if ts is None: raise RuntimeError("This should not happen")
                n = len(ts)
                ts[n] = val
                self.meta.toval = [] # Force all of toval to be regeneratd. (Is this really necessary?)
                self.set_float(coords, n)


    # Normalizes (in place) all of the non-categorical attributes to fall in the range [0., 1.]
    def normalize_inplace(self) -> None:
        for i in range(self.data.shape[self.meta.axis]):
            if self.meta.is_continuous(i):
                if not self.meta.names is None:
                    self.meta.names[i] = 'n_' + self.meta.names[i]
                slice_list_in: List[Any] = [slice(None)] * len(self.data.shape)
                slice_list_in[self.meta.axis] = i
                lo = self.data[tuple(slice_list_in)].min()
                hi = self.data[tuple(slice_list_in)].max()
                self.data[tuple(slice_list_in)] -= lo
                self.data[tuple(slice_list_in)] *= (1.0 / (hi - lo))


    # Normalizes all of the non-categorical attributes to fall in the range [0., 1.]
    def normalize(self) -> "Tensor":
        c = self.deepcopy()
        c.normalize_inplace()
        return c


    # Encodes all of the categorical attributes with a one-hot encoding
    def one_hot(self) -> "Tensor":
        if self.meta.axis < 0:
            raise NotImplementedError("Sorry, I haven't thought about this case yet")

        # Build list of new attribute names, and count the new size
        newnames = []
        newsize = 0
        for i in range(self.data.shape[self.meta.axis]):
            if self.meta.is_continuous(i) or self.meta.categories(i) == 2:
                newsize += 1
                if self.meta.names is None:
                    newnames.append('attr' + str(i))
                else:
                    newnames.append(self.meta.names[i])
            else:
                newsize += self.meta.categories(i)
                for j in range(self.meta.categories(i)):
                    if self.meta.names is None:
                        newnames.append('val' + str(j) + '_attr' + str(i))
                    else:
                        newnames.append(self.meta.tostr[i][j] + '_' + self.meta.names[i]) # type: ignore

        # Allocate buffer for new data
        newshape = self.data.shape[:self.meta.axis] + (newsize,) + self.data.shape[self.meta.axis + 1:]
        newdata = np.zeros(newshape)

        # Transform the data
        o = 0
        for i in range(self.data.shape[self.meta.axis]):
            slice_list_in: List[Any] = [slice(None)] * len(self.data.shape)
            slice_list_in[self.meta.axis] = i
            if self.meta.is_continuous(i) or self.meta.categories(i) == 2:
                # Copy straight over
                slice_list_out: List[Any] = [slice(None)] * len(self.data.shape)
                slice_list_out[self.meta.axis] = o
                newdata[tuple(slice_list_out)] = self.data[tuple(slice_list_in)]
                o += 1
            else:
                # Convert to a one-hot encoding
                olddata = self.data[tuple(slice_list_in)] # isolate just the relevant attribute
                it = np.nditer(olddata, flags=['multi_index']) # iterate over all the elements
                while not it.finished:
                    categorical_value = int(it[0]) # get the categorical value
                    if categorical_value >= 0 and categorical_value < self.meta.categories(i): # if the value is valid
                        hotindex = it.multi_index[:self.meta.axis] + (o + categorical_value,) + it.multi_index[self.meta.axis + 1:]
                        newdata[hotindex] = 1.
                    it.iternext()
                o += self.meta.categories(i)

        # Make a tensor with the transformed data
        newmeta = MetaData([None] * newsize, self.meta.axis, newnames)
        return Tensor(newdata, newmeta)


# Initializes values from any 2d array-like structure.
# Assumes the first row contains column names.
# Determines data types from whatever is in the second row.
def init_2d(data: Iterable[Iterable[Any]]) -> Tensor:
    tostr: List[Optional[Dict[int, str]]] = []
    names: List[str] = []
    r0 = data[0]
    r1 = data[1]
    for col in range(len(r0)):
        colname = r0[col]
        if not isinstance(colname, str):
            raise ValueError('Expected the first row to contain column names as strings')
        names.append(colname)
        el = r1[col]
        if isinstance(el, str): tostr.append({})
        else: tostr.append(None)
    t = Tensor(np.zeros((len(data) - 1, len(names))), MetaData(tostr, 1, names))
    for row in range(len(data) - 1):
        r = data[row + 1]
        if len(r) != len(names):
            raise ValueError("Row " + str(row) + " has " + str(len(r)) + " values. Expected " + str(len(names)))
        for col in range(len(r)):
            if t.meta.is_continuous(col):
                if isinstance(r[col], str):
                    raise ValueError('Expected a numerical value at (' + str(row) + ',' + str(col) + ')')
                t.data[row, col] = r[col]
            else:
                if not isinstance(r[col], str):
                    raise ValueError('Expected a string value at (' + str(row) + ',' + str(col) + ')')
                t.insert_string((row, col), r[col])
    return t


# Loads an ARFF file. Returns a Tensor.
def load_arff(filename: str) -> Tensor:
    data, meta = arff.loadarff(filename)
    tostr: List[Optional[Dict[int, str]]] = []
    for i in meta.types():
        if i == 'nominal': tostr.append({})
        else: tostr.append(None)
    t = Tensor(np.zeros((data.shape[0], len(data[0]))), MetaData(tostr, 1, meta.names()))
    for row in range(t.data.shape[0]):
        for col in range(t.data.shape[1]):
            if t.meta.is_continuous(col):
                t.data[row, col] = data[row][col]
            else:
                t.insert_string((row, col), data[row][col].decode('us-ascii'))
    return t


# Loads from a JSON format that is a list of tuples that redundantly repeat meta-data for every field
def load_json_1(filename: str) -> Tensor:

    # Load the file
    filecontents = None
    with open(filename, mode='rb') as file:
        filecontents = file.read()
    obs = json.loads(filecontents)

    # Extract metadata from the first row
    tostr: List[Optional[Dict[int, str]]] = []
    names: List[str] = []
    for k in obs[0]:
        if isinstance(obs[0][k], str): tostr.append({})
        else: tostr.append(None)
        names.append(k)

    # Extract all the data
    t = Tensor(np.zeros((len(obs), len(names))), MetaData(tostr, 1, names))
    row = 0
    for ob in obs:
        col = 0
        for k in ob:
            if t.meta.is_continuous(col): t.data[row, col] = ob[k]
            else: t.insert_string((row, col), ob[k])
            col += 1
        row += 1
    return t
