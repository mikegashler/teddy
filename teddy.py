import numpy as np
from typing import Union, Dict, Optional, Tuple, Any, List


class MetaData():

    # tostr is a list of dictionaries that map sequential integers starting with 0 to string values.
    # axis specifies which axis this metadata is bound to.
    # names is a list of strings that describe each attribute.
    def __init__(self, tostr: List[Optional[Dict[int, str]]] = [], axis: int = -1, names: Optional[List[str]] = None) -> None:
        self.tostr: List[Optional[Dict[int, str]]] = tostr
        self.toval: List[Optional[Dict[str, int]]] = []
        if axis < 0 and len(tostr) > 1: raise ValueError("Multiple string maps were provided, but no axis was specified.")
        self.axis = axis
        self.names = names


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
        if d: return d[int(val)] # Categorical
        else: return str(val) # Continuous


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


    # Returns true iff the specified attribute is continuous (as opposed to categorical)
    def is_continuous(self, attr: int) -> bool:
        if self.axis < 0: attr = 0 # if axis is -1, the metadata applies to everything
        if attr >= len(self.tostr): return True
        else: return self.tostr[attr] == None


    # Returns the number of categories in a categorical attribute
    def categories(self, attr: int) -> int:
        if self.axis < 0: attr = 0 # if axis is -1, the metadata applies to everything
        return len(tostr[attr]) # type: ignore



# A numpy array with some meta data attached to one axis
class Tensor():

    # data is a float or numpy array
    # meta is a MetaData object that describes the data
    def __init__(self, data: Any, meta: Optional[MetaData] = None) -> None:
        self.data: Any = data
        self.meta = meta or MetaData([], min(self.rank() - 1, 1))
        meta_axis_size = 1 if self.meta.axis < 0 else self.data.shape[self.meta.axis]
        if len(self.meta.tostr) > meta_axis_size:
            raise ValueError(str(len(self.meta.tostr)) + " string maps were provided for axis " + str(self.meta.axis) + ", which only has a size of " + str(meta_axis_size))

    def __getitem__(self, args) -> 'Tensor': # type: ignore
        return Tensor(self.data[args], self.meta[args])

    def rank(self) -> int:
        if isinstance(self.data, np.ndarray):
            return len(self.data.shape)
        else:
            return 0


    # Generates a string represention of the a rank-2 tail-slice.
    # coords should be a full-rank list of integers. The last 2 elements will be overwritten.
    def last_two_dims_str(self, coords: List[int]) -> str:
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
                return self.last_two_dims_str([0, 0])
            else:
                coords = [0] * len(self.data.shape)
                keepgoing = True

                # Visit all coordinates (not including the last 2)
                s = ""
                while keepgoing:

                    # Opening brackets
                    for i in range(len(coords) - 2):
                        if coords[len(coords) - 3 - i] == 0: s += '['
                        else: break
                    s += '\n'

                    # Print the rank-2 slice
                    s += self.last_two_dims_str(coords)

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
    def get_float(self, coords: Tuple[int]) -> Any:
        return self.data[coords]


    # Gets the value at the specified coordinates.
    # If the value is continuous, returns a string representation of its value.
    def get_string(self, coords: Tuple[int]) -> str:
        return self.meta.to_str(coords[self.meta.axis], self.data[coords])


    # Sets the specified continuous value at the specified coordinates
    # (Does not do any validity checking, even if the attribute is categorical.
    # If you know the right enumeration value,
    # calling this method is a bit faster than calling set_string.)
    def set_float(self, coords: Tuple[int], val: Any) -> None:
        self.data[coords] = val


    # Sets a categorical value by string.
    # If the string does not match one of the categories specified in the meta data,
    # or if the attribute is not categorical, then this will raise an error.
    def set_string(self, coords: Tuple[int], val: str) -> None:
        d = self.meta.to_val(coords[self.meta.axis], val)
        self.set_float(coords, d)


    # Sets a value by string, adding a new category if necessary to accomodate it.
    # Raises an error if the specified attribute is not categorical.
    def insert_string(self, coords: Tuple[int], val: str) -> None:
        self.meta.ensure_toval()
        if self.meta.axis < 0: i = 0
        else: i = coords[self.meta.axis]
        d = self.meta.toval[i]
        if d:
            if val in d:
                self.set_float(coords, d[val])
            else:
                n = len(d)
                d[val] = len(d)
                self.set_float(coords, n)
        else:
            raise ValueError("Attempted to add a string to a continuous attribute")
