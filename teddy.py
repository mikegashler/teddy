# The contents of this file are licensed at your option under any and/or all of the following:
# WTFPL, CC0, Apache 2.0, MIT, BSD 3-clause, MPL 2.0, GPL2.0, GPL3.0, LGPL, CDDL1.0, and EPL1.0
# So pick your favorite license, do what you want, and have fun!

from typing import Union, Dict, Mapping, Optional, Tuple, Any, List
import numpy as np
import scipy.io.arff as arff
import copy
import json
import datetime


def sort_order(l: List[Any]) -> List[int]:
    return sorted(range(len(l)), key = l.__getitem__)


# Represents the meta data along a single axis of a Tensor
class MetaData():

    # tostr is a list of dictionaries that map sequential integers starting with 0 to string values.
    # axis specifies which axis this metadata is bound to.
    # names is a list of strings that describe each attribute.
    def __init__(self,
        tostr: List[List[str]] = [],
        axis: int = -1,
        names: List[str] = []
    ) -> None:
        if axis < 0 and len(tostr) > 1: raise ValueError("Multiple attribute specifications were provided, but no axis was specified.")
        self.tostr: List[List[str]] = tostr
        self.toval: List[Dict[str, int]] = []
        for cats in self.tostr:
            t: Dict[str, int] = {}
            for i in range(len(cats)):
                t[cats[i]] = i
            self.toval.append(t)
        self.axis = axis
        self.names = names


    # Returns a deep copy of this object
    def deepcopy(self) -> "MetaData":
        return MetaData(copy.deepcopy(self.tostr), self.axis, copy.deepcopy(self.names))


    # Ensures that the meta data is fully specified
    def complete(self, size: int) -> None:
        while len(self.tostr) < size:
            self.tostr.append([])
        while len(self.names) < size:
            self.names.append('attr_' + str(len(self.names)))


    # Returns metadata sorted with the specified order
    def sort(self, order: List[int]) -> "MetaData":
        self.complete(max(order))
        newtostr = [self.tostr[i] for i in order]
        newnames = [self.names[i] for i in order]
        return MetaData(newtostr, self.axis, newnames)


    # Returns the numerical representation of the specified string given the specified axis
    def to_val(self, attr: int, s: str) -> int:
        if self.axis < 0: attr = 0 # if axis is -1, the metadata applies to every axis
        return self.toval[attr][s]


    # Returns a string representation of the specified value
    def to_str(self, attr: int, val: float) -> str:
        if self.axis < 0: attr = 0 # if axis is -1, the metadata applies to every axis
        if attr >= len(self.tostr): return str(val) # Continuous
        d = self.tostr[attr]
        if len(d) == 0: return str(val) # Continuous
        else:
            cat_index = int(val)
            if cat_index >= 0 and cat_index < len(d):
                return d[int(val)] # Categorical
            else:
                raise ValueError(str(val) + ' is not a valid index into the list of ' + str(len(d)) + ' categories for attribute ' + str(attr))


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
                        newtostr = newtostr[a]
                        newnames = newnames[a]
                    else:
                        newtostr = [self.tostr[a]] # list of one element
                        if len(newnames) > a: newnames = [newnames[a]]
                        else: newnames = []
                        newaxis = -1
                else:
                    break
                i += 1
            return MetaData(newtostr, newaxis, newnames)
        elif isinstance(args, slice): # Slice axis 0
            if self.axis == 0: return MetaData(self.tostr[args], self.axis, self.names[args])
            else: return self # Nothing to change
        else: # Index axis 0
            if self.axis > 0: return MetaData(self.tostr, self.axis - 1, self.names) # Drop first axis
            elif self.axis == 0: return MetaData([self.tostr[args]], -1, self.names[args] if len(self.names) > args else None) # List of one element
            else: return self # Nothing to change


    def __str__(self) -> str:
        s = "MetaData(axis=" + str(self.axis) + '):\n'
        for i in range(len(self.tostr)):
            if len(self.names) > i: s += self.names[i]
            else: s += 'untitled'
            s += ': '
            d = self.tostr[i]
            if len(d) == 0: s += 'Continuous'
            elif len(d) < 16: s += str(d)
            else:
                s += '[' + d[0]
                for i in range(1, 16):
                    s += ',' + d[i]
                s += ', ... (' + str(len(d)) + ' total)]'
            s += '\n'
        return s


    # Returns true iff the specified attribute is continuous (as opposed to categorical)
    def is_continuous(self, attr: int) -> bool:
        if self.axis < 0: attr = 0 # if axis is -1, the metadata applies to everything
        if attr >= len(self.tostr): return True
        else: return len(self.tostr[attr]) == 0


    # Returns the number of categories in a categorical attribute
    def categories(self, attr: int) -> int:
        if self.axis < 0: attr = 0 # if axis is -1, the metadata applies to everything
        return len(self.tostr[attr])



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
            if self.meta.axis == dims - 2:
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
                if len(self.meta.names) > 0: s += self.meta.names[0] + ':'
                return s + self.meta.to_str(0, self.data)
            elif len(self.data.shape) == 1: # Vector
                if self.meta.axis < 0 and len(self.meta.names) > 0: s = self.meta.names[0] + ':'
                else: s = ''
                s += '['
                if self.data.shape[0] > 0:
                    if self.meta.axis == 0 and len(self.meta.names) > 0: s += self.meta.names[0] + ':'
                    s += self.meta.to_str(0, self.data[0])
                for i in range(1, self.data.shape[0]):
                    s += ", "
                    if self.meta.axis == 0 and len(self.meta.names) > i: s += self.meta.names[i] + ':'
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
            if len(self.meta.names) > 0: s += self.meta.names[0] + ':'
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
        if self.meta.is_continuous(coords[self.meta.axis]):
            v = self.get_float(coords)
            return v
        else:
            s = self.get_string(coords)
            return s


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
    def insert_string(self, coords: Tuple[int, ...], s: str) -> None:
        if self.meta.axis < 0: i = 0
        else: i = coords[self.meta.axis]
        tv = self.meta.toval[i]
        if s in tv: # If this is already an existing category...
            self.set_float(coords, tv[s])
        else:
            # Add a new category
            ts = self.meta.tostr[i]
            n = len(ts)
            ts.append(s)
            tv[s] = n
            self.set_float(coords, n)


    # Modifies both self and other such that their meta-data is aligned.
    # Raises an error if the meta-data cannot be aligned.
    def align_meta(self, other: "Tensor") -> None:

        # Check for basic compatibility
        if self.meta.axis != other.meta.axis:
            raise ValueError("These two tensors have meta data on different axes")
        if self.data.shape[self.meta.axis] != other.data.shape[other.meta.axis]:
            raise ValueError("These two tensors have different sizes on their meta axes")
        for attr in range(self.data.shape[self.meta.axis]):
            if self.meta.is_continuous(attr) != other.meta.is_continuous(attr):
                raise ValueError("These two tensors have mismatching data types (one is continuous and one is categorical) in attribute " + str(attr))

        # Align
        for attr in range(self.data.shape[self.meta.axis]):

            # Sort the categories
            if self.meta.is_continuous(attr): continue
            order_self = sort_order(self.meta.tostr[attr])
            cats_self = [self.meta.tostr[attr][i] for i in order_self]
            half_self = sort_order(order_self)
            order_other = sort_order(other.meta.tostr[attr])
            cats_other = [other.meta.tostr[attr][i] for i in order_other]
            half_other = sort_order(order_other)

            # Merge the categories
            it_self = 0
            it_other = 0
            merged: List[str] = []
            map_self: Dict[int, int] = {}
            map_other: Dict[int, int] = {}
            while it_self < len(cats_self) or it_other < len(cats_other):
                if it_other >= len(cats_other):
                    map_self[order_self[it_self]] = len(merged)
                    merged.append(cats_self[it_self])
                    it_self += 1
                elif it_self >= len(cats_self):
                    map_other[order_other[it_other]] = len(merged)
                    merged.append(cats_other[it_other])
                    it_other += 1
                elif cats_self[it_self] < cats_other[it_other]:
                    map_self[order_self[it_self]] = len(merged)
                    merged.append(cats_self[it_self])
                    it_self += 1
                elif cats_other[it_other] < cats_self[it_self]:
                    map_other[order_other[it_other]] = len(merged)
                    merged.append(cats_other[it_other])
                    it_other += 1
                else:
                    map_self[order_self[it_self]] = len(merged)
                    map_other[order_other[it_other]] = len(merged)
                    merged.append(cats_self[it_self])
                    it_self += 1
                    it_other += 1
            self.meta.tostr[attr] = merged
            other.meta.tostr[attr] = merged

            # Remap the data
            attr_slice_tuple = (slice(None),) * self.meta.axis + (attr,) + (slice(None),) * (self.rank() - self.meta.axis - 1) # type: ignore
            slice_of_self_to_remap = self.data[attr_slice_tuple]
            with np.nditer(slice_of_self_to_remap, op_flags = ['readwrite']) as it:
                for x in it:
                    x[...] = map_self[int(x)]
            slice_of_other_to_remap = other.data[attr_slice_tuple]
            with np.nditer(slice_of_other_to_remap, op_flags = ['readwrite']) as it:
                for x in it:
                    x[...] = map_other[int(x)]


    # Extends this tensor along the specified axis by adding zeros.
    # Extending on the axis with meta data is not currently supported.
    def extend_inplace(self, axis: int, amount: int) -> None:
        if amount <= 0:
            if amount == 0: return
            else: raise ValueError("Negative extensions are not allowed. Just use slicing")
        if axis == self.meta.axis: raise ValueError("Sorry, extending along the meta axis is not allowed")
        extension_shape = self.data.shape[ : axis] + (amount,) + self.data.shape[axis + 1 : ]
        extension = np.zeros(extension_shape)
        self.data = np.concatenate([self.data, extension], axis)


    # Normalizes (in place) all of the non-categorical attributes to fall in the range [0., 1.]
    def normalize_inplace(self) -> None:
        for i in range(self.data.shape[self.meta.axis]):
            if self.meta.is_continuous(i):
                if len(self.meta.names) > i:
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
                if len(self.meta.names) <= i:
                    newnames.append('attr' + str(i))
                else:
                    newnames.append(self.meta.names[i])
            else:
                newsize += self.meta.categories(i)
                for j in range(self.meta.categories(i)):
                    if len(self.meta.names) <= i:
                        newnames.append('val' + str(j) + '_attr' + str(i))
                    else:
                        newnames.append(self.meta.tostr[i][j] + '_' + self.meta.names[i])

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
        newmeta = MetaData([[]] * newsize, self.meta.axis, newnames)
        return Tensor(newdata, newmeta)


    # coords should have the same size as the shape of this tensor.
    # coords should contain exactly one element equal to -1.
    # The -1 indicates the axis to sort along.
    # Returns a tensor sorted along the axis specified with a -1 in coords.
    def sort(self, coords: Tuple[int, ...]) -> "Tensor":
        if len(coords) != len(self.data.shape):
            raise ValueError('Expected coords to contain ' + str(len(self.data.shape)) + ' elements, with one -1 to indicate the axis to sort on.')

        # Identify the values that need to be sorted from the specified coordinates
        slice_list_in: List[Any] = []
        axis = -1
        for i in range(len(coords)):
            if coords[i] == -1:
                slice_list_in.append(slice(None))
                if axis > -1: raise ValueError('Expected only one value in coords to contain -1')
                axis = i
            else:
                slice_list_in.append(coords[i])

        # Determine the sort order
        sort_me = self.data[tuple(slice_list_in)]
        attr = coords[self.meta.axis]
        if attr != -1 and not self.meta.is_continuous(attr):
            tostr = self.meta.tostr[attr]
            no_sort_me_instead = [tostr[int(x)] for x in sort_me]
            sort_order = np.array(sorted(range(len(no_sort_me_instead)), key = no_sort_me_instead.__getitem__))
        else:
            sort_order = np.argsort(sort_me, 0)

        # Sort the data
        slice_list_out: List[Any] = []
        for i in range(len(coords)):
            if coords[i] == -1:
                slice_list_out.append(sort_order)
            else:
                slice_list_out.append(slice(None))
        newdata = self.data[tuple(slice_list_out)]

        # Sort the meta data
        if axis == self.meta.axis:
            newmeta = self.meta.sort(sort_order.tolist())
        else:
            newmeta = self.meta
        return Tensor(newdata, newmeta)


    # Assumes self is a 2d tensor with metadata on axis 1.
    # Returns an tensor with the dates in sorted order with exactly one row for each day.
    # Missing categorical values will be carried over from the previous entry.
    # Missing continuous values will be interpolated.
    def fill_missing_dates(self) -> "Tensor":

        # Check assumptions
        if self.rank() != 2: raise ValueError("Expected a rank 2 tensor")
        if self.meta.axis != 1: raise ValueError("Expected metadata along axis 1")
        fmt = '%Y/%m/%d'

        # Sort by date
        date_index = self.meta.names.index('date')
        sorted = self.sort((-1, date_index))

        # Count the number of days in the specified range
        start_date = datetime.datetime.strptime(sorted.get_string((0, date_index)), fmt)
        last_date = datetime.datetime.strptime(sorted.get_string((-1, date_index)), fmt)
        delta = datetime.timedelta(days = 1)
        rows: int = 0
        dt = start_date
        while dt <= last_date:
            rows += 1
            dt += delta
        if rows == sorted.data.shape[0]:
            return sorted

        # Interpolate missing dates
        j: int = 0 # j is the source row
        res = Tensor(np.zeros((rows, sorted.data.shape[1])), sorted.meta)
        dt = start_date # dt is the date of the destination row
        for i in range(rows): # i is the destination row
            res.data[i] = sorted.data[j] # copy from source to destination
            prev_date = datetime.datetime.strptime(res.get_string((i, date_index)), fmt)
            if dt > prev_date: # Is interpolation needed here?
                next_row = min(sorted.data.shape[0] - 1, j + 1)
                next_date = datetime.datetime.strptime(sorted.get_string((next_row, date_index)), fmt)
                for k in range(res.data.shape[1]):
                    if res.meta.is_continuous(k):
                        # Interpolate the value (Note: there seems to be some rounding issues in the next line)
                        res.data[i, k] = ((dt - prev_date) * sorted.data[next_row, k] + (next_date - dt) * sorted.data[j, k]) / (next_date - prev_date)
                res.insert_string((i, date_index), dt.strftime(fmt))

            # Advance
            dt += delta # Advance the destination date
            while j + 1 < sorted.data.shape[0] and datetime.datetime.strptime(sorted.get_string((j + 1, date_index)), fmt) <= dt:
                j += 1 # Advance the source row past the destination date
        if res.data.shape[0] < self.data.shape[0]:
            raise ValueError("Made it smaller? Were there multiple values per day?")
        return res


    # Convert all dates in the "date" column from US to ISO format.
    # Assumes each date is unique.
    def us_to_iso_dates_inplace(self) -> None:

        # Check assumptions
        if self.rank() != 2: raise ValueError("Expected a rank 2 tensor")
        if self.meta.axis != 1: raise ValueError("Expected metadata along axis 1")

        # Convert the dates
        date_index = self.meta.names.index('date')
        for i in range(self.data.shape[0]):
            val = int(self.data[i, date_index])
            old_str = self.get_string((i, date_index))
            d = datetime.datetime.strptime(old_str, '%m/%d/%Y')
            new_str = d.strftime('%Y/%m/%d')
            self.meta.tostr[date_index][val] = new_str


    # Converts this tensor to a list of dicts
    def to_list_of_dict(self, drop_nans: bool = True) -> List[Dict[str, Any]]:

        # Check assumptions
        if self.rank() != 2: raise ValueError("Expected a rank 2 tensor")
        if self.meta.axis != 1: raise ValueError("Expected metadata along axis 1")

        # Make the list of dictionaries
        obs = []
        for i in range(self.data.shape[0]):
            d = {}
            for j in range(self.data.shape[1]):
                if not drop_nans or not np.isnan(self.data[i, j]):
                    d[self.meta.names[j]] = self.get((i, j))
            obs.append(d)
        return obs



# Initializes values from a list of tuples.
# Assumes the first row contains column names.
# Determines data types from whatever is in the second row.
def init_2d(data: List[Tuple[Any,...]]) -> Tensor:
    tostr: List[List[str]] = []
    names: List[str] = []
    r0 = data[0]
    r1 = data[1]
    cat: List[bool] = [] # Used to check for consistency of types
    for col in range(len(r0)):
        colname = r0[col]
        if not isinstance(colname, str):
            raise ValueError('Expected the first row to contain column names as strings')
        names.append(colname)
        tostr.append([])
        el = r1[col]
        if isinstance(el, str): cat.append(True)
        else: cat.append(False)
    t = Tensor(np.zeros((len(data) - 1, len(names))), MetaData(tostr, 1, names))
    for row in range(len(data) - 1):
        r = data[row + 1]
        if len(r) != len(names):
            raise ValueError("Row " + str(row) + " has " + str(len(r)) + " values. Expected " + str(len(names)))
        for col in range(len(r)):
            if cat[col]:
                if not isinstance(r[col], str):
                    raise ValueError('Expected a string value at (' + str(row) + ',' + str(col) + ')')
                t.insert_string((row, col), r[col])
            else:
                if isinstance(r[col], str):
                    raise ValueError('Expected a numerical value at (' + str(row) + ',' + str(col) + ')')
                t.data[row, col] = r[col]
    return t


# Loads an ARFF file. Returns a Tensor.
def load_arff(filename: str) -> Tensor:
    data, meta = arff.loadarff(filename)
    tostr: List[List[str]] = []
    for i in meta.types():
        tostr.append([])
    t = Tensor(np.zeros((data.shape[0], len(data[0]))), MetaData(tostr, 1, meta.names()))
    for row in range(t.data.shape[0]):
        for col in range(t.data.shape[1]):
            if t.meta.is_continuous(col):
                t.data[row, col] = data[row][col]
            else:
                t.insert_string((row, col), data[row][col].decode('us-ascii'))
    return t


# Loads from a JSON format that is a list of tuples that redundantly repeat meta-data for every field
def from_list_of_dict(obs: List[Mapping[str, Any]]) -> Tensor:

    # Extract metadata from the first row
    tostr: List[List[str]] = []
    names: List[str] = []
    cat: List[bool] = [] # Used to check for consistency of types
    for k in obs[0]:
        tostr.append([])
        names.append(k)
        if isinstance(obs[0][k], str): cat.append(True)
        else: cat.append(False)

    # Extract all the data
    t = Tensor(np.zeros((len(obs), len(names))), MetaData(tostr, 1, names))
    row = 0
    for ob in obs:
        col = 0
        for k in ob:
            if cat[col]: t.insert_string((row, col), ob[k])
            else: t.data[row, col] = ob[k]
            col += 1
        row += 1
    return t


# Loads from a JSON format that is a list of tuples that redundantly repeat meta-data for every field.
# Example:
#   [
#       ("date": "2018/02/18", "units": 23),
#       ("date": "2018/02/19", "units": 32)
#   ]
def load_json_list_of_dict(filename: str) -> Tensor:
    filecontents = None
    with open(filename, mode='rb') as file:
        filecontents = file.read()
    obs = json.loads(filecontents)
    return from_list_of_dict(obs)


# Concatenates multiple tensors along the specified axis
def concat(parts: List[Tensor], axis: int) -> Tensor:

    # Check assumptions
    for i in range(1, len(parts)):
        if parts[i].meta.axis != parts[0].meta.axis:
            raise ValueError("Expected all parts to have the same meta axis")
        for j in range(len(parts[0].data.shape)):
            if j != axis and parts[i].data.shape[j] != parts[0].data.shape[j]:
                raise ValueError("Expected all parts to have the same shape except along the joining axis")
        if axis != parts[0].meta.axis:
            for j in range(len(parts[0].meta.tostr)):
                if parts[0].meta.is_continuous(j):
                    if not parts[i].meta.is_continuous(j):
                        raise ValueError("Expected all parts to have matching meta data")
                else:
                    if parts[i].meta.is_continuous(j):
                        raise ValueError("Expected all parts to have matching meta data")

    # Concatenate the parts
    if axis == parts[0].meta.axis:

        # Complete the metadata, so concatenating them will stay aligned
        for i in range(len(parts) - 1):
            parts[i].meta.complete(parts[i].data.shape[parts[i].meta.axis])

        # Fuse
        newtostr: List[List[str]] = []
        newnames: List[str] = []
        for p in parts:
            newtostr = newtostr + p.meta.tostr
            newnames = newnames + p.meta.names
        newmeta = MetaData(newtostr, parts[0].meta.axis, newnames)
        newdata = np.concatenate([p.data for p in parts], axis = axis)
        return Tensor(newdata, newmeta)
    else:
        if len(parts) == 2:
            parts[0].align_meta(parts[1])
        elif len(parts) > 2:
            # Two passes will ensure that all values have been propagated to all parts
            for i in range(1, len(parts)):
                parts[0].align_meta(parts[i])
            for i in range(1, len(parts) - 1):
                parts[0].align_meta(parts[i])
        return Tensor(np.concatenate([p.data for p in parts], axis = axis), parts[0].meta)
