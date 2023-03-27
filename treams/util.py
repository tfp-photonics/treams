"""Utilities.

Collection of functions to make :class:`AnnotatedArray` work. The implementation is
aimed to be quite general, while still providing a solid basis for the rest of the code.
"""

import abc
import collections
import copy
import functools
import itertools
import warnings

import numpy as np

__all__ = [
    "AnnotatedArray",
    "AnnotationDict",
    "AnnotationError",
    "AnnotationSequence",
    "AnnotationWarning",
    "implements",
    "OrderedSet",
    "register_properties",
]


class OrderedSet(collections.abc.Sequence, collections.abc.Set):
    """Ordered set.

    A abstract base class that combines a sequence and set. In contrast to regular sets
    it is expected that the equality comparison only returns `True` if all entries are
    in the same order.
    """

    @abc.abstractmethod
    def __eq__(self, other):
        """Equality test."""
        raise NotImplementedError


class AnnotationWarning(UserWarning):
    """Custom warning for Annotations.

    By default the warning filter is set to 'always'.
    """

    pass


warnings.simplefilter("always", AnnotationWarning)


class AnnotationError(Exception):
    """Custom exception for Annotations."""

    pass


class AnnotationDict(collections.abc.MutableMapping):
    """Dictionary that notifies the user when overwriting keys.

    Behaves mostly similar to regular dictionaries, except that when overwriting
    existing keys a :class:`AnnotationWarning` is emmitted.

    Examples:
        An :class:`AnnotationDict` can be created from other mappings, from a list of
        key-value pairs (note how a warning is emmitted for the duplicate key), and from
        keyword arguments.

        .. code-block:: python

            >>> AnnotationDict({"a": 1, "b": 2})
            AnnotationDict({'a': 1, 'b': 2})
            >>> AnnotationDict([("a", 1), ("b", 2), ("a", 3)])
            treams/util.py:74: AnnotationWarning: overwriting key 'a'
            warnings.warn(f"overwriting key '{key}'", AnnotationWarning)
            AnnotationDict({'a': 3, 'b': 2})
            >>> AnnotationDict({"a": 1, "b": 2}, c=3)
            AnnotationDict({'a': 1, 'b': 2, 'c': 3})

    Warns:
        AnnotationWarning
    """

    def __init__(self, items=(), /, **kwargs):
        """Initialization."""
        self._dct = {}
        for i in (items.items() if hasattr(items, "items") else items, kwargs.items()):
            for key, val in i:
                self[key] = val

    def __getitem__(self, key):
        """Get a value by its key.

        Args:
            key (hashable): Key
        """
        return self._dct[key]

    def __setitem__(self, key, val):
        """Set item specified by key to the defined value.

        When overwriting an existing key an :class:`AnnotationWarning` is emitted.
        Avoid the warning by explicitly deleting the key first.

        Args:
            key (hashable): Key
            val : Value

        Warns:
            AnnotationWarning
        """
        if key in self and self[key] != val:
            warnings.warn(f"overwriting key '{key}'", AnnotationWarning)
        self._dct[key] = val

    def __delitem__(self, key):
        """Delete the key."""
        del self._dct[key]

    def __iter__(self):
        """Iterate over the keys.

        Returns:
            Iterator
        """
        return iter(self._dct)

    def __len__(self):
        """Number of keys contained.

        Returns:
            int
        """
        return len(self._dct)

    def __repr__(self):
        """String representation.

        Returns:
            str
        """
        return f"{self.__class__.__name__}({repr(self._dct)})"

    def match(self, other):
        """Compare the own keys to another dictionary.

        This emits an :class:`AnnotationWarning` for each key that would be overwritten
        by the given dictionary.

        Args:
            other (Mapping)

        Returns:
            None

        Warns:
            AnnotationWarning
        """
        for key, val in self.items():
            if key in other and other[key] != val:
                warnings.warn(f"incompatible key '{key}'", AnnotationWarning)


class AnnotationSequence(collections.abc.Sequence):
    """A Sequence of dictionaries.

    This class is intended to work together with :class:`AnnotationDict`. It provides
    convenience functions to interact with multiple of those dictionaries, which are
    mainly used to keep track of the annotations made to each dimension of an
    :class:`AnnotatedArray`. While the sequence itself is immutable the entries of each
    dictionary is mutable.

    Args:
        *args: Items of the sequence
        mapping (AnnotationDict): Type of mapping to use in the sequence.

    Warns:
        AnnotationWarning
    """

    def __init__(self, *args, mapping=AnnotationDict):
        """Initialization."""
        self._ann = tuple(mapping(i) for i in args)

    def __len__(self):
        """Number of dictionaries in the sequence.

        Returns:
            int
        """
        return len(self._ann)

    def __getitem__(self, key):
        """Get an item or subsequence.

        Indexing works with integers and slices like regular tuples. Additionally, it is
        possible to get a copy of the object with `()`, or a new sequence of mappings in
        a list (or other iterable) of integers.

        Args:
            key (iterable, slice, int)

        Returns:
            mapping
        """
        if isinstance(key, tuple) and key == ():
            return copy.copy(self._ann)
        if isinstance(key, slice):
            return type(self)(*self._ann[key])
        if isinstance(key, (int, np.integer)):
            return self._ann[key]
        res = []
        for k in key:
            if not isinstance(k, int):
                raise TypeError(
                    "sequence index must be integer, slice, list of integers, or '()'"
                )
            res.append(self[k])
        return type(self)(*res)

    def update(self, other):
        """Update all mappings in the sequence at once.

        The given sequence is aliged at the last entry and then updated pairwise.
        Warnings for overwritten keys are extended by the information at which index
        they occurred.

        Args:
            other (Sequence[Mapping]): Mappings to update with.

        Warns:
            AnnotationWarning
        """
        if len(other) > len(self):
            warnings.warn(
                f"argument of length {len(self)} given: "
                f"ignore leading {len(other) - len(self)} entries",
                AnnotationWarning,
            )
        for i in range(-1, -1 - min(len(self), len(other)), -1):
            with warnings.catch_warnings():
                warnings.simplefilter("error", category=AnnotationWarning)
                try:
                    self[i].update(other[i])
                except AnnotationWarning as err:
                    warnings.simplefilter("always", category=AnnotationWarning)
                    warnings.warn(
                        f"at index {len(self) + i}: " + err.args[0],
                        AnnotationWarning,
                    )

    def match(self, other):
        """Match all mappings at once.

        The given sequence is aliged at the last entry and then updated pairwise.
        Warnings for overwritten keys are extended by the information at which index
        they occurred, see also :func:`AnnotationDict.match`.

        Args:
            other (Sequence[Mappings]): Mappings to match.

        Warns:
            AnnotationWarning
        """
        for i in range(-1, -1 - min(len(self), len(other)), -1):
            with warnings.catch_warnings():
                warnings.simplefilter("error", category=AnnotationWarning)
                try:
                    self[i].match(other[i])
                except AnnotationWarning as err:
                    warnings.simplefilter("always", category=AnnotationWarning)
                    warnings.warn(
                        f"at dimension {len(self) + i}: " + err.args[0],
                        AnnotationWarning,
                    )

    def __eq__(self, other):
        """Equality test.

        Two sequences of mappings are considered equal when they have equal length and
        when they all mappings are equal.

        Args:
            other (Sequence[Mapping]): Mappings to compare with.

        Returns:
            bool
        """
        try:
            lenother = len(other)
        except TypeError:
            return False
        if len(self) != lenother:
            return False
        try:
            zipped = zip(self, other)
        except TypeError:
            return False
        for a, b in zipped:
            if a != b:
                return False
        return True

    def __repr__(self):
        """String representation.

        Returns:
            str
        """
        return f"{type(self).__name__}{repr(self._ann)}"


def _cast_nparray(arr):
    """Cast AnnotatedArray to numpy array."""
    return np.asarray(arr) if isinstance(arr, AnnotatedArray) else arr


def _cast_annarray(arr):
    """Cast array to AnnotatedArray."""
    return arr if isinstance(arr, np.generic) else AnnotatedArray(arr)


def _parse_signature(signature, inputdims):
    """Parse a ufunc signature based on the actual inputs.

    The signature is matched with the input dimensions, to get the actual signature. It
    is returned as two lists (inputs and outputs). Each list contains for each
    individual input and output another list of the signature items.

    Example:
        >>> treams.util._parse_signature('(n?,k),(k,m?)->(n?,m?)', (2, 1))
        ([['n?', 'k'], ['k']], [['n?']])

    Args:
        signature (str): Function signature
        inputdims (Iterable[int]): Input dimensions

    Returns:
        tuple[list[list[str]]
    """
    signature = "".join(signature.split())  # remove whitespace
    sigin, sigout = signature.split("->")  # split input and output
    sigin = sigin[1:-1].split("),(")  # split input
    sigin = [i.split(",") for i in sigin]
    sigout = sigout[1:-1].split("),(")  # split output
    sigout = [i.split(",") for i in sigout]
    for i, idim in enumerate(inputdims):
        j = 0
        while j < len(sigin[i]):
            d = sigin[i][j]
            if d.endswith("?") and len(sigin[i]) > idim:
                sigin = [[i for i in s if i != d] for s in sigin]
                sigout = [[i for i in s if i != d] for s in sigout]
            else:
                j += 1
    return sigin, sigout


def _parse_key(key, ndim):
    """Parse a key to index an array.

    This function attempts to replicate the numpy indexing of arrays. It can handle
    integers, slices, an Ellipsis, arrays of integers, arrays of bools, (bare) bools,
    and None. It returns the key with an Ellipsis appended if the number of index
    dimensions does not match the number of dimensions given. Additionally, it informs
    about the number of dimension indexed by fancy indexing, if the fancy indexed
    dimensions will be prepended, and the number of dimensions that the Ellipsis
    contains.

    Args:
        key (tuple): The indexing key
        ndim (int): Number of array dimensions

    Returns:
        tuple
    """
    consumed = 0
    ellipsis = False
    fancy_ndim = 0
    # nfancy = 0
    consecutive_intfancy = 0
    # consecutive_intfancy = 0: no fancy/integer indexing
    # consecutive_intfancy = 1: ongoing consecutive fancy/integer indexing
    # consecutive_intfancy = 2: terminated consecutive fancy/integer indexing
    # consecutive_intfancy >= 2: non-consecutive fancy/integer indexing

    # The first pass gets the consumed dimensions, the presence of an ellipsis, and
    # fancy index properties.
    for k in key:
        if k is not True and k is not False and isinstance(k, (int, np.integer)):
            consumed += 1
            consecutive_intfancy += (consecutive_intfancy + 1) % 2
        elif k is None:
            consecutive_intfancy += consecutive_intfancy % 2
        elif isinstance(k, slice):
            consumed += 1
            consecutive_intfancy += consecutive_intfancy % 2
        elif k is Ellipsis:
            ellipsis = True
            # consumed is determined at the end
            consecutive_intfancy += consecutive_intfancy % 2
        else:
            arr = np.asanyarray(k)
            if arr.dtype == bool:
                consumed += arr.ndim
                fancy_ndim = max(1, fancy_ndim)
            else:
                consumed += 1
                fancy_ndim = max(arr.ndim, fancy_ndim)
            # nfancy += arr.ndim
            consecutive_intfancy += (consecutive_intfancy + 1) % 2

    lenellipsis = ndim - consumed
    if lenellipsis != 0 and not ellipsis:
        key = key + (Ellipsis,)

    return key, fancy_ndim, consecutive_intfancy >= 2, lenellipsis


HANDLED_FUNCTIONS = {}
"""Dictionary of numpy functions implemented for AnnotatedArrays."""


def implements(np_func):
    """Decorator to register an __array_function__ implementation to AnnotatedArrays."""

    def decorator(func):
        HANDLED_FUNCTIONS[np_func] = func
        return func

    return decorator


def _pget(key, arr):
    """Property getter.

    Get the key from all mappings in the annotations of an array and return it as a
    tuple. If it is the same for all items of the sequence it is directly returned. If
    the key is not present in a mapping ``None`` is taken.

    Args:
        key (str): Property name
        arr (AnnotatedArray): Array from which to get the property.
    """
    if arr.ndim == 0:
        return None
    val = [a.get(key) for a in arr.ann]
    if all(v == val[0] for v in val[1:]):
        return val[0]
    return tuple(val)


def _pset(key, arr, val):
    """Property setter.

    Set the key for mappings in the annotations of an array. If the given value is not a
    tuple it is added to all dimensions of the array. Otherwise, it is aligned at the
    last dimension, see also :class:`AnnotationSequence`. `None` values are ignored.

    Args:
        key (str): Property name
        arr (AnnotatedArray): Array from which to get the property.
        val: Value
    """
    if not isinstance(val, tuple):
        val = (val,) * arr.ndim
    arr.ann.update(tuple({} if v is None else {key: v} for v in val))


def _pdel(key, arr):
    """Property deleter.

    The key is removed from all mappings, where it is present.

    Args:
        key (str): Property name
        arr (AnnotatedArray): Array from which to get the property.
    """
    for a in arr.ann:
        a.pop(key, None)


def register_properties(obj):
    """Class decorator to add default properties to an AnnotatedArray.

    The properties are assumed to be stored in the instance attribute `_properties`.

    Args:
        obj (type): Class to add the properties.

    Returns:
        type
    """
    for prop, (doc, *_) in obj._properties.items():
        setattr(
            obj,
            prop,
            property(
                functools.partial(_pget, prop),
                functools.partial(_pset, prop),
                functools.partial(_pdel, prop),
                doc,
            ),
        )
    return obj


class AnnotatedArray(np.lib.mixins.NDArrayOperatorsMixin):
    """Array that keeps track of annotations for each dimension.

    This class acts mostly like numpy arrays, but it is enhanced by the following
    functionalities:

        * Annotations are added to each dimension
        * Annotations are compared and preserved for numpy (generalized)
          :py:class:`numpy.ufunc` (like :py:data:`numpy.add`, :py:data:`numpy.exp`,
          :py:data:`numpy.matmul` and many more "standard" mathematical functions)
        * Special ufunc methods, like :py:meth:`numpy.ufunc.reduce`, are supported
        * A growing subset of other numpy functions are supported, like
          :py:func:`numpy.linalg.solve`
        * Keywords can be specified as scale are also index into, when index the
          AnnotatedArray
        * Annotations can also be exposed as properties

    .. testsetup::

        from treams.util import AnnotatedArray

    Example:
        >>> a = AnnotatedArray([[0, 1], [2, 3]], ({"a": 1}, {"b": 2}))
        >>> b = AnnotatedArray([1, 2], ({"b": 2},))
        >>> a @ b
        AnnotatedArray(
            [2, 8],
            AnnotationSequence(AnnotationDict({'a': 1}),)
        )

    The interoperability with numpy is implemented using :ref:`basics.dispatch`
    by defining :meth:`__array__`, :meth:`__array_ufunc__`, and
    :meth:`__array_function__`.
    """

    _scales = set()
    _properties = {}

    def __init__(self, array, ann=(), /, **kwargs):
        """Initalization."""
        self._array = np.asarray(array)
        self.ann = getattr(array, "ann", ())
        self.ann.update(ann)
        for key, val in kwargs.items():
            if not isinstance(val, tuple):
                val = (val,) * self.ndim
            self.ann.update(tuple({} if v is None else {key: v} for v in val))

    @classmethod
    def relax(cls, *args, mro=None, **kwargs):
        """Try creating AnnotatedArray subclasses if possible.

        Subclasses can impose stricter conditions on the Annotations. To allow a simple
        "decaying" of those subclasses it is possible to create them with this
        classmethod. It attempts array creations along the method resolution order until
        it succeeds.

        Args:
            mro (array-like, optional): Method resolution order along which to create
                the subclass. By default it takes the order of the calling class.

        Note:
            All other arguments are the same as for the default initialization.
        """
        mro = cls.__mro__[1:] if mro is None else mro
        try:
            return cls(*args, **kwargs)
        except AnnotationError as err:
            if cls == AnnotatedArray:
                raise err from None
        cls, *mro = mro
        return cls.relax(*args, mro=mro, **kwargs)

    def __str__(self):
        """String of the array itself."""
        return str(self._array)

    def __repr__(self):
        """String representation."""
        repr_arr = "    " + repr(self._array)[6:-1].replace("\n  ", "\n")
        return f"{self.__class__.__name__}(\n{repr_arr},\n    {self.ann[:]}\n)"

    def __array__(self, dtype=None):
        """Convert to an numpy array.

        This function returns the bare array without annotations. This function does not
        necessarily make a copy of the array.

        Args:
            dtype (optional): Type of the returned array.
        """
        return np.asarray(self._array, dtype=dtype)

    @property
    def ann(self):
        """Array annotations."""
        return self._ann

    @ann.setter
    def ann(self, ann):
        """Set array annotations.

        This function copies the given sequence of dictionaries.
        """
        self._ann = AnnotationSequence(*(({},) * self.ndim))
        self._ann.update(ann)

    def __bool__(self):
        """Boolean value of the array."""
        return bool(self._array)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Implement ufunc API."""
        # Compute result first to use numpy's comprehensive checks on the arguments
        inputs_noaa = tuple(map(_cast_nparray, inputs))

        ann_out = ()
        out = kwargs.get("out")
        out = out if isinstance(out, tuple) else (out,)
        ann_out = tuple(getattr(i, "ann", None) for i in out)
        if len(out) != 1 or out[0] is not None:
            kwargs["out"] = tuple(map(_cast_nparray, out))

        res = getattr(ufunc, method)(*inputs_noaa, **kwargs)
        istuple, res = (True, res) if isinstance(res, tuple) else (False, (res,))
        if len(out) == 1 and out[0] is None:
            res = tuple(map(_cast_annarray, res))
        for a, r in zip(ann_out, res):
            if a is not None:
                r.ann.update(a)

        inputs_and_where = inputs + ((kwargs["where"],) if "where" in kwargs else ())

        if ufunc.signature is None or all(
            map(lambda x: x in " (),->", ufunc.signature)
        ):
            if method in ("__call__", "reduceat", "accumulate") or (
                method == "reduce" and kwargs.get("keepdims", False)
            ):
                self._ufunc_call(inputs_and_where, res)
            elif method == "reduce":
                self._ufunc_reduce(inputs_and_where, res, kwargs.get("axis", 0))
            elif method == "at":
                self._ufunc_at(inputs)
                return
            elif method == "outer":
                self._ufunc_outer(inputs, res, kwargs.get("where", True))
            else:
                warnings.warn("unrecognized ufunc method", AnnotationWarning)
        else:
            res = self._gufunc_call(ufunc, inputs, kwargs, res)
        res = tuple(r if isinstance(r, np.generic) else self.relax(r) for r in res)
        return res if istuple else res[0]

    @staticmethod
    def _ufunc_call(inputs_and_where, res):
        for out, in_ in itertools.product(res, inputs_and_where):
            if not isinstance(out, AnnotatedArray):
                continue
            out.ann.update(getattr(in_, "ann", ()))

    @staticmethod
    def _ufunc_reduce(inputs_and_where, res, axis=0, keepdims=False):
        out = res[0]  # reduce only allowed for single output functions
        if not isinstance(out, AnnotatedArray):
            return
        axis = axis if isinstance(axis, tuple) else (axis,)
        in_ = inputs_and_where[0]
        axis = sorted(map(lambda x: x % np.ndim(in_) - np.ndim(in_), axis))
        for in_ in inputs_and_where:
            ann = list(getattr(in_, "ann", []))
            if not keepdims:
                for a in axis:
                    try:
                        del ann[a]
                    except IndexError:
                        pass
            out.ann.update(ann)

    @staticmethod
    def _ufunc_at(inputs):
        out = inputs[0]
        if not isinstance(out, AnnotatedArray):
            return
        if any(d != {} for d in getattr(inputs[1], "ann", ())):
            warnings.warn("annotations in indices are ignored", AnnotationWarning)
        for in_ in inputs[2:]:
            ann = getattr(in_, "ann", ())
            out.ann.update(ann)

    @staticmethod
    def _ufunc_outer(inputs, res, where=True):
        for out in res:
            if not isinstance(out, AnnotatedArray):
                continue
            in_ann = tuple(
                i for a in inputs for i in getattr(a, "ann", np.ndim(a) * ({},))
            )
            out.ann.update(in_ann)
            where_ann = getattr(where, "ann", ())
            out.ann.update(where_ann)

    @staticmethod
    def _gufunc_call(ufunc, inputs, kwargs, res):
        sigin, sigout = _parse_signature(ufunc.signature, map(np.ndim, inputs))
        if kwargs.get("keepdims", False):
            sigout = [sigin[0] for _ in range(ufunc.nout)]
        ndims = [np.ndim(i) for x in (inputs, res) for i in x]
        axes = getattr(kwargs, "axes", None)
        if axes is None:
            axis = getattr(kwargs, "axis", None)
            if axis is None:
                axes = [tuple(range(-len(i), 0)) for i in sigin + sigout]
            else:
                axes = [(axis,) for _ in range(ufunc.nin)]
        else:
            axes = [
                tuple(a) if isinstance(a, collections.abc.Iterable) else (a,)
                for a in axes
            ]
        append = axes[0] if kwargs.get("keepdims", False) else ()
        axes += [append] * (ufunc.nin + ufunc.nout - len(axes))
        axes = [(*(i % ndim - ndim for i in a),) for a, ndim in zip(axes, ndims)]
        iterdims = [
            [i for i in range(-1, -1 - ndim, -1) if i not in a]
            for a, ndim in zip(axes, ndims)
        ]

        # compare core dimensions
        coredims = {}
        for i, (ax, sig) in enumerate(zip(axes, sigin + sigout)):
            for a, key in zip(ax, sig):
                if key.isnumeric():
                    continue
                coredims.setdefault(key, []).append((i, a))
        inout = tuple(inputs) + res
        for val in coredims.values():
            for (isrc, dimsrc), (idest, dimdest) in itertools.combinations(val, 2):
                source = getattr(inout[isrc], "ann", {dimsrc: {}})[dimsrc]
                dest = getattr(inout[idest], "ann", {dimdest: AnnotationDict()})[
                    dimdest
                ]
                if isrc < ufunc.nin <= idest:
                    dest.update(source)
                else:
                    dest.match(source)

        # compare iteration dimensions
        for iout, out in enumerate(res):
            if not isinstance(out, AnnotatedArray):
                continue
            for idim, dim in enumerate(iterdims[ufunc.nin + iout]):
                dest = out.ann[dim]
                for in_, iterdim in zip(inputs, iterdims):
                    if idim >= len(iterdim) or getattr(in_, "ann", None) is None:
                        continue
                    source = getattr(in_, "ann", {iterdim[idim]: {}})[iterdim[idim]]
                    dest.update(source)
        return res

    def __array_function__(self, func, types, args, kwargs):
        """Function calls on the array.

        Calls defined function in :data:`HANDLED_FUNCTIONS` otherwise raises exception.
        Add functions to it by using the decorator :func:`implements` for custom
        implementations.
        """
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # if not all(issubclass(t, self.__class__) for t in types):
        #     return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __getitem__(self, key):
        """Get an item from the AnnotatedArray.

        The indexing supports most of numpys regular and fancy indexing.
        """
        res = AnnotatedArray(self._array[key])
        if isinstance(res, np.generic) or res.ndim == 0:
            return res

        key = key if isinstance(key, tuple) else (key,)
        key, fancy_ndim, prepend_fancy, lenellipsis = _parse_key(key, self.ndim)
        source = 0
        dest = fancy_ndim if prepend_fancy else 0
        for k in key:
            if k is not True and k is not False and isinstance(k, (int, np.integer)):
                source += 1
            elif k is None:
                dest += 1
            elif isinstance(k, slice):
                for kk, val in self.ann[source].items():
                    if kk in self._scales:
                        res.ann[dest][kk] = val[k]
                    else:
                        res.ann[dest][kk] = val
                dest += 1
                source += 1
            elif k is Ellipsis:
                for _ in range(lenellipsis):
                    res.ann[dest].update(self.ann[source])
                    dest += 1
                    source += 1
            else:
                k = np.asanyarray(k)
                ksq = k.squeeze()
                if ksq.ndim == 1:
                    pos = (np.array(k.shape) == k.size).argmax()
                    ann = (
                        self.ann[source + pos] if k.dtype == bool else self.ann[source]
                    )
                    pos += int(not prepend_fancy) * dest + fancy_ndim - k.ndim
                    for kk, val in ann.items():
                        if kk in self._scales:
                            res.ann[pos][kk] = val[ksq]
                        else:
                            res.ann[pos][kk] = val
                source += k.ndim if k.dtype == bool else 1
                if not prepend_fancy:
                    dest += fancy_ndim
                    fancy_ndim = 0
        return self.relax(res)

    def __setitem__(self, key, value):
        """Set values.

        If the provided value is an AnnotatedArray the annotations of corresponding
        dimensions will be matched.
        """
        self._array[key] = value
        if not hasattr(value, "ann"):
            return
        key = key if isinstance(key, tuple) else (key,)
        key, fancy_ndim, prepend_fancy, lenellipsis = _parse_key(key, self.ndim)
        source = 0
        dest = fancy_ndim if prepend_fancy else 0
        for k in key:
            if k is not True and k is not False and isinstance(k, (int, np.integer)):
                source += 1
            elif k is None:
                dest += 1
            elif isinstance(k, slice):
                for kk, val in self.ann[source].items():
                    if kk not in value.ann[dest]:
                        continue
                    if kk in self._scales:
                        val = val[k]
                    if val != value.ann[dest][kk]:
                        warnings.warn("incompatible annotations", AnnotationWarning)
                dest += 1
                source += 1
            elif k is Ellipsis:
                for _ in range(lenellipsis):
                    self.ann[source].match(value.ann[dest])
                    dest += 1
                    source += 1
            else:
                k = np.asanyarray(k)
                ksq = k.squeeze()
                if ksq.ndim == 1:
                    pos = (np.array(k.shape) == k.size).argmax()
                    ann = (
                        self.ann[source + pos] if k.dtype == bool else self.ann[source]
                    )
                    pos += int(prepend_fancy) * dest + fancy_ndim - k.ndim
                    for kk, val in ann.items():
                        if kk not in value.ann[pos]:
                            continue
                        if kk in self._scales:
                            val = val[k]
                        if val != value.ann[pos][kk]:
                            warnings.warn("incompatible annotations", AnnotationWarning)
                source += k.ndim if k.dtype == bool else 1
                if not prepend_fancy:
                    dest += fancy_ndim
                    fancy_ndim = 0

    @property
    def T(self):
        """Transpose.

        See also :py:attr:`numpy.ndarray.T`.
        """
        return self.transpose()

    @implements(np.all)
    def all(self, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
        """Test if all elements (along an axis) are True.

        See also :py:meth:`numpy.ndarray.all`.
        """
        return np.logical_and.reduce(self, axis, dtype, out, keepdims, where=where)

    @implements(np.any)
    def any(self, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
        """Test if any element (along an axis) is True.

        See also :py:meth:`numpy.ndarray.any`.
        """
        return np.logical_or.reduce(self, axis, dtype, out, keepdims, where=where)

    @implements(np.max)
    def max(self, axis=None, out=None, keepdims=False, initial=None, where=True):
        """Maximum (along an axis).

        See also :py:meth:`numpy.ndarray.max`.
        """
        return np.maximum.reduce(self, axis, None, out, keepdims, initial, where)

    @implements(np.min)
    def min(self, axis=None, out=None, keepdims=False, initial=None, where=True):
        """Minimum (along an axis).

        See also :py:meth:`numpy.ndarray.min`.
        """
        return np.minimum.reduce(self, axis, None, out, keepdims, initial, where)

    @implements(np.sum)
    def sum(
        self, axis=None, dtype=None, out=None, keepdims=False, initial=0, where=True
    ):
        """Sum of elements (along an axis).

        See also :py:meth:`numpy.ndarray.sum`.
        """
        return np.add.reduce(self, axis, dtype, out, keepdims, initial, where)

    @implements(np.prod)
    def prod(
        self, axis=None, dtype=None, out=None, keepdims=False, initial=1, where=True
    ):
        """Product of elements (along an axis).

        See also :py:meth:`numpy.ndarray.prod`.
        """
        return np.multiply.reduce(self, axis, dtype, out, keepdims, initial, where)

    @implements(np.cumsum)
    def cumsum(self, axis=None, dtype=None, out=None):
        """Cumulative sum of elements (along an axis).

        See also :py:meth:`numpy.ndarray.cumsum`.
        """
        if axis is None:
            return np.add.accumulate(self.flatten(), 0, dtype, out)
        return np.add.accumulate(self, axis, dtype, out)

    @implements(np.cumprod)
    def cumprod(self, axis=None, dtype=None, out=None):
        """Cumulative product of elements (along an axis).

        See also :py:meth:`numpy.ndarray.cumprod`.
        """
        if axis is None:
            return np.multiply.accumulate(self.flatten(), 0, dtype, out)
        return np.multiply.accumulate(self, axis, dtype, out)

    def flatten(self, order="C"):
        """Flatten array to one dimension.

        See also :py:meth:`numpy.ndarray.flatten`.
        """
        res = self.relax(self._array.flatten(order))
        if res.shape == self.shape:
            res.ann.update(self.ann)
        elif len(tuple(filter(lambda x: x != 1, self.shape))) == 1:
            res.ann[0].update(self.ann[self.shape.index(res.size)])
        return res

    @implements(np.trace)
    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        """Trace of an array.

        See also :py:meth:`numpy.ndarray.trac`.
        """
        ann = tuple(a for i, a in enumerate(self.ann[:]) if i not in (axis1, axis2))
        return self.relax(self._array.trace(offset, axis1, axis2, dtype, out), ann)

    def astype(self, *args, **kwargs):
        """Return array as given type.

        See also :py:meth:`numpy.ndarray.astype`.
        """
        return self.relax(self._array.astype(*args, **kwargs), self.ann)

    @property
    @implements(np.ndim)
    def ndim(self):
        """Number of array dimensions.

        See also :py:attr:`numpy.ndarray.ndim`.
        """
        return self._array.ndim

    @property
    @implements(np.shape)
    def shape(self):
        """Array shape.

        See also :py:attr:`numpy.ndarray.shape`.
        """
        return self._array.shape

    @property
    @implements(np.size)
    def size(self):
        """Array size.

        See also :py:attr:`numpy.ndarray.size`.
        """
        return self._array.size

    @property
    @implements(np.imag)
    def imag(self):
        """Imaginary part of the array.

        See also :py:attr:`numpy.ndarray.imag`.
        """
        return self.relax(self._array.imag, self.ann)

    @property
    @implements(np.real)
    def real(self):
        """Real part of the array.

        See also :py:attr:`numpy.ndarray.real`.
        """
        return self.relax(self._array.real, self.ann)

    def conjugate(self, *args, **kwargs):
        """Complex conjugate elementwise.

        See also :py:meth:`numpy.ndarray.conjugate`.
        """
        return np.conjugate(self, *args, **kwargs)

    @implements(np.diagonal)
    def diagonal(self, offset=0, axis1=0, axis2=1):
        """Get the diagonal of the array.

        See also :py:meth:`numpy.ndarray.diagonal`.
        """
        ann = tuple(a for i, a in enumerate(self.ann[:]) if i not in (axis1, axis2))
        return self.relax(self._array.diagonal(offset, axis1, axis2), ann)

    @implements(np.transpose)
    def transpose(self, axes=None):
        """Transpose array dimensions.

        See also :py:meth:`numpy.ndarray.transpose`.
        """
        axes = range(self.ndim - 1, -1, -1) if axes is None else axes
        return self.relax(self._array.transpose(axes), self.ann[axes])

    conj = conjugate


@implements(np.linalg.solve)
def solve(a, b):
    """Solve linear system.

    See also :py:func:`numpy.linalg.solve`.
    """
    if issubclass(type(a), type(b)) or (
        not issubclass(type(b), type(a)) and isinstance(a, AnnotatedArray)
    ):
        restype = type(a)
    else:
        restype = type(b)
    res = restype.relax(np.linalg.solve(np.asanyarray(a), np.asanyarray(b)))
    a_ann = list(getattr(a, "ann", [{}, {}]))
    b_ann = list(getattr(b, "ann", [{}, {}]))
    if np.ndim(b) == np.ndim(a) - 1:
        map(lambda x: x[0].match(x[1]), zip(a_ann[-2::-1], b_ann[-1::-1]))
        del a_ann[-2]
        del b_ann[-1]
    else:
        map(lambda x: x[0].match(x[1]), zip(a_ann[-2::-1], b_ann[-2::-1]))
        del a_ann[-2]
        del b_ann[-2]
        a_ann += [{}]
    res.ann.update(a_ann)
    res.ann.update(b_ann)
    return res


@implements(np.linalg.lstsq)
def lstsq(a, b, rcond="warn"):
    """Solve linear system using least squares.

    See also :py:func:`numpy.linalg.lstsq`.
    """
    if issubclass(type(a), type(b)) or (
        not issubclass(type(b), type(a)) and isinstance(a, AnnotatedArray)
    ):
        restype = type(a)
    else:
        restype = type(b)
    res = list(np.linalg.lstsq(np.asanyarray(a), np.asanyarray(b), rcond))
    res[0] = restype.relax(res[0])
    a_ann = getattr(a, "ann", ({},))
    b_ann = getattr(b, "ann", ({},))
    a_ann[0].match(b_ann[0])
    res[0].ann[0].update(a_ann[-1])
    if np.ndim(b) == 2:
        res[0].ann[1].update(b_ann[-1])
    return tuple(res)


@implements(np.linalg.svd)
def svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    """Compute the singular value decomposition.

    See also :py:func:`numpy.linalg.svd`.
    """
    res = list(np.linalg.svd(np.asanyarray(a), full_matrices, compute_uv, hermitian))
    ann = getattr(a, "ann", ({},))
    if compute_uv:
        res[0] = a.relax(res[0], ann[:-1] + ({},))
        res[1] = a.realx(res[1], ann[:-2] + ({},))
        res[2] = a.relax(res[2], ann[:-2] + ({}, ann[-1]))
        return res
    return a.relax(res, tuple(ann[:-2]) + ({},))


@implements(np.diag)
def diag(a, k=0):
    """Extract diagonal from an array or create an diagonal array.

    See also :py:func:`numpy.diag`.
    """
    res = np.diag(np.asanyarray(a), k)
    ann = a.ann
    if a.ndim == 1:
        ann = (ann[0], copy.copy(ann[0]))
    elif k == 0:
        ann[0].match(ann[1])
        ann = ({**ann[0], **ann[1]},)
    else:
        ann = ()
    return a.relax(res, ann)


@implements(np.tril)
def tril(a, k=0):
    """Get lower trinagular matrix.

    See also :py:func:`numpy.tril`.
    """
    res = np.tril(np.asanyarray(a), k)
    return a.relax(res, getattr(a, "ann", ({},)))


@implements(np.triu)
def triu(a, k=0):
    """Get upper trinagular matrix.

    See also :py:func:`numpy.triu`.
    """
    res = np.triu(np.asanyarray(a), k)
    return a.relax(res, getattr(a, "ann", ({},)))


@implements(np.zeros_like)
def zeros_like(a, dtype=None, order="K", shape=None):
    """Create a numpy array like the given array containing zeros."""
    return np.zeros_like(np.asarray(a), dtype=dtype, order=order, shape=shape)


@implements(np.ones_like)
def ones_like(a, dtype=None, order="K", shape=None):
    """Create a numpy array like the given array containing ones."""
    return np.ones_like(np.asarray(a), dtype=dtype, order=order, shape=shape)
