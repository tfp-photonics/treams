"""Utilities."""

import abc
import collections
import copy
import functools
import itertools
import warnings

import numpy as np

__all__ = [
    "AnnotationWarning",
    "AnnotationError",
    "AnnotationDict",
    "AnnotationSequence",
    "AnnotatedArray",
    "OrderedSet",
    "implements",
    "register_properties",
]


class OrderedSet(collections.abc.Sequence, collections.abc.Set):
    """
    Ordered set

    A abstract base class that combines a sequence and set. In contrast to regular sets
    it is expected that the equality comparison only returns `True` if all entries are
    in the same order.
    """

    @abc.abstractmethod
    def __eq__(self, other):
        raise NotImplementedError


class AnnotationWarning(UserWarning):
    """Custom warning for AnnotatedArrays."""

    pass


warnings.simplefilter("always", AnnotationWarning)


class AnnotationError(Exception):
    """Custom exception for AnnotatedArrays."""

    pass


class AnnotationDict(collections.abc.MutableMapping):
    def __init__(self, items=(), /, **kwargs):
        self._dct = {}
        for i in (items.items() if hasattr(items, "items") else items, kwargs.items()):
            for key, val in i:
                self[key] = val

    def __getitem__(self, key):
        return self._dct[key]

    def __setitem__(self, key, val):
        if key in self and self[key] != val:
            warnings.warn(f"overwriting key '{key}'", AnnotationWarning)
        self._dct[key] = val

    def __delitem__(self, key):
        del self._dct[key]

    def __iter__(self):
        return iter(self._dct)

    def __len__(self):
        return len(self._dct)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self._dct)})"

    def match(self, other):
        for key, val in self.items():
            if key in other and other[key] != val:
                warnings.warn(f"incompatible key '{key}'", AnnotationWarning)


class AnnotationSequence(collections.abc.Sequence):
    def __init__(self, *args, container=AnnotationDict):
        self._ann = tuple(container(i) for i in args)

    def __len__(self):
        return len(self._ann)

    def __getitem__(self, key):
        if isinstance(key, tuple) and key == ():
            return copy.copy(self._ann)
        if isinstance(key, slice):
            return type(self)(*self._ann[key])
        if isinstance(key, int):
            return self._ann[key]
        res = []
        for k in key:
            if not isinstance(k, int):
                raise TypeError(
                    "list index must be integer, slice, list of integers, or '()'"
                )
            res.append(self[k])
        return type(self)(*res)

    def update(self, other):
        if len(other) > len(self):
            warnings.warn(
                f"argument of length {len(self)} given: "
                f"ignore leading {len(other) - len(self)} dimensions",
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
                        f"at dimension {len(self) + i}: " + err.args[0],
                        AnnotationWarning,
                    )

    def match(self, other):
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
        if len(self) != len(other):
            return False
        for a, b in zip(self, other):
            if a != b:
                return False
        return True

    def __repr__(self):
        return f"{type(self).__name__}{repr(self._ann)}"


def _cast_nparray(arr):
    return np.asarray(arr) if isinstance(arr, AnnotatedArray) else arr


def _cast_annarray(arr):
    return arr if isinstance(arr, np.generic) else AnnotatedArray(arr)


def _parse_signature(signature, inputs):
    signature = "".join(signature.split())  # remove whitespace
    sigin, sigout = signature.split("->")  # split input and output
    sigin = sigin[1:-1].split("),(")  # split input
    sigin = [i.split(",") for i in sigin]
    sigout = sigout[1:-1].split("),(")  # split output
    sigout = [i.split(",") for i in sigout]
    for i, in_ in enumerate(inputs):
        j = 0
        while j < len(sigin[i]):
            d = sigin[i][j]
            if d.endswith("?") and len(sigin[i]) > np.ndim(in_):
                sigin = [[i for i in s if i != d] for s in sigin]
                sigout = [[i for i in s if i != d] for s in sigout]
            else:
                j += 1
    return sigin, sigout


def _parse_key(key, ndim):
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


def implements(np_func):
    "Register an __array_function__ implementation to AnnotatedArrays."

    def decorator(func):
        HANDLED_FUNCTIONS[np_func] = func
        return func

    return decorator


def _pget(key, arr):
    if arr.ndim == 0:
        return None
    val = [a.get(key) for a in arr.ann]
    if all(v == val[0] for v in val[1:]):
        return val[0]
    return tuple(val)


def _pset(key, arr, val):
    if not isinstance(val, tuple):
        val = (val,) * arr.ndim
    arr.ann.update(tuple({} if v is None else {key: v} for v in val))


def _pdel(key, arr):
    for a in arr.ann:
        a.pop(key, None)


def register_properties(obj):
    for prop in obj._properties:
        setattr(
            obj,
            prop,
            property(
                functools.partial(_pget, prop),
                functools.partial(_pset, prop),
                functools.partial(_pdel, prop),
            ),
        )
    return obj


class AnnotatedArray(np.lib.mixins.NDArrayOperatorsMixin):
    _scales = set()
    _properties = set()

    @classmethod
    def relax(cls, *args, mro=None, **kwargs):
        mro = cls.__mro__[1:] if mro is None else mro
        try:
            return cls(*args, **kwargs)
        except AnnotationError as err:
            if cls == AnnotatedArray:
                raise err from None
        cls, *mro = mro
        return cls.relax(*args, mro=mro, **kwargs)

    def __init__(self, array, ann=(), **kwargs):
        self._array = np.asarray(array)
        self.ann = getattr(array, "ann", ())
        self.ann.update(ann)
        for key, val in kwargs.items():
            if not isinstance(val, tuple):
                val = (val,) * self.ndim
            self.ann.update(tuple({} if v is None else {key: v} for v in val))

    def __str__(self):
        return str(self._array)

    def __repr__(self):
        repr_arr = "    " + repr(self._array)[6:-1].replace("\n  ", "\n")
        return f"{self.__class__.__name__}(\n{repr_arr},\n    ann={self.ann[:]}\n)"

    def __array__(self, dtype=None):
        return np.asarray(self._array, dtype=dtype)

    @property
    def ann(self):
        return self._ann

    @ann.setter
    def ann(self, ann):
        self._ann = AnnotationSequence(*(({},) * self.ndim))
        self._ann.update(ann)

    def __bool__(self):
        return bool(self._array)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Compute result first to use numpy's comprehensive checks on the arguments
        inputs_noaa = tuple(map(_cast_nparray, inputs))

        ann_out = ()
        out = kwargs.get("out")
        out = out if isinstance(out, tuple) else (out,)
        ann_out = tuple(getattr(i, "ann", None) for i in out)
        if out != (None,):
            kwargs["out"] = tuple(map(_cast_nparray, out))

        res = getattr(ufunc, method)(*inputs_noaa, **kwargs)
        istuple, res = (True, res) if isinstance(res, tuple) else (False, (res,))
        if out == (None,):
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
        sigin, sigout = _parse_signature(ufunc.signature, inputs)
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
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # if not all(issubclass(t, self.__class__) for t in types):
        #     return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __getitem__(self, key):
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
        return self.transpose()

    @implements(np.all)
    def all(self, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
        return np.logical_and.reduce(self, axis, dtype, out, keepdims, where=where)

    @implements(np.any)
    def any(self, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
        return np.logical_or.reduce(self, axis, dtype, out, keepdims, where=where)

    @implements(np.max)
    def max(self, axis=None, out=None, keepdims=False, initial=None, where=True):
        return np.maximum.reduce(self, axis, None, out, keepdims, initial, where)

    @implements(np.min)
    def min(self, axis=None, out=None, keepdims=False, initial=None, where=True):
        return np.minimum.reduce(self, axis, None, out, keepdims, initial, where)

    @implements(np.sum)
    def sum(
        self, axis=None, dtype=None, out=None, keepdims=False, initial=0, where=True
    ):
        return np.add.reduce(self, axis, dtype, out, keepdims, initial, where)

    @implements(np.prod)
    def prod(
        self, axis=None, dtype=None, out=None, keepdims=False, initial=1, where=True
    ):
        return np.multiply.reduce(self, axis, dtype, out, keepdims, initial, where)

    @implements(np.cumsum)
    def cumsum(self, axis=None, dtype=None, out=None):
        if axis is None:
            return np.add.accumulate(self.flatten(), 0, dtype, out)
        return np.add.accumulate(self, axis, dtype, out)

    @implements(np.cumprod)
    def cumprod(self, axis=None, dtype=None, out=None):
        if axis is None:
            return np.multiply.accumulate(self.flatten(), 0, dtype, out)
        return np.multiply.accumulate(self, axis, dtype, out)

    def flatten(self, order="C"):
        res = self.relax(self._array.flatten(order))
        if res.shape == self.shape:
            res.ann.update(self.ann)
        elif len(tuple(filter(lambda x: x != 1, self.shape))) == 1:
            res.ann[0].update(self.ann[self.shape.index(res.size)])
        return res

    @implements(np.trace)
    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        ann = tuple(a for i, a in enumerate(self.ann[:]) if i not in (axis1, axis2))
        return self.relax(self._array.trace(offset, axis1, axis2, dtype, out), ann)

    def astype(self, *args, **kwargs):
        return self.relax(self._array.astype(*args, **kwargs), self.ann)

    @property
    @implements(np.ndim)
    def ndim(self):
        return self._array.ndim

    @property
    @implements(np.shape)
    def shape(self):
        return self._array.shape

    @property
    @implements(np.size)
    def size(self):
        return self._array.size

    @property
    @implements(np.imag)
    def imag(self):
        return self.relax(self._array.imag, self.ann)

    @property
    @implements(np.real)
    def real(self):
        return self.relax(self._array.real, self.ann)

    def conjugate(self, *args, **kwargs):
        return np.conjugate(self, *args, **kwargs)

    @implements(np.diagonal)
    def diagonal(self, offset=0, axis1=0, axis2=1):
        ann = tuple(a for i, a in enumerate(self.ann[:]) if i not in (axis1, axis2))
        return self.relax(self._array.diagonal(offset, axis1, axis2), ann)

    @implements(np.transpose)
    def transpose(self, axes=None):
        axes = range(self.ndim - 1, -1, -1) if axes is None else axes
        return self.relax(self._array.transpose(axes), self.ann[axes])

    conj = conjugate


@implements(np.linalg.solve)
def solve(a, b):
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
    res = np.tril(np.asanyarray(a), k)
    return a.relax(res, getattr(a, "ann", ({},)))


@implements(np.triu)
def triu(a, k=0):
    res = np.triu(np.asanyarray(a), k)
    return a.relax(res, getattr(a, "ann", ({},)))


@implements(np.zeros_like)
def zeros_like(a, dtype=None, order='K', shape=None):
    return np.zeros_like(np.asarray(a), dtype=dtype, order=order, shape=shape)


@implements(np.ones_like)
def ones_like(a, dtype=None, order='K', shape=None):
    return np.ones_like(np.asarray(a), dtype=dtype, order=order, shape=shape)
