import collections
import itertools
import warnings

import numpy as np


class AnnotatedArrayWarning(UserWarning):
    pass


def _match(a, b):
    for key, val in a.items():
        if key in b and val != b[key]:
            return False
    return True


warnings.simplefilter("always", AnnotatedArrayWarning)


class ArrayAnnotations:
    def __init__(self, ndim=1):
        self._ann_dims = [{} for _ in range(ndim)]
        self._ann_global = {}

    @property
    def ndim(self):
        return len(self._ann_dims)

    def __len__(self):
        return self.ndim

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if len(key) == 2:
                if key[0] == "global":
                    return self._ann_global[key[1]]
                elif isinstance(key[0], int):
                    return self._ann_dims[key[0]][key[1]]
            raise KeyError("invalid key")
        if key == "global":
            return self._ann_global
        if key is Ellipsis:
            return [self._ann_global] + self._ann_dims
        return self._ann_dims[key]

    def __setitem__(self, key, item):
        if isinstance(key, tuple):
            if len(key) == 2:
                if key[0] == "global":
                    self._ann_global[key[1]] = item
                    return
                elif isinstance(key[0], int):
                    self._ann_dims[key[0]][key[1]] = item
                    return
        else:
            if not isinstance(item, dict):
                raise ValueError("invalid item type")
            if key == "global":
                self._ann_global = item
                return
            elif isinstance(key, int):
                self._ann_dims[key] = item
                return
        raise KeyError("invalid key")

    def update(self, other):
        self["global"].update(other["global"])
        for i in range(-1 - self.ndim, -1 - other.ndim, -1):
            self._ann_dims.append({})
        for i in range(-1, -1 - min(self.ndim, other.ndim), -1):
            self[i].update(other[i])

    def __eq__(self, other):
        if not _match(self["global"], other["global"]):
            return False
        for i in range(-1, -1 - min(self.ndim, other.ndim), -1):
            if not _match(self[i], other[i]):
                return False
        return True


class AnnotatedArray(np.ndarray):
    def __new__(cls, arr, annotations=None):
        arr = np.asanyarray(arr)
        obj = arr.view(cls)
        if annotations is not None:
            obj._annotations = annotations
        return obj

    @property
    def annotations(self):
        return self._annotations

    @annotations.setter
    def annotations(self, ann):
        if ann is None:
            ann = ArrayAnnotations(self.ndim)
        if self.ndim != ann.ndim:
            warnings.warn(
                f"incompatible number of dimensions '{self.ndim} != {ann.ndim}'",
                AnnotatedArrayWarning,
            )
        self._annotations = ann

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.annotations = getattr(obj, "annotations", None)

    def __getitem__(self, key):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AnnotatedArrayWarning)
            res = super().__getitem__(key)

        if isinstance(res, np.generic):
            return res

        res.annotations = None
        res.annotations["global"].update(self.annotations["global"])

        if res.ndim == 0:
            return res

        key = key if isinstance(key, tuple) else (key,)
        # The indexing in numpy is incredibly complicated (see also NEP 21), mainly due
        # to the presence of two indexing modes: regular and advanced (also called
        # fancy) indexing. Regular indexing uses slices, the ellipsis, None, and
        # integers, advanced indexing is triggered by anything else `array_like`. Once
        # triggered, this also affects integers. Additionally, pure Booleans have been
        # introduced with https://github.com/numpy/numpy/pull/3798, adding even more
        # complexity.

        # The main points are:
        # 1. Boolean arrays get converted to `ndim` flat integer arrays.
        # 2. Integer arrays index one dimension to `ndim` dimensions.
        # 3. Unless they all come consecutively fancy index dimensions are prepended,
        #    otherwise they stay at their position (here integers play a role).
        # 4. Pure Bools are a special fancy index that add one dimension of size 0 or 1
        #    (depending on if one of them is False or not), but don't consume any
        #    dimension. They broadcast with other fancy index types, which can make them
        #    "vanish".

        # We only copy annotations for slices and the ellipsis. If only one fancy index
        # with ndim 1 is present, its annotations are also copied.

        consumed = 0
        ellipsis = False

        fancy_ndim = 0
        nfancy = 0
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
                nfancy += arr.ndim
                consecutive_intfancy += (consecutive_intfancy + 1) % 2

        lenellipsis = self.ndim - consumed
        if lenellipsis != 0 and not ellipsis:
            key = key + (Ellipsis,)

        source = 0
        dest = 0
        if consecutive_intfancy > 2:
            dest = fancy_ndim
        for k in key:
            if k is not True and k is not False and isinstance(k, (int, np.integer)):
                source += 1
            elif k is None:
                dest += 1
            elif isinstance(k, slice):
                res.annotations[dest] = self.annotations[source]
                dest += 1
                source += 1
            elif k is Ellipsis:
                for _ in range(lenellipsis):
                    res.annotations[dest] = self.annotations[source]
                    dest += 1
                    source += 1
            else:
                arr = np.asanyarray(k)
                if nfancy == arr.ndim == 1:
                    if consecutive_intfancy > 2:
                        res.annotations[0] = self.annotations[source]
                    else:
                        res.annotations[dest] = self.annotations[source]

                source += arr.ndim if arr.dtype == bool else 1

                if consecutive_intfancy <= 2:
                    dest += fancy_ndim
                    fancy_ndim = 0

        return res

    @staticmethod
    def _ufunc_call(inputs_and_where, res):
        for out in res:
            if isinstance(out, np.generic):
                continue
            for dim in range(-1, -1 - out.ndim, -1):
                destination = out.annotations[dim]
                for in_ in inputs_and_where:
                    if dim < -in_.ndim or getattr(in_, "annotations", None) is None:
                        continue
                    source = in_.annotations[dim]
                    if not _match(source, destination):
                        warnings.warn(
                            f"incompatible annotations at dimension '{dim}'",
                            AnnotatedArrayWarning,
                        )
                    destination.update(source)

    @staticmethod
    def _ufunc_outer(inputs, res, where=True):
        where = np.asanyarray(where)
        for out in res:
            if isinstance(out, np.generic):
                continue
            dim = 0
            for in_ in inputs:
                for dim_in in range(in_.ndim):
                    destination = out.annotations[dim]
                    dim += 1
                    if getattr(in_, "annotations", None) is None:
                        continue
                    source = in_.annotations[dim_in]
                    if not _match(source, destination):
                        warnings.warn(
                            f"incompatible annotations at dimension '{dim - 1}'",
                            AnnotatedArrayWarning,
                        )
                    destination.update(source)
            for dim in range(-1, -1 - where.ndim, -1):
                destination = out.annotations[dim]
                if dim < -where.ndim or getattr(where, "annotations", None) is None:
                    break
                source = where.annotations[dim]
                if not _match(source, destination):
                    warnings.warn(
                        f"incompatible annotations at dimension '{dim}'",
                        AnnotatedArrayWarning,
                    )
                destination.update(source)

    @staticmethod
    def _ufunc_at(inputs):
        out = inputs[0]
        if isinstance(out, np.generic):
            return
        inputs = inputs[2:]
        for dim in range(-1, -1 - out.ndim, -1):
            destination = out.annotations[dim]
            for in_ in inputs:
                if dim < -in_.ndim or getattr(in_, "annotations", None) is None:
                    continue
                source = in_.annotations[dim]
                if not _match(source, destination):
                    warnings.warn(
                        f"incompatible annotations at dimension '{dim}'",
                        AnnotatedArrayWarning,
                    )
                destination.update(source)

    @staticmethod
    def _ufunc_reduce(inputs_and_where, res, axis):
        out = res[0]  # reduce only allowed for single output functions
        if isinstance(out, np.generic):
            return
        axis = tuple(range(-1, -1 - inputs[0].ndim, -1)) if axis is None else axis
        axis = axis if isinstance(axis, collections.abc.Iterable) else (axis,)
        axis = [a % inputs_and_where[0].ndim - inputs_and_where[0].ndim for a in axis]
        dim_in = 0
        for dim in range(-1, -1 - out.ndim, -1):
            dim_in -= 2 if dim in axis else 1
            destination = out.annotations[dim]
            for in_ in inputs_and_where:
                if dim_in < -in_.ndim or getattr(in_, "annotations", None) is None:
                    continue
                source = in_.annotations[dim_in]
                if not _match(source, destination):
                    warnings.warn(
                        f"incompatible annotations at dimension '{dim_in}'",
                        AnnotatedArrayWarning,
                    )
                destination.update(source)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        # Compute result first to make use of numpys comprehensive checks on the arguments
        inputs = [np.asanyarray(i) for i in inputs]
        inputs_ndarray = [np.asarray(i) for i in inputs]

        annotations_out = None
        if "out" in kwargs:
            out = kwargs["out"]
            if isinstance(out, AnnotatedArray):
                annotations_out = [out.annotations]
                kwargs["out"] = np.asarray(out)
            elif isinstance(out, tuple):
                annotations_out = []
                for i in out:
                    if isinstance(out, AnnotatedArray):
                        annotations_out.append(i.annotations)
                    else:
                        annotations_out.append(None)
                kwargs["out"] = (*(np.asarray(i) for i in out),)

        if kwargs.get("subok", True):
            res = super().__array_ufunc__(ufunc, method, *inputs_ndarray, **kwargs)
        else:
            warnings.warn("ignored setting 'subok=False'", AnnotatedArrayWarning)
            kwargs["subok"] = True

            res = np.asarray(self).__array_ufunc__(
                ufunc, method, *inputs_ndarray, **kwargs
            )

        if isinstance(res, tuple):
            istuple = True
        else:
            istuple = False
            res = (res,)
        res = (
            *(
                r if isinstance(r, np.generic) else np.asanyarray(r).view(type(self))
                for r in res
            ),
        )

        if annotations_out is not None:
            for a, r in zip(annotations_out, res):
                if a is not None:
                    r.annotations.update(a)

        # compare "global"
        inputs_and_where = (
            inputs + [np.asanyarray(kwargs["where"])] if "where" in kwargs else inputs
        )
        for out in res:
            if isinstance(out, np.generic):
                continue
            destination = out.annotations["global"]
            for in_ in inputs_and_where:
                if getattr(in_, "annotations", None) is None:
                    continue
                source = in_.annotations["global"]
                if not _match(source, destination):
                    warnings.warn(
                        "incompatible annotations of category 'global'",
                        AnnotatedArrayWarning,
                    )
                destination.update(source)

        if (
            ufunc.signature is None
            or "".join(
                [i for i in ufunc.signature if i not in ["(", ")", ",", "-", ">"]]
            ).isspace()
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
                warinings.warn("unrecognized ufunc method", AnnotatedArrayWarning)

        else:
            sigin, sigout = _parse_signature(ufunc.signature, inputs_ndarray)

            if kwargs.get("keepdims", False):
                sigout = [sigin[0] for _ in range(ufunc.nout)]

            ndims = [i.ndim for x in (inputs, res) for i in x]

            axes = getattr(kwargs, "axes", None)
            if axes is None:
                axis = getattr(kwargs, "axis", None)
                if axis is None:
                    axes = [
                        tuple(range(-len(i), 0)) for sig in (sigin, sigout) for i in sig
                    ]
                else:
                    axes = [(axis,) for _ in range(ufunc.nin)]
            else:
                axes = [
                    tuple(a) if not isinstance(a, collections.abc.Iterable) else a
                    for a in axes
                ]
            append = axes[0] if kwargs.get("keepdims", False) else ()
            while len(axes) < ufunc.nin + ufunc.nout:
                axes.append(append)
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
                    if key in coredims:
                        coredims[key].append((i, a))
                    else:
                        coredims[key] = [(i, a)]
            inout = tuple(inputs) + res
            for key, val in coredims.items():
                for (isource, dimsource), (idest, dimdest) in itertools.combinations(
                    val, 2
                ):
                    source = inout[isource].annotations[dimsource]
                    dest = inout[idest].annotations[dimdest]
                    if not _match(source, dest):
                        warnings.warn(
                            f"incompatible annotations at core dimensions",
                            AnnotatedArrayWarning,
                        )
                    if isource < ufunc.nin <= idest:
                        dest.update(source)

            # compare iteration dimensions
            for iout, out in enumerate(res):
                if isinstance(out, np.generic):
                    continue
                for idim, dim in enumerate(iterdims[ufunc.nin + iout]):
                    destination = out.annotations[dim]
                    for in_, iterdim in zip(inputs, iterdims):
                        if (
                            idim >= len(iterdim)
                            or getattr(in_, "annotations", None) is None
                        ):
                            continue
                        source = in_.annotations[iterdim[idim]]
                        if not _match(source, destination):
                            warnings.warn(
                                f"incompatible annotations at broadcasted dimension",
                                AnnotatedArrayWarning,
                            )
                        destination.update(source)

        return res if istuple else res[0]

    def diagonal(self, offset=0, axis1=0, axis2=1):
        if not _match(self.annotations[axis1], self.annotations[axis2]):
            warnings.warn("incompatible annotations")
        res = np.asarray(self).diagonal(offset, axis1, axis2)
        res = res.view(type(self))
        res.annotations["global"].update(self.annotations["global"])
        for i, j in enumerate([k for k in range(self.ndim) if k not in (axis1, axis2)]):
            res.annotations[i].update(self.annotations[j])
        res.annotations[-1].update(self.annotations[axis1])
        res.annotations[-1].update(self.annotations[axis2])
        return res

    def trace(self, offset=0, axis1=0, axis2=1):
        if not _match(self.annotations[axis1], self.annotations[axis2]):
            warnings.warn("incompatible annotations")
        res = np.asarray(np.asarray(self).trace(offset, axis1, axis2))
        res = res.view(type(self))
        res.annotations["global"].update(self.annotations["global"])
        for i, j in enumerate([k for k in range(self.ndim) if k not in (axis1, axis2)]):
            res.annotations[i].update(self.annotations[j])
        return res

    def transpose(self, *args):
        while len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]
        if args == () or (len(args) == 1 and args[0] is None):
            args = (*range(self.ndim - 1, -1, -1),)
        res = np.asarray(self).transpose(args)
        res = res.view(type(self))
        res.annotations["global"].update(self.annotations["global"])
        for i, j in enumerate(args):
            res.annotations[i].update(self.annotations[j])
        return res

    @property
    def T(self):
        return self.transpose()

    def dot(self, other):
        if not isinstance(other, AnnotatedArray):
            other = np.asanyarray(other).view(AnnotatedArray)
        if not (
            _match(self.annotations["global"], other.annotations["global"])
            and (
                self.ndim == 0
                or other.ndim == 0
                or _match(
                    self.annotations[-1], other.annotations[-1 - int(other.ndim > 1)]
                )
            )
        ):
            warnings.warn("incompatible annotations", AnnotatedArrayWarning)
        res = np.asarray(np.asarray(self).dot(np.asarray(other)))
        res = res.view(type(self))
        res.annotations["global"].update(self.annotations["global"])
        res.annotations["global"].update(other.annotations["global"])
        for i in range(0, self.ndim - 1):
            res.annotations[i].update(self.annotations[i])
        offset = i + int(i > 0)
        for i, j in enumerate(
            [k for k in range(other.ndim) if k != max(other.ndim - 2, 0)]
        ):
            res.annotations[offset + i].update(other.annotations[j])
        return res

    def reshape(self, newshape):
        res = np.asarray(self).reshape(newshape)
        res = res.view(type(self))
        res.annotations["global"].update(self.annotations["global"])

        oldacc = newacc = 1

        i = 0

        for new, ann in zip(res.shape, res.annotations):
            while oldacc < newacc:
                i += 1
                oldacc *= self.shape[i]
            if oldacc == newacc and new == self.shape[i]:
                ann.update(self.annotations[i])
            newacc *= new
        return res

    def flatten(self):
        res = np.asarray(self).flatten()
        res = res.view(type(self))
        res.annotations["global"].update(self.annotations["global"])
        if res.shape == self.shape:
            res.annotations[0].update(self.annotations[0])
        elif len(_squeeze(self.shape)) == 1:
            res.annotations[0].update(self.annotations[self.shape.index(res.size)])
        return res

    def ravel(self):
        res = np.asarray(self).ravel()
        res = res.view(type(self))
        res.annotations["global"].update(self.annotations["global"])
        if res.shape == self.shape:
            res.annotations[0].update(self.annotations[0])
        elif len(_squeeze(self.shape)) == 1:
            res.annotations[0].update(self.annotations[self.shape.index(res.size)])
        return res

    def squeeze(self, axis=None):
        res = np.asarray(self).squeeze(axis)
        res = res.view(type(self))
        res.annotations["global"].update(self.annotations["global"])
        if axis is None:
            for i, j in enumerate([k for k, m in enumerate(self.shape) if m != 1]):
                res.annotations[i].update(self.annotations[j])
        else:
            axis = (axis,) if isinstance(axis, int) else axis
            axis = [a % self.ndim for a in axis]
            i = 0
            for ann in res.annotations:
                while i in axis:
                    i += 1
                ann.update(self.annotations[i])
                i += 1
        return res

    def argmax(self, axis=None, out=None, *, keepdims=np._NoValue):
        kwargs = (
            {} if keepdims is np._NoValue else {"keepdims": keepdims}
        )  # Needed for backward compatibility
        out = np.asarray(self).argmax(axis, out, *kwargs)
        if isinstance(out, np.generic):
            return
        keepdims = False if keepdims is np._NoValue else keepdims
        out = out.view(type(self))
        destination = out.annotations["global"]
        source = self.annotations["global"]
        if not _match(source, destination):
            warnings.warn(
                "incompatible annotations of category 'global'",
                AnnotatedArrayWarning,
            )
        destination.update(source)
        axis = axis % self.ndim - self.ndim
        dim_in = 0
        for dim in range(-1, -1 - out.ndim, -1):
            dim_in -= 2 if dim == axis and not keepdims else 1
            destination = out.annotations[dim]
            source = self.annotations[dim_in]
            if not _match(source, destination):
                warnings.warn(
                    f"incompatible annotations at dimension '{dim_in}'",
                    AnnotatedArrayWarning,
                )
            destination.update(source)
        return out

    def argmin(self, axis=None, out=None, *, keepdims=np._NoValue):
        kwargs = (
            {} if keepdims is np._NoValue else {"keepdims": keepdims}
        )  # Needed for backward compatibility
        out = np.asarray(self).argmin(axis, out, *kwargs)
        if isinstance(out, np.generic):
            return
        keepdims = False if keepdims is np._NoValue else keepdims
        out = out.view(type(self))
        destination = out.annotations["global"]
        source = self.annotations["global"]
        if not _match(source, destination):
            warnings.warn(
                "incompatible annotations of category 'global'",
                AnnotatedArrayWarning,
            )
        destination.update(source)
        axis = axis % self.ndim - self.ndim
        dim_in = 0
        for dim in range(-1, -1 - out.ndim, -1):
            dim_in -= 2 if dim == axis and not keepdims else 1
            destination = out.annotations[dim]
            source = self.annotations[dim_in]
            if not _match(source, destination):
                warnings.warn(
                    f"incompatible annotations at dimension '{dim_in}'",
                    AnnotatedArrayWarning,
                )
            destination.update(source)
        return out

    def argpartition(self, kth, axis=-1, kind="introselect", order=None):
        if axis is None:
            return self.flatten().argpartition(kth, -1, kind, order)
        return super().argpartition(kth, axis, kind, order)

    def argsort(self, axis=-1, kind=None, order=None):
        if axis is None:
            return self.flatten().argsort(-1, kind, order)
        return super().argsort(axis, kind, order)

    def compress(self, a, axis=None, out=None):
        pass


def _parse_signature(signature, inputs):
    signature = "".join(signature.split())  # remove whitespace
    sigin, sigout = signature.split("->")  # split input and output
    sigin = sigin[1:-1].split("),(")  # split input
    sigin = [i.split(",") for i in sigin]
    sigout = sigout[1:-1].split("),(")  # split output
    sigout = [i.split(",") for i in sigout]
    for i in range(len(sigin)):
        j = 0
        while j < len(sigin[i]):
            d = sigin[i][j]
            if d.endswith("?") and len(sigin[i]) > inputs[i].ndim:
                sigin = [[i for i in s if i != d] for s in sigin]
                sigout = [[i for i in s if i != d] for s in sigout]
            j += 1
    return sigin, sigout


def _squeeze(shape):
    return (*(i for i in shape if i != 1),)
