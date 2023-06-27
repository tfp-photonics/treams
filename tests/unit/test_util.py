import copy

import pytest
import numpy as np

from treams import util


class TestAnnotationDict:
    def test_init_dict(self):
        a = util.AnnotationDict({"a": 1, "b": 2})
        assert a == {"a": 1, "b": 2}

    def test_init_tuple(self):
        a = util.AnnotationDict((("a", 1), ("b", 2)))
        assert a == {"a": 1, "b": 2}

    def test_init_anndict(self):
        a = util.AnnotationDict({"a": 1, "b": 2})
        b = util.AnnotationDict(a)
        assert a == b

    def test_kwargs(self):
        with pytest.warns(util.AnnotationWarning):
            a = util.AnnotationDict({"a": 0, "c": 3}, a=1, b=2)
        assert a == {"a": 1, "b": 2, "c": 3}

    def test_getitem(self):
        a = util.AnnotationDict({"a": 1})
        assert a["a"] == 1

    def test_delitem(self):
        a = util.AnnotationDict({"a": 1})
        del a["a"]
        assert a == {}

    def test_iter(self):
        a = util.AnnotationDict({"a": 1, "b": 2})
        assert ["a", "b"] == list(a)

    def test_len(self):
        a = util.AnnotationDict({"a": 1, "b": 2})
        assert len(a) == 2

    def test_repr(self):
        a = util.AnnotationDict({"a": 1, "b": 2})
        assert repr(a) == "AnnotationDict({'a': 1, 'b': 2})"

    def test_match(self):
        a = util.AnnotationDict({"a": 1, "b": 2})
        b = {"a": 1, "c": 3}
        assert a.match(b) is None

    def test_match_fail(self):
        a = util.AnnotationDict({"a": 1, "b": 2})
        b = {"a": 2, "c": 3}
        with pytest.warns(util.AnnotationWarning):
            a.match(b)


class TestAnnotationSequence:
    def test_init(self):
        a = util.AnnotationSequence({"a": 1}, {"a": 2})
        assert a == ({"a": 1}, {"a": 2})

    def test_repr(self):
        a = util.AnnotationSequence({"a": 1}, {"a": 2}, mapping=dict)
        assert repr(a) == "AnnotationSequence({'a': 1}, {'a': 2})"

    def test_len(self):
        a = util.AnnotationSequence({"a": 1}, {"a": 2})
        assert len(a) == 2

    def test_getitem_int(self):
        a = util.AnnotationSequence({"a": 1}, {"a": 2})
        assert a[1] == {"a": 2}

    def test_getitem_tuple(self):
        a = util.AnnotationSequence({"a": 1}, {"a": 2})[()]
        assert a == ({"a": 1}, {"a": 2}) and isinstance(a, tuple)

    def test_getitem_slice(self):
        a = util.AnnotationSequence({"a": 1}, {"a": 2})
        assert a[1:] == ({"a": 2},) and isinstance(a, util.AnnotationSequence)

    def test_getitem_list(self):
        a = util.AnnotationSequence({"a": 1}, {"a": 2})
        assert a[[1, 0, 1]] == ({"a": 2}, {"a": 1}, {"a": 2})

    def test_getitem_error(self):
        a = util.AnnotationSequence({"a": 1}, {"a": 2})
        with pytest.raises(TypeError):
            a["fail"]

    def test_update_warn_len(self):
        a = util.AnnotationSequence({})
        with pytest.warns(util.AnnotationWarning):
            a.update(({}, {}))

    def test_update_warn_overwrite(self):
        a = util.AnnotationSequence({"a": 1})
        with pytest.warns(util.AnnotationWarning):
            a.update(({"a": 2},))

    def test_update(self):
        a = util.AnnotationSequence({"a": 1}, {"b": 2})
        a.update(({"c": 3}, {"a": 4}))
        assert a == util.AnnotationSequence({"a": 1, "c": 3}, {"b": 2, "a": 4})

    def test_match_warn_overwrite(self):
        a = util.AnnotationSequence({"a": 1})
        with pytest.warns(util.AnnotationWarning):
            a.match(({"a": 2},))

    def test_match(self):
        a = util.AnnotationSequence({"a": 1})
        assert a.match(({"b": 2},)) is None

    def test_eq_error_len_get(self):
        assert not (util.AnnotationSequence({}) == 1)

    def test_eq_error_len(self):
        assert not (util.AnnotationSequence({}, {}) == (None,))

    def test_eq_false(self):
        assert not (
            util.AnnotationSequence({"a": 1}) == util.AnnotationSequence({"a": 2})
        )

    def test_eq_true(self):
        assert util.AnnotationSequence({"a": 1}) == util.AnnotationSequence({"a": 1})

    def test_add(self):
        assert util.AnnotationSequence({"a": 1}) + (
            {"b": 2},
        ) == util.AnnotationSequence({"a": 1}, {"b": 2})

    def test_radd(self):
        assert ({"a": 1},) + util.AnnotationSequence(
            {"b": 2}
        ) == util.AnnotationSequence({"a": 1}, {"b": 2})


class TestSequenceAsDict:
    def test_set(self):
        ann = util.AnnotationSequence({}, {})
        ann.as_dict = {"a": (1, 2)}
        assert ann.as_dict["a"] == (1, 2)

    def test_getitem(self):
        ann = util.AnnotationSequence({"a": 1}, {"a": 1})
        assert ann.as_dict["a"] == (1, 1)

    def test_getitem_missing(self):
        ann = util.AnnotationSequence({})
        with pytest.raises(KeyError):
            ann.as_dict["fail"]

    def test_setitem_delnone(self):
        ann = util.AnnotationSequence({"a": 1}, {})
        ann.as_dict["a"] = (1,)
        assert ann.as_dict["a"] == (None, 1)

    def test_delitem(self):
        ann = util.AnnotationSequence({"a": 1})
        del ann.as_dict["a"]
        assert ann == util.AnnotationSequence({})

    def test_delitem_error(self):
        ann = util.AnnotationSequence({})
        with pytest.raises(KeyError):
            del ann.as_dict["fail"]

    def test_iter(self):
        ann = util.AnnotationSequence({"a": 1}, {"a": 1, "b": 2})
        assert set(iter(ann.as_dict)) == {"a", "b"}

    def test_len(self):
        ann = util.AnnotationSequence({"a": 1, "b": 2})
        assert len(ann.as_dict) == 2

    def test_repr(self):
        ann = util.AnnotationSequence({"a": 1}, {"a": 1, "b": 2})
        assert repr(ann.as_dict) in (
            "SequenceAsDict({'a': (1, 1), 'b': (None, 2)})",
            "SequenceAsDict({'b': (None, 2), 'a': (1, 1)})",
        )


class TestAnnotatedArray:
    def test_init(self):
        arr = util.AnnotatedArray([1, 2, 3], a=(1,))
        assert (arr == [1, 2, 3]).all() and arr.ann == ({"a": 1},)

    def test_getattr_same(self):
        arr = util.AnnotatedArray([[0, 1], [2, 3], [4, 5]], a=(0, 0))
        assert arr.a == 0

    def test_getattr(self):
        arr = util.AnnotatedArray([[0, 1], [2, 3], [4, 5]], a=(0, 1))
        assert arr.a == (0, 1)

    def test_setattr(self):
        arr = util.AnnotatedArray([[0, 1], [2, 3], [4, 5]], a=(1,))
        arr.a = (0, 1)
        assert arr.a == (0, 1)

    def test_delattr(self):
        arr = util.AnnotatedArray([[0, 1], [2, 3], [4, 5]], a=(1,))
        del arr.a
        assert arr.ann == util.AnnotationSequence({}, {})

    def test_copy(self):
        a = util.AnnotatedArray([[0, 1], [2, 3], [4, 5]], a=(1,))
        b = copy.copy(a)
        assert (a == b).all() and a.ann == b.ann and (a is not b)

    def test_ufunc_out(self):
        a = util.AnnotatedArray([[2, 3], [4, 5]], a=(2, 1))
        b = util.AnnotatedArray([2, 4], a=(1,))
        x = util.AnnotatedArray([[-1, -1], [-1, -1]])
        y = util.AnnotatedArray([[-1, -1], [-1, -1]])
        np.divmod(a, b, out=(x, y))
        assert (
            (x == [[1, 0], [2, 1]]).all()
            and (y == [[0, 3], [0, 1]]).all()
            and x.ann == y.ann == a.ann
        )

    def test_bool(self):
        a = util.AnnotatedArray([0, 0], a=(0,))
        with pytest.raises(ValueError):
            bool(a)

    def test_bool(self):
        a = util.AnnotatedArray([1], a=(0,))
        assert bool(a)

    def test_ufunc_reduce(self):
        x = util.AnnotatedArray([[1, 2], [3, 4]], a=(2, 1))
        y = np.add.reduce(x)
        assert (y == [4, 6]).all() and y.ann == ({"a": 1},)

    def test_ufunc_at(self):
        arr = util.AnnotatedArray([[2, 3], [4, 5]], a=(2, 1))
        np.add.at(arr, 1, [1, 2])
        assert (arr == [[2, 3], [5, 7]]).all() and arr.a == (2, 1)

    def test_ufunc_outer(self):
        x = util.AnnotatedArray([[2, 3], [4, 5]], a=(2, 1))
        y = util.AnnotatedArray([10, -1], a=(3,), b=(4,))
        z = np.add.outer(x, y)
        assert (
            (z == [[[12, 1], [13, 2]], [[14, 3], [15, 4]]]).all()
            and z.a == (2, 1, 3)
            and z.b == (None, None, 4)
        )

    def test_gufunc(self):
        x = util.AnnotatedArray([[1, 2], [3, 4]], a=(2, 1))
        y = util.AnnotatedArray([[[1], [-1]], [[10], [-1]]], a=(3, 1, None), b=(4, 4))
        z = x @ y
        assert (z == [[[-1], [-1]], [[8], [26]]]).all() and z.ann == (
            {"a": 3},
            {"a": 2},
            {"b": 4},
        )

    def test_str(self):
        assert str(util.AnnotatedArray([0, 1], a=(1,))) == "[0 1]"

    def test_repr(self):
        assert (
            repr(util.AnnotatedArray([0, 1], a=(1,)))
            == """AnnotatedArray(
    [0, 1],
    AnnotationSequence(AnnotationDict({'a': 1})),
)"""
        )

    def test_getitem_slice(self):
        x = util.AnnotatedArray([[1, 2], [3, 4]], a=(2, 1))
        y = x[:, 0]
        assert (y == [1, 3]).all() and y.ann == ({"a": 2},)

    def test_getitem_none(self):
        x = util.AnnotatedArray([[1, 2], [3, 4]], a=(2, 1))
        y = x[:, None, 0]
        assert (y == [[1], [3]]).all() and y.ann == ({"a": 2}, {})

    def test_getitem_ellipsis(self):
        x = util.AnnotatedArray([[1, 2], [3, 4]], a=(2, 1))
        y = x[1, ...]
        assert (y == [3, 4]).all() and y.ann == ({"a": 1},)

    def test_getitem_fancy(self):
        x = util.AnnotatedArray([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], a=(2, 1, 2))
        y = x[[True, False], :, [1, 0]]
        assert (y == [[1, 3], [0, 2]]).all() and y.ann == ({"a": 2}, {"a": 1})

    def test_getitem_ellipsis_implicit(self):
        x = util.AnnotatedArray([[1, 2], [3, 4]], a=(2, 1))
        y = x[1]
        assert (y == [3, 4]).all() and y.ann == ({"a": 1},)

    def test_setitem_slice(self):
        x = util.AnnotatedArray([[1, 2], [3, 4]], a=(0, 1))
        x[:, 1] = util.AnnotatedArray([5, 6], a=(0,))
        assert (x == [[1, 5], [3, 6]]).all() and x.ann == ({"a": 0}, {"a": 1})

    def test_setitem_none(self):
        x = util.AnnotatedArray([[1, 2], [3, 4]], a=(2, 1))
        x[:, None, 1] = util.AnnotatedArray([[5], [6]], a=(2, -1))
        assert (x == [[1, 5], [3, 6]]).all() and x.ann == ({"a": 2}, {"a": 1})

    def test_setitem_ellipsis(self):
        x = util.AnnotatedArray([[1, 2], [3, 4]], a=(2, 1))
        x[..., 1] = util.AnnotatedArray([5, 6], a=(2,))
        assert (x == [[1, 5], [3, 6]]).all() and x.ann == ({"a": 2}, {"a": 1})

    def test_setitem_fancy(self):
        x = util.AnnotatedArray([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], a=(1, 2, 1))
        x[[True, False], :, [1, 0]] = util.AnnotatedArray(
            [[-1, -2], [-3, -4]], a=(1, 2), b=(-1, -2)
        )
        assert (x == [[[-3, -1], [-4, -2]], [[4, 5], [6, 7]]]).all() and x.ann == (
            {"a": 1},
            {"a": 2},
            {"a": 1},
        )

    def test_transpose(self):
        x = util.AnnotatedArray([[0, 1], [2, 3]], a=(2, 3)).T
        assert (x == [[0, 2], [1, 3]]).all() and x.a == (3, 2)

    def test_any_empty(self):
        assert not util.AnnotatedArray([]).any()

    def test_any(self):
        x = util.AnnotatedArray([[1, 1], [1, 0], [0, 0]], a=(1, 2)).any(
            1, keepdims=True
        )
        assert (x == [[True], [True], [False]]).all() and x.a == (1, 2)

    def test_all_empty(self):
        assert util.AnnotatedArray([]).all()

    def test_all(self):
        x = util.AnnotatedArray([[1, 1], [1, 0], [0, 0]], a=(1, 2)).all(1)
        assert (x == [True, False, False]).all() and x.ann == ({"a": 1},)

    def test_max(self):
        assert util.AnnotatedArray([0, 1, 2], a=(1,)).max() == 2

    def test_min(self):
        assert util.AnnotatedArray([0, 1, 2], a=(1,)).min() == 0

    def test_sum(self):
        assert np.sum(util.AnnotatedArray([1, 2]), initial=-1) == 2

    def test_prod(self):
        assert np.prod(util.AnnotatedArray([1, 2]), initial=-1) == -2

    def test_cumsum(self):
        x = util.AnnotatedArray([[1], [0], [2]], a=(1, 1)).cumsum()
        assert (x == [1, 1, 3]).all() and x.ann == ({"a": 1},)

    def test_cumsum_axis(self):
        x = util.AnnotatedArray([[1, 1], [1, 0], [0, 0]], a=(1, 2)).cumsum(1)
        assert (x == [[1, 2], [1, 1], [0, 0]]).all() and x.a == (1, 2)

    def test_cumprod(self):
        x = util.AnnotatedArray([1, -1, 2], a=(1,)).cumprod()
        assert (x == [1, -1, -2]).all() and x.ann == ({"a": 1},)

    def test_cumprod_axis(self):
        x = util.AnnotatedArray([[1, 1], [1, 0], [0, 0]], a=(1, 2)).cumprod(1)
        assert (x == [[1, 1], [1, 0], [0, 0]]).all() and x.a == (1, 2)

    def test_trace(self):
        x = util.AnnotatedArray([[[1, 2, 3], [4, 5, 6]]], a=(1, 2, 3)).trace(1, 1, 2)
        assert x == 8 and x.ann.as_dict["a"] == (1,)

    def test_imag(self):
        x = util.AnnotatedArray([1 + 2j, 3 + 4j], a=(1,)).imag
        assert (x == [2, 4]).all() and x.ann.as_dict["a"] == (1,)

    def test_real(self):
        x = util.AnnotatedArray([1 + 2j, 3 + 4j], a=(1,)).real
        assert (x == [1, 3]).all() and x.ann.as_dict["a"] == (1,)

    def test_conjugate(self):
        x = util.AnnotatedArray([1 + 2j, 3 + 4j], a=(1,)).conjugate()
        assert (x == [1 - 2j, 3 - 4j]).all() and x.ann.as_dict["a"] == (1,)

    def test_diagonal(self):
        x = util.AnnotatedArray([[[1, 2, 3], [4, 5, 6]]], a=(1, 2, 3)).diagonal(1, 0, 2)
        assert (x == [[2], [5]]).all() and x.ann.as_dict["a"] == (2, None)


class TestImplements:
    def test_solve(self):
        m = util.AnnotatedArray([[1, 2], [3, -1]], a=(3, 1), b=(1,))
        b = util.AnnotatedArray([7, -7], a=(3,))
        a = np.linalg.solve(m, b)
        assert (a == [-1, 4]).all() and a.ann == ({"a": 1, "b": 1},)

    def test_solve_multiple(self):
        m = util.AnnotatedArray([[1, 2], [3, -1]], a=(3, 1), b=(1,))
        b = util.AnnotatedArray([[7, 3], [-7, 2]], a=(3, 2), b=(3, None))
        a = np.linalg.solve(m, b)
        assert (a == [[-1, 1], [4, 1]]).all() and a.ann == ({"a": 1, "b": 1}, {"a": 2})

    def test_lstsq(self):
        m = util.AnnotatedArray([[1, 2], [3, -1]], a=(3, 1), b=(1,))
        b = util.AnnotatedArray([7, -7], a=(3,))
        a = np.linalg.lstsq(m, b, None)[0]
        assert (np.abs(a - [-1, 4]) < 1e-14).all() and a.ann == ({"a": 1, "b": 1},)

    def test_lstsq_multiple(self):
        m = util.AnnotatedArray([[1, 2], [3, -1]], a=(3, 1), b=(1,))
        b = util.AnnotatedArray([[7, 3], [-7, 2]], a=(3, 2), b=(3, None))
        a = np.linalg.lstsq(m, b, None)[0]
        assert (np.abs(a - [[-1, 1], [4, 1]]) < 1e-14).all() and a.ann == (
            {"a": 1, "b": 1},
            {"a": 2},
        )

    def test_svd(self):
        m = util.AnnotatedArray([[5, 2], [2, 2]], a=(3, 1))
        u, s, v = np.linalg.svd(m)
        assert (
            (np.abs(s - [6, 1]) < 1e-14).all() and u.a == (3, None) and v.a == (None, 1)
        )

    def test_svd_no_uv(self):
        m = util.AnnotatedArray([[5, 2], [2, 2]], a=(3, 1))
        s = np.linalg.svd(m, compute_uv=False)
        assert (np.abs(s - [6, 1]) < 1e-14).all()

    def test_diag(self):
        x = np.diag(util.AnnotatedArray([[0, 1], [2, 3]], a=(3, 3)))
        assert (x == [0, 3]).all() and x.a == 3

    def test_diag_create(self):
        x = np.diag(util.AnnotatedArray([1], a=(2,)), -1)
        assert (x == [[0, 0], [1, 0]]).all() and x.a == 2

    def test_tril(self):
        x = np.tril(util.AnnotatedArray([[1, 2], [3, 4]], a=(5, 6)))
        assert (x == [[1, 0], [3, 4]]).all() and x.a == (5, 6)

    def test_triu(self):
        x = np.triu(util.AnnotatedArray([[1, 2], [3, 4]], a=(5, 6)), 1)
        assert (x == [[0, 2], [0, 0]]).all() and x.a == (5, 6)

    def test_zeros_like(self):
        x = np.zeros_like(util.AnnotatedArray([1, 2], a=1))
        assert (x == [0, 0]).all() and x.dtype == int and isinstance(x, np.ndarray)

    def test_ones_like(self):
        x = np.ones_like(util.AnnotatedArray([1, 2], a=1))
        assert (x == [1, 1]).all() and x.dtype == int and isinstance(x, np.ndarray)
