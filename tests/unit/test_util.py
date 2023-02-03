import pytest

from ptsa import util


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
        a = util.AnnotationSequence({"a": 1}, {"a": 2}, container=dict)
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
