import warnings

from numpy import linalg as _la

from ptsa.numpy._annotatedarray import AnnotatedArray, AnnotatedArrayWarning


def solve(a, b):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AnnotatedArrayWarning)
        res = _la.solve(a, b)

    if not isinstance(res, AnnotatedArray):
        return res

    if not isinstance(a, AnnotatedArray):
        a = AnnotatedArray(a)
    if not isinstance(b, AnnotatedArray):
        b = AnnotatedArray(b)

    res.annotations = None

    if not (
        (
            {} in (a.annotations["global"], b.annotations["global"])
            or a.annotations["global"] == b.annotations["global"]
        )
        and (a.annotations[-2] == b.annotations[-1 - int(b.ndim != a.ndim - 1)],)
    ):
        warnings.warn("incompatible annotations", AnnotatedArrayWarning)

    res.annotations["global"].update(a.annotations["global"])
    res.annotations["global"].update(b.annotations["global"])

    if b.ndim == a.dim - 1:
        for i in range(-1, -1 - b.ndim, -1):
            res.annotations[i - 1].update(a.annotations[i - 1])
            if (
                {} not in (res.annotations[i], b.annotations[i])
                and res.annotations[i] != b.annotations[i]
            ):
                warnings.warn("incompatible annotations", AnnotatedArrayWarning)
            res.annotations[i].update(b.annotations[i])
    else:
        res.annotations[-1].update(b.annotations[-1])
        for x in (a, b):
            for i in range(-2, -1 - x.ndim, -1):
                if (
                    {} not in (res.annotations[i], x.annotations[i])
                    and res.annotations[i] != x.annotations[i]
                ):
                    warnings.warn("incompatible annotations", AnnotatedArrayWarning)
                res.annotations[i].update(x.annotaions[i])

    return res


def lstsq(a, b, rcond="warn"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AnnotatedArrayWarning)
        res = _la.lstsq(a, b, rcond)

    if not isinstance(res[0], AnnotatedArray):
        return res

    if not isinstance(a, AnnotatedArray):
        a = AnnotatedArray(a)
    if not isinstance(b, AnnotatedArray):
        b = AnnotatedArray(b)

    res[0].annotations = None
    res[1].annotations = None

    if not (
        (
            {} in (a.annotations["global"], b.annotations["global"])
            or a.annotations["global"] == b.annotations["global"]
        )
        and (a.annotations[0] == b.annotations[0],)
    ):
        warnings.warn("incompatible annotations", AnnotatedArrayWarning)

    for i in range(2):
        res[i].annotations["global"].update(a.annotations["global"])
        res[i].annotations["global"].update(b.annotations["global"])
    res[0].annotations[0] = a.annotations[1]
    if b.ndim == 2:
        res[0].annotations[1] = b.annotations[1]
        res[1].annotations[0] = b.annotations[1]
    return res
