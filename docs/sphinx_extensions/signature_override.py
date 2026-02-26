import numpy as np

def preprocess_ufunc(app, what, name, obj, options, lines):
    """Remove the first four docstring lines from numpy ufuncs.

    The docstring of ufuncs have a line containing the function name and signature and a
    blank line automatically prepended. The signature contains all arguments and keyword
    arguments but the positional arguments only get dummy variable names ('x1', 'x2',
    ...) and is therefore often confusing.

    By convention all ufuncs contain in their first manually added lines the function
    name with a simplified signature followed by a blank line. We explicitly add the
    signature with a separate process to autodoc.

    The four lines containing function names and signatures or being blank are therefore
    removed for a clean documentation produced by sphinx.
    """
    if isinstance(obj, np.ufunc):
        lines.pop(0)
        lines.pop(0)
        lines.pop(0)
        lines.pop(0)


def ufunc_signature(app, what, name, obj, options, signature, return_annotation):
    """Extract the signature from the docstring for numpy ufuncs.

    As described in :func:`preprocess_ufunc` the third line in a ufunc docstring
    contains the manually added function name and signature intended to be displayed in
    the documentation (the preceeding two lines are automatically added). We extract the
    signature and add it manually.
    """
    if isinstance(obj, np.ufunc):
        signature = "(" + obj.__doc__.split("\n")[2].split("(")[1]
    return signature, return_annotation


def setup(app):
    """Add processes to autodoc."""
    app.connect("autodoc-process-docstring", preprocess_ufunc)
    app.connect("autodoc-process-signature", ufunc_signature)