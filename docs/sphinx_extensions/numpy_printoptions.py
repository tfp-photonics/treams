import numpy as np

def setup(app):
    """Set numpy print options for the documentation."""

    npversion = (np.__version__).split(".")
    npversion = float(npversion[0]) + float(npversion[2])/10 
    if npversion < 2.2:
        np.set_printoptions(precision=3, suppress=True)
    else:
        np.set_printoptions(precision=3, legacy="1.25", suppress=True)