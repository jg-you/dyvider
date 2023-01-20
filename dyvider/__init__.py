"""
dyvider

Authors:
Alice Patania <alice.patania@uvm.edu>
Jean-Gabriel Young <jean-gabriel.young@uvm.edu>
"""
import pkg_resources

from . import (
    utilities,
    objectives,
    algorithms,
    layers
)


__version__ = pkg_resources.require("dyvider")[0].version