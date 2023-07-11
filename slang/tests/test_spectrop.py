from cgi import test
from slang.spectrop import Projector


def test_projector():
    p = Projector()
    # test covers changes to dataclasses in python 3.11
    # which disallows numpy arrays as default values
    assert p.scalings_ is not Projector.scalings_


test_projector()
