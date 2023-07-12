from slang.spectrop import Projector


def test_projector():
    Projector()
    # test covers changes to dataclasses in python 3.11
    # which disallows numpy arrays as default values
