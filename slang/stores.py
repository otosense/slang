"""Tools to provide persistence for slang"""

import os
from functools import partial
from typing import Union, Callable, Iterable
from os.path import join as path_join, sep as path_sep

from dol import (
    Files,
    wrap_kvs,
    mk_dirs_if_missing as mk_dirs_if_missing_wrap,
    add_ipython_key_completions,
    Pipe,
    DirReader,
)

from recode import mk_codec  # https://otosense.github.io/recode/module_docs/recode.html

from slang.settings import SLANG_DATA_DIR
from slang import dflt_snips_to_str

SnipsCodecFunc = Callable[[Iterable[int]], bytes]
SnipsCodec = Union[SnipsCodecFunc, str]
SnipsStrFunc = Callable[[Iterable[int]], str]

# B for "unsigned char" (0-255)
# H for "unsigned short (0-65535 -- uint16)"
# I for "unsigned int (0-(2**32-1) -- uint32)"
# see https://docs.python.org/3/library/struct.html#format-characters
DFLT_SNIPS_CODEC = mk_codec('H')


class Rootdirs:
    def __init__(self, rootdir=SLANG_DATA_DIR) -> None:
        self.rootdir = rootdir

    @property
    def spaces(self):
        return os.path.join(self.rootdir, 'spaces')

    def space(self, space: str):
        return os.path.join(self.spaces, space)

    def snips(self, space: str):
        return os.path.join(self.space(space), 'snips', 'd')


def snips_store_wrap(
    *,
    snips_to_str: Union[bool, SnipsStrFunc] = False,
    codec: SnipsCodec = DFLT_SNIPS_CODEC,  # TODO: extend to adapt codec to alphabet size
    ipython_key_completions: bool = True,
    additional_wrappers: Iterable[Callable] = (),
):
    """Make a local snips store for the given space"""

    def gen_wrappers():
        yield wrap_kvs(obj_of_data=codec.decode, data_of_obj=codec.encode)

        if snips_to_str:
            if snips_to_str is True:
                snips_to_str_func = dflt_snips_to_str
            else:
                snips_to_str_func = snips_to_str
            assert callable(snips_to_str_func)
            yield wrap_kvs(obj_of_data=snips_to_str_func)

        if ipython_key_completions:
            yield add_ipython_key_completions

    return Pipe(*gen_wrappers(), *additional_wrappers)


store_wrap = snips_store_wrap()


# TODO: Figure out how to make wrap-parametrized snips stores and malls!!!
#   This is limited!
@store_wrap
class LocalSnipsStore(Files):
    def __init__(
        self,
        rootdir=SLANG_DATA_DIR,
        # snips_to_str: Union[bool, SnipsStrFunc] = False,
        # codec: SnipsCodec = DFLT_SNIPS_CODEC,  # TODO: extend to adapt codec to alphabet size
        # ipython_key_completions: bool = True,
        # additional_wrappers: Iterable[Callable] = (),
    ):
        super().__init__(rootdir)
        # store_wrap = snips_store_wrap(
        #     snips_to_str=snips_to_str,
        #     codec=codec,
        #     ipython_key_completions=ipython_key_completions,
        #     additional_wrappers=additional_wrappers,
        # )
        # store_wrap(self)

    def with_snips_to_str(self, snips_to_str: Union[bool, SnipsStrFunc] = True):
        return wrap_kvs(self, obj_of_data=snips_to_str)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.rootdir!r})'

    """Store of snips"""


def snips_mall(
    rootdir=SLANG_DATA_DIR,
    *,
    snips_to_str: Union[bool, SnipsStrFunc] = False,
    **snips_stores_kwargs,
):
    spaces_rootdir = Rootdirs(rootdir).spaces
    # TODO: Just too ugle this snips_to_str handling! Figure it out!
    _mk_local_snips_store = partial(LocalSnipsStore, **snips_stores_kwargs)
    if snips_to_str:
        mk_local_snips_store = lambda rootdir: _mk_local_snips_store(rootdir).with_snips_to_str()
    else:
        mk_local_snips_store = _mk_local_snips_store
    return wrap_kvs(
        DirReader(spaces_rootdir, max_levels=0).with_relative_paths(),
        key_of_id=lambda id: id[:-1],  # TODO: use picklable func
        id_of_key=lambda key: key + path_sep,  # TODO: use picklable func
        # TODO: the path_join(v.rootdir, 'snips', 'd') should be centralized to Rootdirs
        postget=lambda k, v: mk_local_snips_store(path_join(v.rootdir, 'snips', 'd')),
    )


# ---------------------------- old ----------------------------
def local_snips_store(
    space: str = 'temp',
    *,
    rootdir=SLANG_DATA_DIR,
    snips_to_str: Union[bool, SnipsStrFunc] = False,
    codec: SnipsCodec = DFLT_SNIPS_CODEC,  # TODO: extend to adapt codec to alphabet size
    mk_dirs_if_missing: bool = True,
    ipython_key_completions: bool = True,
    name: str = 'LocalSnipsStore',
):
    """Make a local snips store for the given space"""

    if space is None:
        # from dol import Collection
        return Files(Rootdirs(rootdir).spaces)
    else:

        def mk_store_wrap():
            additional_wrappers = []
            if mk_dirs_if_missing:
                additional_wrappers = [mk_dirs_if_missing_wrap]

            return snips_store_wrap(
                snips_to_str=snips_to_str,
                codec=codec,
                ipython_key_completions=ipython_key_completions,
                additional_wrappers=additional_wrappers,
            )

        # make instance
        store_wrap = mk_store_wrap()
        snips_store_cls = store_wrap(Files)
        snips_store_cls.__name__ = name
        snips_rootdir = Rootdirs(rootdir).snips
        if mk_dirs_if_missing:
            os.makedirs(snips_rootdir, exist_ok=True)
        return snips_store_cls(rootdir)


# def _snips_store_wrap(
#     *,
#     snips_to_str: Union[bool, SnipsStrFunc] = False,
#     codec: SnipsCodec = DFLT_SNIPS_CODEC,  # TODO: extend to adapt codec to alphabet size
#     ipython_key_completions: bool = True,
#     additional_wrappers: Iterable[Callable] = (),
# ):
#     """Make a local snips store for the given space"""

#     def gen_wrappers():
#         yield wrap_kvs(obj_of_data=codec.decode, data_of_obj=codec.encode)

#         if snips_to_str:
#             if snips_to_str is True:
#                 snips_to_str_func = dflt_snips_to_str
#             else:
#                 snips_to_str_func = snips_to_str
#             assert callable(snips_to_str_func)
#             yield wrap_kvs(obj_of_data=snips_to_str_func)

#         if ipython_key_completions:
#             yield add_ipython_key_completions

#     return Pipe(*gen_wrappers(), *additional_wrappers)
