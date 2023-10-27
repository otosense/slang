"""Tools to provide persistence for slang"""

import os
from typing import Union, Callable, Iterable
from dol import (
    Files,
    wrap_kvs,
    mk_dirs_if_missing as mk_dirs_if_missing_wrap,
    add_ipython_key_completions,
    Pipe,
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
    rootdir = os.path.join(rootdir, 'spaces', space, 'snips', 'd')
    if mk_dirs_if_missing:
        os.makedirs(rootdir, exist_ok=True)
    return snips_store_cls(rootdir)
