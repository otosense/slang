from odat.mdat.freesounds import mk_dacc
import numpy as np
from functools import partial
from i2 import FuncFactory
from slang import local_snips_store
from slang.util import pickle_load, pickle_dump


def example_slang_train(
    dacc,
    *,
    wf_to_chks=2028,
    chk_to_fv=50,
    fv_to_snip=250,
    save_to='last_snipper',
    verbose=True,
):
    from slang import mk_chunker
    from slang.snippers import LdaChkToFv, PcaChkToFv, KMeansFvToSnip
    from slang import mk_chk_fft
    from i2 import Pipe
    from functools import partial
    from lkj import clog, print_progress

    _clog = clog(verbose, log_func=print_progress)

    n_unik_tags = 50  # default
    try:
        n_unik_tags = len(dacc.tags)
        n_sounds = len(dacc.tag_counts.values())
        _clog(f'{n_unik_tags=} {n_sounds=}')
    except Exception:
        pass

    def mk_spectrizer(chk_size, chk_step=None):
        chunker = mk_chunker(chk_size, chk_step)
        to_fft = mk_chk_fft(chk_size=chk_size)
        return Pipe(chunker, partial(map, to_fft))

    if isinstance(wf_to_chks, int):
        chk_size = wf_to_chks
        wf_to_chks = mk_spectrizer(chk_size)
    if isinstance(chk_to_fv, int):
        n_components = chk_to_fv
        chk_to_fv = PcaChkToFv(n_components)
    if isinstance(fv_to_snip, int):
        n_clusters = wf_to_chks
        fv_to_snip = KMeansFvToSnip(n_clusters=n_clusters)
        # fv_to_snip = {'n_clusters': n_clusters}

    # Just to make sure wf_to_chks works.
    chk, tag = next(dacc.chk_tag_pairs(wf_to_chks=wf_to_chks))
    assert chk.shape == (1025,), f'chk does not seem to be a spectrum'

    from slang import ClassificationSnipper
    from slang.snippers import PcaChkToFv, LdaChkToFv

    def save_snipper(snipper, save_to=save_to):
        clog(f"Saving to {save_to}")
        pickle_dump(snipper, save_to)

    try:
        snipper = ClassificationSnipper(
            wf_to_chks=wf_to_chks,
            chk_to_fv=chk_to_fv,
            fv_to_snip={'n_clusters': n_unik_tags * 5},
        )

        _clog(f'fit_chk_to_fv')
        snipper = snipper.fit_chk_to_fv(dacc.chk_tag_pairs(snipper.wf_to_chks))

        _clog(f'fit_fv_to_snip')
        # fit on (fv, tag) pairs
        snipper = snipper.fit_fv_to_snip(
            dacc.fv_tag_pairs(snipper.wf_to_chks, snipper.chk_to_fv)
        )

        # _clog(f"fit_snip_to_score")
        # snipper = snipper.fit_snip_to_score(  # fit on (snip tag) pairs
        #     (
        #         (snipper.fv_to_snip(fv), tags)
        #         for fv, tags in dacc.fv_tag_pairs(snipper.wf_to_chks, snipper.chk_to_fv)
        #     )
        # )
    except (Exception, KeyboardInterrupt) as e:
        print(f"error training snipper: {e}")
        error_save_to = save_to + '.error.snipper.pkl'
        save_snipper(snipper, error_save_to)
        raise e

    save_snipper(snipper, save_to)

    return snipper


def mk_train_dacc():
    return mk_dacc(audio_key='audio_train/', annots_key='train_post_competition.csv')


def train_snipper(
    mk_dacc='odat.mdat.freesounds.mk_dacc',
    save_to='last_snipper',
    verbose=True,
    *,
    dacc_kwargs=(),
):
    if isinstance(mk_dacc, str):
        import lkj

        mk_dacc_location_dot_path = mk_dacc
        mk_dacc = lkj.import_object(mk_dacc_location_dot_path)

    if isinstance(dacc_kwargs, str):
        import json

        dacc_kwargs = json.loads(dacc_kwargs)
    dacc_kwargs = dict(dacc_kwargs)
    dacc = mk_dacc(**dacc_kwargs)

    snipper = example_slang_train(dacc, save_to=save_to, verbose=verbose)

    return snipper


def resolve_resource(resource):
    if isinstance(resource, FuncFactory):
        resource_factory = resource
        resource = resource_factory()
    return resource


def compute_snips(
    snipper=FuncFactory(partial(pickle_load, 'a_snipper.pkl')),
    dacc=FuncFactory(mk_train_dacc),
    snips_store=FuncFactory(partial(local_snips_store, 'freesounds_train')),
    *,
    verbose=True,
):
    from lkj import clog, print_progress

    _clog = clog(verbose, log_func=print_progress)

    _clog('-------------------------------------------')
    _clog('Resolving resources...')
    snipper = resolve_resource(snipper)
    dacc = resolve_resource(dacc)
    snips_store = resolve_resource(dacc)
    _clog(f'Resources: {snipper=} {dacc=} {snips_store=}')
    _clog('-------------------------------------------')
    _clog('Looping through audio, computing and saving snips, and learning scores')
    n = len(dacc.wfs)
    for i, (k, wf) in enumerate(dacc.wfs.items(), 1):
        _clog(f'({i}/{n}) processing {k} ({len(wf)=})')
        snips = np.array(list(snipper(wf)))
        snips_store[k] = snips
        snipper


# _clog(f"fit_snip_to_score")
# snipper = snipper.fit_snip_to_score(  # fit on (snip tag) pairs
#     (
#         (snipper.fv_to_snip(fv), tags)
#         for fv, tags in dacc.fv_tag_pairs(snipper.wf_to_chks, snipper.chk_to_fv)
#     )
# )

if __name__ == '__main__':
    from argh import dispatch_command

    dispatch_command(compute_snips)
