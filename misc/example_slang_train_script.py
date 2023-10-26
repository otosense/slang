def example_slang_train(dacc, *, save_to='last_snipper', verbose=True):
    from slang import mk_chunker
    from slang.snippers import LdaChkToFv, PcaChkToFv
    from slang import mk_chk_fft
    from i2 import Pipe
    from functools import partial
    from pathlib import Path
    import pickle
    from lkj import clog, print_progress

    _clog = clog(verbose, log_func=print_progress)

    n_unik_tags = 50  # default
    try:
        n_unik_tags = len(dacc.tags)
        n_sounds = len(dacc.tag_counts.values())
        _clog(f'{n_unik_tags=} {n_sounds=}')
    except Exception:
        pass

    def mk_spectrizer(chk_size=2048, chk_step=None):
        chunker = mk_chunker(chk_size, chk_step)
        to_fft = mk_chk_fft(chk_size=chk_size)
        return Pipe(chunker, partial(map, to_fft))

    spectrizer = mk_spectrizer()

    chk, tag = next(dacc.chk_tag_pairs(wf_to_chks=spectrizer))
    assert chk.shape == (1025,), f'chk does not seem to be a spectrum'

    from slang import ClassificationSnipper
    from slang.snippers import PcaChkToFv, LdaChkToFv

    try:
        snipper = ClassificationSnipper(
            wf_to_chks=spectrizer,
            chk_to_fv=PcaChkToFv(n_components=73),
            fv_to_snip={'n_clusters': n_unik_tags * 5},
        )

        _clog(f'fit_chk_to_fv')
        snipper = snipper.fit_chk_to_fv(dacc.chk_tag_pairs(snipper.wf_to_chks))

        _clog(f'fit_fv_to_snip')
        # fit on (fv, tag) pairs
        snipper = snipper.fit_fv_to_snip(
            dacc.fv_tag_pairs(snipper.wf_to_chks, snipper.chk_to_fv)
        )

        _clog(f'fit_snip_to_score')
        snipper = snipper.fit_snip_to_score(  # fit on (snip tag) pairs
            (
                (snipper.fv_to_snip(fv), tags)
                for fv, tags in dacc.fv_tag_pairs(snipper.wf_to_chks, snipper.chk_to_fv)
            )
        )
    except Exception as e:
        print(f'error training snipper: {e}')
        error_save_to = save_to + '.error.snipper.pkl'
        print(f'Saving to {error_save_to}')
        with open(error_save_to, 'wb') as f:
            pickle.dump(snipper, f)

    clog(f'Saving to {save_to}')
    with open(save_to, 'wb') as f:
        pickle.dump(snipper, f)

    return snipper


def train_snipper(
    mk_dacc='odat.mdat.freesounds.mk_dacc', save_to='last_snipper', verbose=True
):
    if isinstance(mk_dacc, str):
        import lkj

        mk_dacc_location_dot_path = mk_dacc
        mk_dacc = lkj.import_object(mk_dacc_location_dot_path)

    dacc = mk_dacc()

    snipper = example_slang_train(dacc, save_to=save_to, verbose=verbose)

    return snipper


if __name__ == '__main__':
    from argh import dispatch_command

    dispatch_command(train_snipper)
