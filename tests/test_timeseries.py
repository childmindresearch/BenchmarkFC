import numpy as np
from scipy import sparse

from arfcexp.timeseries import (
    extract_timeseries,
    parc_one_hot_encode,
    preprocess_timeseries,
)


def test_extract_timeseries():
    rng = np.random.default_rng(42)
    n_samples, n_parcels, n_features = 100, 20, 200
    series = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    parc = rng.integers(0, n_parcels + 1, size=(n_features,))
    data_mask = rng.random(size=(n_features,)) < 0.9
    series = series * data_mask

    parc_one_hot = parc_one_hot_encode(parc, sparse=False)
    parc_one_hot_sprs: sparse.csr_array = parc_one_hot_encode(parc, sparse=True)

    assert sparse.issparse(parc_one_hot_sprs)
    assert parc_one_hot.shape == parc_one_hot_sprs.shape == (n_parcels, n_features)
    assert np.allclose(parc_one_hot, parc_one_hot_sprs.toarray())

    parc_series = extract_timeseries(series, parc_one_hot)
    parc_series_sprs = extract_timeseries(series, parc_one_hot_sprs)
    assert parc_series.dtype == parc_series_sprs.dtype == np.float32
    assert parc_series.shape == parc_series_sprs.shape == (n_samples, n_parcels)
    assert np.allclose(parc_series, parc_series_sprs, atol=1e-6)


def test_preprocess_timeseries():
    rng = np.random.default_rng(42)

    # 8 cycles / 100 samples / 2 secs per cycle = 0.04 hz
    # frequency range: (0.04, 0.08 hz)
    # global signal: 0.01 hz
    n_samples, n_features = 100, 50
    n_cycles = 8
    n_gs_cycles = 2
    tr = 2.0

    # synthetic clean time series
    theta = np.linspace(0, n_cycles * 2 * np.pi, n_samples)
    phase = np.linspace(0, np.pi, n_features)
    rate = np.linspace(1, 2, n_features)
    series = np.sin((theta[:, None] + phase) * rate)

    # add global signal and random noise
    mean = 2.0 * np.sin(np.linspace(0, n_gs_cycles * 2 * np.pi, n_samples))
    noise = 0.2 * rng.normal(size=(n_samples, n_features))

    # random missing samples
    sample_mask = rng.random(size=n_samples) < 0.9

    # raw noisy data
    raw = series + mean[:, None] + noise
    raw = np.where(sample_mask[:, None], raw, np.nan)

    def _checks(clean_series: np.ndarray):
        assert clean_series.shape == (n_samples, n_features)
        assert not np.any(np.isnan(clean_series))

    # basic checks, nb we also visually check in a separate notebook
    _checks(preprocess_timeseries(raw, tr=tr, sample_mask=sample_mask))
    _checks(preprocess_timeseries(raw, tr=tr, sample_mask=None))
    _checks(preprocess_timeseries(raw, tr=tr, sample_mask=sample_mask, pad=None))
    _checks(preprocess_timeseries(raw, tr=tr, sample_mask=sample_mask, gsr=False))
