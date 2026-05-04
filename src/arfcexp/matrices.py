import concurrent.futures
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.dataset as pads
import pyarrow.parquet as pq
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

def import_matrix(mat: np.ndarray, *, require_square: bool = True) -> np.ndarray:
    """Reshape a flat or 2-D array to a square float64 matrix.
    """
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim == 1:
        n = int(np.sqrt(arr.size))
        if n * n != arr.size:
            raise ValueError(
                f"Cannot reshape 1-D array of length {arr.size} to a square matrix."
            )
        arr = arr.reshape(n, n)
    elif arr.ndim == 2:
        if require_square and arr.shape[0] != arr.shape[1]:
            raise ValueError(f"Matrix must be square, got shape {arr.shape}.")
    else:
        raise ValueError("Expected a 1-D or 2-D array.")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
 
 
def collapse_maxabs(A: np.ndarray) -> np.ndarray:
    """Symmetrise a (possibly asymmetric) matrix by keeping the signed value
    from whichever direction has the larger absolute value."""
    absA, absAT = np.abs(A), np.abs(A.T)
    W = np.where(absA >= absAT, A, A.T)
    return 0.5 * (W + W.T)

class EfficientMatrixReader:
    """Memory-efficient reader for large parquet files with matrix columns.

    The standard approach of filtering a LazyFrame and collecting loads entire
    row groups (~1.4 GB per group for the ``mat`` column), causing 30+ GB memory
    spikes.  This class pre-builds a lightweight index of non-matrix columns, then
    uses PyArrow to read only the specific rows needed, reducing memory usage to
    near zero.

    Parameters
    ----------
    parquet_path : str | Path
        Path to the parquet file.
    index_columns : list[str] | None
        Columns to include in the lightweight index.  If *None* (default), all
        columns except ``mat```, ``scores``, and ``err`` are indexed.

    Examples
    --------
    >>> reader = EfficientMatrixReader("/path/to/hcp.parquet")
    >>> mats = reader.get_matrices("pyspi", "cov_EmpiricalCovariance", limit=5)
    >>> df = reader.query(columns=["mat", "sub"], method="skarf", func="linear_ridge", lag=0)
    """

    # Heavy/large columns excluded from the index by default.
    # Includes list columns and large string columns that are not useful for filtering.
    _HEAVY_COLUMNS = {"mat", "scores", "err"}

    def __init__(self, parquet_path: str | Path, index_columns: list[str] | None = None):
        self.parquet_path = str(parquet_path)
        self.pf = pq.ParquetFile(self.parquet_path)
        self._index_columns = index_columns
        self._index_df: pl.DataFrame | None = None
        self._row_group_offsets: list[tuple[int, int, int]] | None = None

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _resolve_index_columns(self) -> list[str]:
        """Determine which columns to index."""
        if self._index_columns is not None:
            return list(self._index_columns)
        schema = self.pf.schema_arrow
        return [f.name for f in schema if f.name not in self._HEAVY_COLUMNS]

    def _build_index(self) -> None:
        """Build index of filter columns with row numbers (one-time)."""
        if self._index_df is not None:
            return

        cols = self._resolve_index_columns()
        logger.info("Building row index (columns=%s) ...", cols)
        df = (
            pl.scan_parquet(self.parquet_path)
            .select(cols)
            .with_row_index()
            .collect()
        )
        assert isinstance(df, pl.DataFrame)
        self._index_df = df

        self._row_group_offsets = []
        cumulative = 0
        for rg_idx in range(self.pf.metadata.num_row_groups):
            rg = self.pf.metadata.row_group(rg_idx)
            self._row_group_offsets.append((cumulative, cumulative + rg.num_rows, rg_idx))
            cumulative += rg.num_rows

        logger.info(
            "Index built: %d rows, %d row groups",
            len(self._index_df),
            len(self._row_group_offsets),
        )

    def _find_row_group(self, row_idx: int) -> tuple[int, int]:
        """Map a global row index to ``(row_group_idx, local_idx)``."""
        assert self._row_group_offsets is not None
        for start, end, rg_idx in self._row_group_offsets:
            if start <= row_idx < end:
                return rg_idx, row_idx - start
        raise ValueError(f"Row index {row_idx} out of range")

    @property
    def index(self) -> pl.DataFrame:
        """The lightweight index DataFrame (built lazily on first access)."""
        self._build_index()
        assert self._index_df is not None
        return self._index_df

    # ------------------------------------------------------------------
    # Filtering helpers
    # ------------------------------------------------------------------

    def _apply_filters(self, df: pl.DataFrame, **filters: object) -> pl.DataFrame:
        """Apply keyword filters to a polars DataFrame.

        Raises ``ValueError`` if a filter key is not present in the DataFrame.
        A filter value of ``None`` is translated to an ``is_null()`` check.
        """
        indexed_cols = set(df.columns)
        for key, value in filters.items():
            if key not in indexed_cols:
                raise ValueError(
                    f"Filter column {key!r} not in index. "
                    f"Available columns: {sorted(indexed_cols - {'index'})}"
                )
            if value is None:
                df = df.filter(pl.col(key).is_null())
            else:
                df = df.filter(pl.col(key) == value)
        return df

    # ------------------------------------------------------------------
    # Data retrieval
    # ------------------------------------------------------------------

    def _read_rows(self, row_indices: list[int], columns: list[str]) -> list[dict]:
        """Read specific rows and columns from the parquet file.

        Returns a list of dicts, one per row, with the requested columns.
        """
        # Group by row group to minimise I/O
        rg_rows: dict[int, list[tuple[int, int]]] = {}  # rg_idx -> [(local, global), ...]
        for global_idx in row_indices:
            rg_idx, local_idx = self._find_row_group(global_idx)
            rg_rows.setdefault(rg_idx, []).append((local_idx, global_idx))

        results: list[tuple[int, dict]] = []
        for rg_idx in sorted(rg_rows):
            table = self.pf.read_row_group(rg_idx, columns=columns)
            for local_idx, global_idx in rg_rows[rg_idx]:
                row = {}
                for col in columns:
                    value = table[col][local_idx].as_py()
                    row[col] = value
                results.append((global_idx, row))

        # Restore original ordering
        results.sort(key=lambda x: row_indices.index(x[0]))
        return [r[1] for r in results]
    
    def query(
        self,
        columns: list[str] | None = None,
        limit: int | None = None,
        **filters: object,
    ) -> pl.DataFrame:
        """General-purpose query returning a polars DataFrame.

        Parameters
        ----------
        columns : list[str] | None
            Data columns to retrieve from the parquet file (e.g. ``["mat", "sub"]``).
            Defaults to ``["mat"]``.
        limit : int | None
            Maximum number of rows.  *None* returns all matches.
        **filters
            Filter kwargs matching index column names (e.g. ``method="pyspi"``,
            ``func="cov_EmpiricalCovariance"``, ``lag=0``).

        Returns
        -------
        pl.DataFrame
            DataFrame with the requested columns plus all index columns.
        """
        if columns is None:
            columns = ["mat"]

        self._build_index()
        assert self._index_df is not None

        filtered = self._apply_filters(self._index_df, **filters)
        if limit is not None:
            filtered = filtered.limit(limit)

        matching = filtered.select("index").to_series().to_list()
        if not matching:
            # Return empty DataFrame with expected columns
            all_cols = {c: pl.Series([], dtype=pl.Utf8) for c in columns}
            return pl.DataFrame(all_cols)

        rows = self._read_rows(matching, columns=columns)

        # Merge index metadata with retrieved data columns
        index_rows = filtered.to_dicts()
        for idx_row, data_row in zip(index_rows, rows):
            idx_row.update(data_row)

        return pl.DataFrame(index_rows)

    def get_matrices(
        self,
        method: str,
        func: str,
        success: bool = True,
        limit: int | None = None,
        **filters: object,
    ) -> list[np.ndarray]:
        """Get matrices for a specific method/func combination.

        This is a convenience wrapper around :meth:`query` for the common case of
        retrieving flattened matrices as numpy arrays.

        Parameters
        ----------
        method : str
            Method name (``"pyspi"`` or ``"skarf"``).
        func : str
            Function / SPI name.
        success : bool
            Filter for successful computations (default ``True``).
        limit : int | None
            Maximum number of matrices to return.  *None* returns all matches.
        **filters
            Additional filter kwargs applied to the index (e.g. ``lag=0``,
            ``sub="100206"``).

        Returns
        -------
        list[np.ndarray]
            Flattened matrices as numpy arrays.
        """
        self._build_index()
        assert self._index_df is not None

        # Build combined filters
        all_filters = {"method": method, "func": func, "success": success, **filters}

        filtered = self._apply_filters(self._index_df, **all_filters)
        if limit is not None:
            filtered = filtered.limit(limit)

        matching = filtered.select("index").to_series().to_list()
        if not matching:
            return []

        rows = self._read_rows(matching, columns=["mat"])
        return [np.asarray(r["mat"]) for r in rows]

    def batch_get_matrices(
        self,
        queries: list[dict],
        limit: int | None = None,
        max_workers: int = 8,
    ) -> list[list[np.ndarray]]:
        """Fetch matrices for multiple queries in a single parquet pass.

        Reads each row group **at most once** across all queries, rather than
        once per ``get_matrices`` call.  This is critical when many queries touch
        overlapping row groups: sequential calls would decompress the same row group
        repeatedly, whereas this method decompresses it once and distributes the
        matrices to all queries that need them.

        Parameters
        ----------
        queries : list[dict]
            List of filter dicts, each matching the ``**filters`` signature of
            :meth:`get_matrices` (e.g.
            ``[{"method": "pyspi", "func": "cov_EmpiricalCovariance", "success": True}, ...]``).
        limit : int | None
            Maximum matrices to return per query.  *None* returns all matches.
        max_workers : int
            Number of threads to use for parallel row group reads (default 8).
            PyArrow releases the GIL so threads provide genuine parallelism here.
            Set to 1 to disable parallelism.

        Returns
        -------
        list[list[np.ndarray]]
            One list of flattened matrices per query, preserving input order.
        """
        self._build_index()
        assert self._index_df is not None

        # Step 1: Resolve row indices for every query (index lookups only — no I/O)
        query_row_indices: list[list[int]] = []
        all_needed: set[int] = set()
        for query_filters in queries:
            filtered = self._apply_filters(self._index_df, **query_filters)
            if limit is not None:
                filtered = filtered.limit(limit)
            indices = filtered.select("index").to_series().to_list()
            query_row_indices.append(indices)
            all_needed.update(indices)

        if not all_needed:
            return [[] for _ in queries]

        # Step 2: Group all needed rows by row group
        rg_rows: dict[int, list[tuple[int, int]]] = {}
        for global_idx in all_needed:
            rg_idx, local_idx = self._find_row_group(global_idx)
            rg_rows.setdefault(rg_idx, []).append((local_idx, global_idx))

        logger.info(
            "Batch fetch: %d queries → %d unique rows across %d row groups",
            len(queries),
            len(all_needed),
            len(rg_rows),
        )

        # Step 3: Read each row group exactly once (in parallel), cache all needed matrices.
        # Each thread opens its own ParquetFile handle to avoid thread-safety issues.
        parquet_path = self.parquet_path

        def _read_rg(rg_idx: int) -> dict[int, np.ndarray]:
            pf = pq.ParquetFile(parquet_path)
            table = pf.read_row_group(rg_idx, columns=["mat"])
            return {
                global_idx: np.asarray(table["mat"][local_idx].as_py())
                for local_idx, global_idx in rg_rows[rg_idx]
            }

        mat_cache: dict[int, np.ndarray] = {}
        if max_workers == 1:
            for rg_idx in sorted(rg_rows.keys()):
                mat_cache.update(_read_rg(rg_idx))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as exe:
                futures = {exe.submit(_read_rg, rg_idx): rg_idx for rg_idx in sorted(rg_rows.keys())}
                for future in concurrent.futures.as_completed(futures):
                    mat_cache.update(future.result())

        # Step 4: Distribute cached matrices back to each query
        return [
            [mat_cache[idx] for idx in indices if idx in mat_cache]
            for indices in query_row_indices
        ]


def load_symmetry_lookup(project_root: Path | None = None) -> dict:
    """Load symmetry lookup dictionary from JSON file.
    
    Returns:
        Dictionary mapping "method__func" keys to boolean symmetry values.
    """
    if project_root is None:
        project_root = Path(os.environ["PROJECT_ROOT"])
    
    lookup_path = project_root / "resources/matrix_symmetry_lookup.json"
    
    if not lookup_path.exists():
        raise FileNotFoundError(
            f"Symmetry lookup file not found: {lookup_path}\n"
            "Please run notebooks/detect_matrix_symmetry.ipynb to generate it."
        )
    
    with open(lookup_path, 'r') as f:
        symmetry_lookup = json.load(f)
    
    return symmetry_lookup


def compute_pearson_kernel(X: np.ndarray) -> np.ndarray:
    # Center each sample
    X = X - np.nanmean(X, axis=1, keepdims=True)
    # Fill NaN.
    X = np.where(np.isnan(X), 0.0, X)
    # Cosine kernel, i.e. Pearson correlation since the samples are centered.
    K = cosine_similarity(X)
    return K


def load_avg_mats(mats_dir: Path, sub_list: list[str]) -> pd.DataFrame:
    """Load average FC matrices from an FC matrix dataset for a list of subjects.

    Return array of average matrices and the run counts. Subjects with missing data are
    given all zero matrices.
    """
    mats_ds = pads.dataset(sorted(mats_dir.rglob("*.arrow")), format="arrow")
    mats_df = mats_ds.to_table().to_pandas()

    # Average across sessions/runs
    avg_mats_df = mats_df.groupby(["sub"]).agg(
        {"success": "sum", "mat": average_matrices}
    )

    mat_shape, mat_dtype = next(
        (mat.shape, mat.dtype) for mat in avg_mats_df["mat"] if mat is not None
    )

    avg_mats = []
    counts = []
    for sub in sub_list:
        if sub in avg_mats_df.index:
            mat = avg_mats_df.loc[sub, "mat"]
            if mat is None:
                mat = np.zeros(mat_shape, dtype=mat_dtype)
            count = avg_mats_df.loc[sub, "success"]
        else:
            mat = np.zeros(mat_shape, dtype=mat_dtype)
            count = 0
        avg_mats.append(mat)
        counts.append(count)

    avg_mats_df = pd.DataFrame({"Count": counts, "Matrix": avg_mats}, index=sub_list)
    return avg_mats_df


def load_avg_mats_from_parquet(
    parquet_path: Path,
    method: str,
    func: str,
    sub_list: list[str],
    lag: int = 0,
    fill_missing: bool = True,
    reader: EfficientMatrixReader | None = None,
) -> pd.DataFrame:
    """Load average FC matrices from aggregated parquet without sparsity thresholding.

    Args:
        parquet_path: Path to aggregated parquet file.
        method: Method name ("pyspi" or "skarf").
        func: Function/SPI name.
        sub_list: Subject IDs to align outputs to.
        lag: Lag value for skarf entries.
        fill_missing: Whether to fill missing matrices with zeros (matches existing loaders) or keep them as None.
        reader: Pre-built EfficientMatrixReader instance. If None, one is created
            internally. Pass a shared reader when calling in a loop to avoid
            rebuilding the index each time.

    Returns:
        DataFrame indexed by subject with columns:
            - "Count": number of successful runs used in averaging
            - "Matrix": averaged flattened matrix (or missing value)
    """
    if reader is None:
        reader = EfficientMatrixReader(parquet_path)

    filters: dict = {"method": method, "func": func, "success": True}
    if method == "skarf":
        filters["lag"] = lag

    df_pl = reader.query(columns=["mat"], **filters)

    if len(df_pl) == 0:
        return pd.DataFrame(
            {"Count": [0] * len(sub_list), "Matrix": [None] * len(sub_list)},
            index=sub_list,
        )

    df_pd = df_pl.select(["sub", "mat"]).to_pandas()

    avg_mats_df = df_pd.groupby("sub").agg({"mat": ["count", average_matrices]})
    avg_mats_df.columns = ["Count", "Matrix"]

    mat_shape = None
    mat_dtype = None
    for mat in avg_mats_df["Matrix"]:
        if mat is not None:
            mat_shape = mat.shape
            mat_dtype = mat.dtype
            break

    avg_mats = []
    counts = []
    for sub in sub_list:
        if sub in avg_mats_df.index:
            mat = avg_mats_df.loc[sub, "Matrix"]
            count = avg_mats_df.loc[sub, "Count"]
            if mat is None and fill_missing and mat_shape is not None:
                mat = np.zeros(mat_shape, dtype=mat_dtype)
        else:
            count = 0
            if fill_missing and mat_shape is not None:
                mat = np.zeros(mat_shape, dtype=mat_dtype)
            else:
                mat = None
        avg_mats.append(mat)
        counts.append(count)

    result_df = pd.DataFrame({"Count": counts, "Matrix": avg_mats}, index=sub_list)
    return result_df


def average_matrices(mats: list[np.ndarray]) -> np.ndarray:
    mats = [mat for mat in mats if mat is not None]
    if len(mats) == 0:
        return None
    return np.nanmean(np.stack(mats), axis=0)


def apply_sparsity(mat: np.ndarray, is_symmetric: bool, sparsity: float = 0.8) -> np.ndarray:
    """Apply sparsity threshold to matrix.
    
    Args:
        mat: Flattened connectivity matrix.
        is_symmetric: Whether the matrix is symmetric.
        sparsity: Sparsity level to impose (default 0.8 = keep top 20%).
    
    Returns:
        Flattened sparsity-thresholded matrix.
    """
    # Reshape flattened array to 2D matrix
    n_elements = len(mat)
    n = int(np.sqrt(n_elements))
    if n * n != n_elements:
        raise ValueError(f"Cannot reshape array of length {n_elements} to square matrix")
    
    mat = mat.reshape(n, n)
    
    if is_symmetric:
        # For symmetric matrices, threshold on upper triangle only
        # Extract upper triangle (excluding diagonal)
        triu_indices = np.triu_indices(n, k=1)
        upper_vals = mat[triu_indices]
        
        # Apply threshold to upper triangle
        threshold = np.nanpercentile(np.abs(upper_vals), sparsity * 100)
        upper_sparse = np.where(np.abs(upper_vals) >= threshold, upper_vals, 0.0)
        
        # Reconstruct symmetric matrix
        mat_sparse = np.zeros_like(mat)
        mat_sparse[triu_indices] = upper_sparse
        mat_sparse = mat_sparse + mat_sparse.T  # Mirror to lower triangle
        np.fill_diagonal(mat_sparse, mat.diagonal())  # Preserve diagonal
    else:
        # For non-symmetric matrices, threshold entire matrix
        threshold = np.nanpercentile(np.abs(mat), sparsity * 100)
        mat_sparse = np.where(np.abs(mat) >= threshold, mat, 0.0)
    
    # Flatten back to 1D for storage
    return mat_sparse.flatten()


def load_avg_mats_and_impose_sparsity(
    parquet_path: Path,
    method: str,
    func: str,
    sub_list: list[str],
    sparsity: float | None = 0.8,
    symmetry_lookup: dict | None = None,
    lag: int = 0,
    fill_missing: bool = True,
    reader: EfficientMatrixReader | None = None,
) -> pd.DataFrame:
    """Load average FC matrices from parquet file with optional sparsity thresholding.

    Delegates loading and averaging to :func:`load_avg_mats_from_parquet`, then
    optionally applies sparsity thresholding to each averaged matrix.

    Args:
        parquet_path: Path to the aggregated parquet file.
        method: Method name ("pyspi" or "skarf").
        func: Function name (e.g., "cov_EmpiricalCovariance", "linear_ridge").
        sub_list: List of subject IDs to load.
        sparsity: Sparsity level to impose (default 0.8 = keep top 20%).
            Use ``None`` to skip sparsity thresholding entirely.
        symmetry_lookup: Dictionary mapping "method__func" to boolean symmetry.
            If None, will not use symmetric matrix optimization.
        lag: Lag value for skarf methods (default 0). Only used for filtering data.
        fill_missing: Whether to fill missing matrices with zeros or keep them as None.
        reader: Pre-built EfficientMatrixReader instance. If None, one is created
            internally. Pass a shared reader when calling in a loop to avoid
            rebuilding the index each time.

    Returns:
        DataFrame with columns "Count" (number of runs) and "Matrix" (averaged
        matrices, optionally sparsity-thresholded), indexed by subject ID.
    """
    result_df = load_avg_mats_from_parquet(
        parquet_path, method, func, sub_list,
        lag=lag, fill_missing=fill_missing, reader=reader,
    )

    if sparsity is None:
        return result_df

    # Check if this method/func produces symmetric matrices
    is_symmetric = False
    if symmetry_lookup is not None:
        lookup_key = f"{method}__{func}"
        is_symmetric = symmetry_lookup.get(lookup_key, False)

    # Apply sparsity threshold to each matrix
    result_df["Matrix"] = result_df["Matrix"].apply(
        lambda mat: apply_sparsity(mat, is_symmetric, sparsity) if mat is not None else mat
    )

    return result_df
