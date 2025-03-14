# ARFC experiments Logbook

## 2025-03-13

- Set up skeleton for experiments.
- Set up uv environment.
- Added `skarf` dependency.
- Trying to update `skarf` to include the `pyspi` dependency, but `uv sync -P skarf` is getting stuck.

```
  × Failed to download `torch==2.6.0`
  ├─▶ Failed to extract archive
  ╰─▶ Failed to download distribution due to network timeout. Try increasing UV_HTTP_TIMEOUT
      (current value: 30s).
  help: `torch` (v2.6.0) was included because `arfc-experiments` (v0.1.0) depends on
        `skarf[pyspi]` (v0.1.0a1.dev1+gef2b16e) which depends on `pyspi` (v1.1.1) which depends
        on `torch`
```


## 2025-03-14

- Added `skarf` as a submodule editable dependency.
