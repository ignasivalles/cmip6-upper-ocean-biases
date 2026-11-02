"""Download the CMIP6 historical fields needed to add a model to the ensemble
(files overlapping 1970-2014), trying all replicas of each file.

Output: RAW_DIR/models/<SOURCE_ID>/<variable>/*.nc

Usage:
    python preprocessing/09_download_model.py <SOURCE_ID> <MEMBER_ID> <GRID_LABEL> [VARIABLE ...]
    python preprocessing/09_download_model.py CNRM-ESM2-1 r1i1p1f2 gn
    python preprocessing/09_download_model.py CESM2 r1i1p1f1 gn thetao so
"""

import sys
import warnings

import requests
from pyesgf.search import SearchConnection

from common.paths import RAW_DIR

warnings.filterwarnings("ignore")

FIELDS = [("Omon", "thetao"), ("Omon", "so"), ("Omon", "tos"), ("Omon", "sos"),
          ("Omon", "hfds"), ("Amon", "tauu"), ("Amon", "tauv"), ("Amon", "evspsbl"),
          ("Amon", "pr")]
ESGF_NODES = ["https://esgf-node.llnl.gov/esg-search", "https://esgf-data.dkrz.de/esg-search"]


def overlaps_period(fname, start=1970, end=2014):
    period = fname.rsplit("_", 1)[-1].replace(".nc", "")
    try:
        return int(period[7:11]) >= start and int(period[:4]) <= end
    except ValueError:
        return True


def download(url, path):
    try:
        r = requests.get(url, stream=True, timeout=(20, 300), verify=False)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=4 << 20):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"    failed: {e}")
        path.unlink(missing_ok=True)
        return False


def search(source_id, member_id, table, var, grids):
    """First dataset found, trying each grid label on each node. Some models
    put different variables on different grids."""
    for grid in grids:
        for node in ESGF_NODES:
            try:
                ctx = SearchConnection(node, distrib=False).new_context(
                    project="CMIP6", experiment_id="historical", table_id=table, variable=var,
                    source_id=source_id, member_id=member_id, grid_label=grid, latest=True,
                    facets="source_id")
                results = list(ctx.search())
            except Exception as e:
                print(f"    {node}: {e}")
                continue
            if results:
                return results
    return []


def main(source_id, member_id, grid_label, variables=None):
    out_dir = RAW_DIR / "models" / source_id
    grids = list(dict.fromkeys([grid_label, "gr", "gn", "gr1"]))

    fields = [(t, v) for t, v in FIELDS if not variables or v in variables]
    for table, var in fields:
        var_dir = out_dir / var
        if var_dir.exists() and any(var_dir.glob("*.nc")):
            continue
        results = search(source_id, member_id, table, var, grids)
        if not results:
            print(f"{var}: not found")
            continue

        replicas = {}
        for r in results:
            for f in r.file_context().search(ignore_facet_check=True):
                if f.download_url and f.filename.endswith(".nc"):
                    replicas.setdefault(f.filename, []).append(f.download_url)

        var_dir.mkdir(parents=True, exist_ok=True)
        for fname, urls in replicas.items():
            if not overlaps_period(fname) or (var_dir / fname).exists():
                continue
            print(f"{var}: {fname}")
            if not any(download(u, var_dir / fname) for u in urls):
                print(f"    all replicas failed for {fname}")

    missing = [v for _, v in fields if not any((out_dir / v).glob("*.nc"))]
    if missing:
        print(f"Missing variables (rerun to retry): {missing}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(*sys.argv[1:4], variables=sys.argv[4:])
