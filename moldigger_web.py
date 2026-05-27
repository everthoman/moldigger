#!/usr/bin/env python3
"""
MolDigger Web — Browser-based Molecular Structure Search
=========================================================
Flask web server version of the MolDigger desktop app.
Uses FPSim2 for similarity search and RDKit for substructure search.

Usage:
    python moldigger_web.py [--host 0.0.0.0] [--port 8000] [--reload]

    By default the server binds to all interfaces (0.0.0.0). To restrict to a
    specific IP or interface, pass --host, e.g.:
        python moldigger_web.py --host 192.168.1.10
"""

import sys
import os
import io
import json
import time
import pickle
import logging
import tempfile
import threading
import uuid
import concurrent.futures
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("moldigger_web")

# ── Dependency detection ──────────────────────────────────────────────────────

RDKIT_OK = False
FPSIM2_OK = False
GPU_OK = False

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem import rdSubstructLibrary
    from rdkit.DataStructs import ExplicitBitVect
    RDKIT_OK = True
except ImportError:
    pass

try:
    from FPSim2 import FPSim2Engine
    from FPSim2.io import create_db_file
    FPSIM2_OK = True
except ImportError:
    pass

try:
    from FPSim2 import FPSim2CudaEngine
    import cupy  # noqa: F401
    GPU_OK = True
except Exception:
    pass

try:
    from fastapi import FastAPI, Request, HTTPException, Query, UploadFile, File, Form
    from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
    import uvicorn
    FASTAPI_OK = True
except ImportError:
    FASTAPI_OK = False
    log.error("FastAPI/uvicorn not found — install them: pip install fastapi uvicorn")
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────

KETCHER_DIR = Path.home() / ".moldigger" / "ketcher"
JOB_TTL = 300  # seconds before completed jobs are GC'd

FP_TYPES = {
    "Morgan / ECFP4  (radius=2, 2048 bits)":       ("Morgan", {"radius": 2, "fpSize": 2048}),
    "Morgan / ECFP6  (radius=3, 2048 bits)":       ("Morgan", {"radius": 3, "fpSize": 2048}),
    "RDKit Topological  (minPath=1, maxPath=7)":   ("RDKit",  {"minPath": 1, "maxPath": 7, "fpSize": 2048}),
    "MACCS Keys  (166 bits)":                      ("MACCSKeys", {}),
    "Atom Pairs  (2048 bits)":                     ("AtomPair", {"fpSize": 2048}),
    "Topological Torsion  (2048 bits)":            ("TopologicalTorsion", {"fpSize": 2048}),
}

# Filesystem-safe short tags per FP label, used when building multi-FP sibling
# files like <base>.<tag>.h5. Must be unique and stable across versions.
_FP_TAGS = {
    "Morgan / ECFP4  (radius=2, 2048 bits)":       "morgan_ecfp4",
    "Morgan / ECFP6  (radius=3, 2048 bits)":       "morgan_ecfp6",
    "RDKit Topological  (minPath=1, maxPath=7)":   "rdkit_topological",
    "MACCS Keys  (166 bits)":                      "maccs",
    "Atom Pairs  (2048 bits)":                     "atom_pairs",
    "Topological Torsion  (2048 bits)":            "topological_torsion",
}

def _compute_fp(mol, fp_type: str, fp_params: dict):
    """Compute a bit-vector fingerprint matching the FPSim2 fp_type/fp_params convention."""
    p = fp_params or {}
    if fp_type == "Morgan":
        return AllChem.GetMorganFingerprintAsBitVect(mol, p.get("radius", 2), nBits=p.get("fpSize", 2048))
    elif fp_type == "RDKit":
        return Chem.RDKFingerprint(mol, minPath=p.get("minPath", 1), maxPath=p.get("maxPath", 7), fpSize=p.get("fpSize", 2048))
    elif fp_type == "MACCSKeys":
        return rdMolDescriptors.GetMACCSKeysFingerprint(mol)
    elif fp_type == "AtomPair":
        return rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=p.get("fpSize", 2048))
    elif fp_type == "TopologicalTorsion":
        return rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=p.get("fpSize", 2048))
    else:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


EXAMPLE_SMILES = [
    ("— Quick examples —",              ""),
    ("Benzene",                          "c1ccccc1"),
    ("Aspirin",                          "CC(=O)Oc1ccccc1C(=O)O"),
    ("Caffeine",                         "Cn1cnc2c1c(=O)n(c(=O)n2C)C"),
    ("Ibuprofen",                        "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("Sildenafil",                       "CCCC1=NN(C2=CC(=C(C=C2)S(=O)(=O)N3CCN(CC3)C)OCC)C(=O)C4=C1NC(=NC4=O)C"),
]

# ── App state ─────────────────────────────────────────────────────────────────

_state_lock = threading.Lock()
_state = {
    "engine": None,
    "mol_map": {},   # {str(mol_id): {"smiles": ..., "name": ...}}
    "db_path": None,
    "set_path": None,   # set to the .fpset directory path when loaded via alias
    "fp_name": None,
    "mol_count": 0,
    # Multi-FP set: when a DB is built with sibling FP files, these map
    # FP display-name -> open FPSim2 engine and FP display-name -> .h5 path.
    # For a single-FP DB both contain exactly one entry.
    "fp_engines": {},
    "fp_files": {},
    # Substructure search uses RDKit's rdSubstructLibrary (C++, multi-threaded):
    #   sub_library     -> rdSubstructLibrary.SubstructLibrary
    #   sub_ids[i]      -> mol_id key into mol_map, for library index i
    "sub_library": None,
    "sub_ids": [],
}

_jobs = {}
_jobs_lock = threading.Lock()

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="MolDigger Web")

# ── Job system ────────────────────────────────────────────────────────────────

def new_job() -> str:
    jid = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[jid] = {
            "id": jid,
            "status": "running",
            "progress": [],
            "result": None,
            "error": None,
            "created": time.time(),
            "finished_at": None,
        }
    return jid


def update_job_progress(jid: str, msg: str):
    with _jobs_lock:
        if jid in _jobs:
            _jobs[jid]["progress"].append(msg)
            log.info(f"[job {jid[:8]}] {msg}")


def finish_job(jid: str, result=None, error=None):
    with _jobs_lock:
        if jid in _jobs:
            _jobs[jid]["status"] = "error" if error else "done"
            _jobs[jid]["result"] = result
            _jobs[jid]["error"] = error
            _jobs[jid]["finished_at"] = time.time()


def get_job(jid: str) -> dict | None:
    with _jobs_lock:
        job = _jobs.get(jid)
        if job is None:
            return None
        return dict(job)  # shallow copy


def cleanup_jobs():
    now = time.time()
    with _jobs_lock:
        to_delete = [
            jid for jid, j in _jobs.items()
            if j["status"] in ("done", "error")
            and j.get("finished_at") is not None
            and (now - j["finished_at"]) > JOB_TTL
        ]
        for jid in to_delete:
            del _jobs[jid]

# ── Molecule helpers ──────────────────────────────────────────────────────────

def smiles_to_svg(smiles: str, width: int = 200, height: int = 150,
                  highlight_atoms=None, highlight_bonds=None,
                  query_mol=None, do_highlight: bool = True) -> str:
    """Render SMILES to an SVG string (no XML header). Returns placeholder on failure.

    highlight_atoms / highlight_bonds : explicit atom/bond index lists (substructure mode)
    query_mol      : RDKit Mol — MCS is computed against this mol (similarity mode)
    do_highlight   : set False to skip all highlighting
    """
    if not RDKIT_OK or not smiles:
        return _placeholder_svg(width, height)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return _placeholder_svg(width, height)
        AllChem.Compute2DCoords(mol)

        h_atoms, h_bonds = [], []

        if do_highlight:
            if highlight_atoms:
                h_atoms = list(highlight_atoms)
            elif query_mol is not None:
                # MCS highlight for similarity results
                try:
                    from rdkit.Chem import rdFMCS
                    res = rdFMCS.FindMCS(
                        [query_mol, mol],
                        timeout=1,
                        ringMatchesRingOnly=False,
                        completeRingsOnly=False,
                    )
                    if res.numAtoms > 1:
                        patt = Chem.MolFromSmarts(res.smartsString)
                        match = mol.GetSubstructMatch(patt)
                        if match:
                            h_atoms = list(match)
                except Exception:
                    pass

            if h_atoms:
                match_set = set(h_atoms)
                h_bonds = [
                    b.GetIdx() for b in mol.GetBonds()
                    if b.GetBeginAtomIdx() in match_set and b.GetEndAtomIdx() in match_set
                ]

        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.drawOptions().addStereoAnnotation = True
        if h_atoms:
            drawer.DrawMolecule(mol, highlightAtoms=h_atoms, highlightBonds=h_bonds)
        else:
            drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        # Strip XML declaration for inline embedding
        if svg.startswith("<?xml"):
            svg = svg[svg.index("<svg"):]
        return svg
    except Exception:
        return _placeholder_svg(width, height)


def _placeholder_svg(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<rect width="{width}" height="{height}" fill="#f8f9fa" stroke="#dee2e6"/>'
        f'<text x="{width//2}" y="{height//2}" text-anchor="middle" '
        f'dominant-baseline="middle" fill="#adb5bd" font-size="12">No structure</text>'
        f'</svg>'
    )


def assign_cluster_ids_from_smiles(smiles_list: list, cutoff: float = 0.4) -> list:
    """Butina-cluster a list of SMILES. Returns list of cluster_id (int|None)."""
    dummy_rows = [{"smiles": s} for s in smiles_list]
    assign_cluster_ids(dummy_rows, cutoff=cutoff)
    return [r["cluster_id"] for r in dummy_rows]


def assign_cluster_ids(rows: list, cutoff: float = 0.4) -> None:
    """Butina-cluster rows in-place using Morgan ECFP4 (best general-purpose clustering FP).
    cutoff is a similarity threshold. Adds 'cluster_id' (int) to each row dict."""
    if not RDKIT_OK or not rows:
        for r in rows:
            r["cluster_id"] = None
        return
    try:
        from rdkit.ML.Cluster import Butina
        from rdkit import DataStructs

        smiles_list = [r["smiles"] for r in rows]
        fps, valid_idx = [], []
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol:
                fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
                valid_idx.append(i)

        n = len(fps)
        dists = []
        for i in range(1, n):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend(1.0 - s for s in sims)

        clusters = Butina.ClusterData(dists, n, 1.0 - cutoff, isDistData=True)

        pos_to_cid = {}   # position-in-fps → cluster_id
        for cid, members in enumerate(clusters, start=1):
            for pos in members:
                pos_to_cid[pos] = cid

        orig_to_pos = {orig: pos for pos, orig in enumerate(valid_idx)}
        for i, row in enumerate(rows):
            pos = orig_to_pos.get(i)
            row["cluster_id"] = pos_to_cid.get(pos) if pos is not None else None
    except Exception:
        log.exception("Clustering failed")
        for r in rows:
            r["cluster_id"] = None


def compute_props(smiles: str) -> dict:
    """Return {"mw": float|None, "clogp": float|None}."""
    if not RDKIT_OK or not smiles:
        return {"mw": None, "clogp": None}
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"mw": None, "clogp": None}
        return {
            "mw": round(Descriptors.MolWt(mol), 2),
            "clogp": round(Descriptors.MolLogP(mol), 2),
        }
    except Exception:
        return {"mw": None, "clogp": None}


def _fp_name_from_engine(engine) -> str:
    """Infer display name from engine fp_type and fp_params. Morgan variants
    (ECFP4 vs ECFP6) share fp_type='Morgan' and differ only by radius."""
    try:
        fp_type = engine.fp_type
        fp_params = engine.fp_params or {}
    except AttributeError:
        return "Unknown"
    if fp_type == "Morgan":
        radius = fp_params.get("radius", 2)
        for name, (t, p) in FP_TYPES.items():
            if t == "Morgan" and p.get("radius", 2) == radius:
                return name
        return "Morgan"
    for name, (t, _p) in FP_TYPES.items():
        if t == fp_type:
            return name
    return fp_type

# ── Result builder ────────────────────────────────────────────────────────────

def _build_result_rows(hits, mol_map: dict, match_atoms: dict | None = None,
                       query_mol=None, do_highlight: bool = True) -> list:
    rows = []
    for mol_id, score in hits:
        key = str(mol_id)
        entry = mol_map.get(key) or mol_map.get(int(mol_id) if isinstance(mol_id, str) else mol_id)
        if entry is None:
            continue
        smiles = entry.get("smiles", "") if isinstance(entry, dict) else str(entry)
        name = entry.get("name", key) if isinstance(entry, dict) else key
        atoms = None
        if match_atoms:
            atoms = match_atoms.get(int(mol_id)) or match_atoms.get(str(mol_id))
        svg = smiles_to_svg(smiles, width=150, height=110,
                            highlight_atoms=atoms, query_mol=query_mol,
                            do_highlight=do_highlight)
        props = compute_props(smiles)
        rows.append({
            "mol_id": int(mol_id),
            "name": name,
            "smiles": smiles,
            "score": float(score),
            "svg": svg,
            "mw": props["mw"],
            "clogp": props["clogp"],
        })
    return rows

# ── Lists store ───────────────────────────────────────────────────────────────
#
# Server-global file at ~/.moldigger/lists.json keyed by absolute DB path:
#   {
#     "/abs/path/to/db.h5": {
#       "list_name": {"created": "ISO-8601", "ids": [int, int, ...]},
#       ...
#     },
#     ...
#   }
# IDs are mol_id integers (FPSim2 sequential IDs from the .h5).

LISTS_DIR = Path.home() / ".moldigger"
LISTS_PATH = LISTS_DIR / "lists.json"
STATE_PATH = LISTS_DIR / "state.json"
_lists_lock = threading.Lock()


def _persist_last_db_path(path: str) -> None:
    """Remember the most recently loaded DB path so we can restore on startup."""
    try:
        LISTS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = STATE_PATH.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"last_db_path": str(Path(path).resolve())}, fh)
        os.replace(tmp, STATE_PATH)
    except Exception as exc:
        log.warning(f"Could not persist last_db_path: {exc}")


def _read_last_db_path() -> str | None:
    if not STATE_PATH.exists():
        return None
    try:
        with open(STATE_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        p = data.get("last_db_path")
        return p if p and Path(p).exists() else None
    except Exception:
        return None


def _lists_load_all() -> dict:
    if not LISTS_PATH.exists():
        return {}
    try:
        with open(LISTS_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return {}
        return data
    except Exception as exc:
        log.warning(f"Could not read lists file, treating as empty: {exc}")
        return {}


def _lists_save_all(data: dict) -> None:
    LISTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = LISTS_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, LISTS_PATH)


def _lists_db_key() -> str | None:
    """Absolute path of the currently-loaded DB, used as the key in lists.json."""
    with _state_lock:
        path = _state.get("db_path")
    if not path:
        return None
    return str(Path(path).resolve())


def _lists_for_current_db() -> dict:
    """Read-only view of lists for the loaded DB, or {} if none / no DB loaded."""
    key = _lists_db_key()
    if not key:
        return {}
    with _lists_lock:
        return dict(_lists_load_all().get(key, {}))


def _lists_write_one(name: str, ids: list, overwrite: bool) -> tuple[bool, str]:
    """Persist a single list under the current DB. Returns (ok, message)."""
    key = _lists_db_key()
    if not key:
        return False, "No database loaded."
    name = (name or "").strip()
    if not name:
        return False, "List name is required."
    if any(c in name for c in "/\\\n\r\t"):
        return False, "List name contains invalid characters."
    clean_ids = sorted({int(i) for i in ids if i is not None})
    with _lists_lock:
        data = _lists_load_all()
        bucket = data.setdefault(key, {})
        if name in bucket and not overwrite:
            return False, f"List '{name}' already exists."
        bucket[name] = {
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "ids": clean_ids,
        }
        _lists_save_all(data)
    return True, f"Saved list '{name}' ({len(clean_ids):,} molecules)."


def _lists_delete_one(name: str) -> tuple[bool, str]:
    key = _lists_db_key()
    if not key:
        return False, "No database loaded."
    with _lists_lock:
        data = _lists_load_all()
        bucket = data.get(key, {})
        if name not in bucket:
            return False, f"List '{name}' not found."
        del bucket[name]
        if not bucket:
            data.pop(key, None)
        _lists_save_all(data)
    return True, f"Deleted list '{name}'."


def _lists_resolve_identifiers(names_list: list) -> tuple[list, list]:
    """Resolve a list of molecule identifiers (name strings) to mol_ids in
    the loaded DB. Lookup is exact-match on the 'name' field stored in
    mol_map. Returns (resolved_ids, unresolved_names)."""
    with _state_lock:
        mol_map = dict(_state["mol_map"])
    if not mol_map:
        return [], list(names_list)
    idx: dict[str, int] = {}
    for mid_str, entry in mol_map.items():
        name = entry.get("name", "") if isinstance(entry, dict) else ""
        if not name:
            continue
        idx.setdefault(str(name), int(mid_str))
    resolved, missing = [], []
    for raw in names_list:
        s = (raw or "").strip()
        if not s:
            continue
        hit = idx.get(s)
        if hit is None:
            missing.append(s)
        else:
            resolved.append(hit)
    return resolved, missing


def _lists_resolve_smiles(smiles_list: list) -> tuple[list, list]:
    """Resolve a list of SMILES strings to mol_ids in the loaded DB.

    Returns (resolved_ids, unresolved_smiles). Matching is by canonical SMILES;
    builds a one-shot SMILES -> mol_id index from mol_map.
    """
    if not RDKIT_OK:
        return [], list(smiles_list)
    with _state_lock:
        mol_map = dict(_state["mol_map"])
    if not mol_map:
        return [], list(smiles_list)
    # Build canonical SMILES index (mol_map stores already-canonical SMILES from
    # build_db, but we re-canonicalize defensively).
    idx = {}
    for mid_str, entry in mol_map.items():
        smi = entry.get("smiles", "") if isinstance(entry, dict) else str(entry)
        if not smi:
            continue
        idx.setdefault(smi, int(mid_str))
    resolved, missing = [], []
    for raw in smiles_list:
        s = (raw or "").strip()
        if not s:
            continue
        # Try as-is, then canonicalized.
        hit = idx.get(s)
        if hit is None:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                try:
                    canon = Chem.MolToSmiles(mol, canonical=True)
                    hit = idx.get(canon)
                except Exception:
                    pass
        if hit is None:
            missing.append(s)
        else:
            resolved.append(hit)
    return resolved, missing


def _lists_universe() -> set:
    """All mol_ids in the loaded DB. Used for NOT semantics."""
    with _state_lock:
        mol_map = dict(_state["mol_map"])
    return {int(k) for k in mol_map.keys()}


# ── Substructure library cache ────────────────────────────────────────────────
#
# Substructure search is powered by RDKit's rdSubstructLibrary.SubstructLibrary
# — a C++ multi-threaded engine with built-in pattern-FP screening. The library
# is built once at DB load, persisted next to the .h5 file, and reloaded on
# subsequent loads of the same DB.

_SUBCACHE_VERSION = 2  # bumped: format changed from pickle to library serialization


def _subcache_path(db_path: str) -> str:
    return db_path + ".subcache.bin"


def _source_mtimes(db_path: str) -> tuple:
    """Return mtimes of files that, if changed, should invalidate the cache."""
    db_mt = os.path.getmtime(db_path) if os.path.exists(db_path) else 0.0
    companion = db_path + ".smiles.json"
    cm_mt = os.path.getmtime(companion) if os.path.exists(companion) else 0.0
    return (db_mt, cm_mt)


def _build_substructure_library(mol_map: dict):
    """Build a SubstructLibrary + parallel ids list from the loaded DB.

    Uses CachedTrustedSmilesMolHolder (skips sanitization on retrieval, the
    SMILES from our build_db pipeline are already canonical & trusted) plus
    PatternHolder for built-in pattern-FP screening.

    Returns (library, ids). Entries that fail to parse are dropped.
    """
    if not RDKIT_OK or not mol_map:
        return None, []
    lib = rdSubstructLibrary.SubstructLibrary(
        rdSubstructLibrary.CachedTrustedSmilesMolHolder(),
        rdSubstructLibrary.PatternHolder(),
    )
    ids = []
    for mid, entry in mol_map.items():
        smi = entry.get("smiles", "") if isinstance(entry, dict) else str(entry)
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            lib.AddMol(mol)
        except Exception:
            continue
        ids.append(int(mid))
    return lib, ids


def _save_substructure_cache(db_path: str, library, ids) -> int:
    """Persist library + ids to disk next to the .h5. Returns bytes written, or 0 on failure."""
    if not RDKIT_OK or library is None or not ids:
        return 0
    cache_path = _subcache_path(db_path)
    try:
        lib_blob = library.Serialize()
        header = {
            "version": _SUBCACHE_VERSION,
            "source_mtimes": _source_mtimes(db_path),
            "ids": ids,
        }
        header_blob = pickle.dumps(header, protocol=pickle.HIGHEST_PROTOCOL)
        tmp = cache_path + ".tmp"
        with open(tmp, "wb") as fh:
            # Layout: [4 bytes BE: header length][header pickle][library blob]
            fh.write(len(header_blob).to_bytes(4, "big"))
            fh.write(header_blob)
            fh.write(lib_blob)
        os.replace(tmp, cache_path)
        return os.path.getsize(cache_path)
    except Exception as exc:
        log.warning(f"Could not save substructure cache: {exc}")
        try:
            if os.path.exists(cache_path + ".tmp"):
                os.remove(cache_path + ".tmp")
        except Exception:
            pass
        return 0


def _load_substructure_cache(db_path: str):
    """Try to load a valid cache. Returns (library, ids) or None if missing/stale."""
    if not RDKIT_OK:
        return None
    cache_path = _subcache_path(db_path)
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "rb") as fh:
            hdr_len = int.from_bytes(fh.read(4), "big")
            header = pickle.loads(fh.read(hdr_len))
            lib_blob = fh.read()
    except Exception as exc:
        log.warning(f"Substructure cache unreadable, will rebuild: {exc}")
        return None
    if header.get("version") != _SUBCACHE_VERSION:
        log.info("Substructure cache version mismatch — rebuilding.")
        return None
    if tuple(header.get("source_mtimes", ())) != _source_mtimes(db_path):
        log.info("Substructure cache stale (source mtime changed) — rebuilding.")
        return None
    try:
        lib = rdSubstructLibrary.SubstructLibrary(lib_blob)
        ids = header["ids"]
    except Exception as exc:
        log.warning(f"Substructure cache deserialize failed, will rebuild: {exc}")
        return None
    return lib, ids


# ── Search runners ────────────────────────────────────────────────────────────

def _run_similarity_search(jid: str, smiles: str, threshold: float, threshold_max: float,
                           n_workers: int, use_gpu: bool, metric: str,
                           tversky_a: float, tversky_b: float, max_results: int,
                           do_highlight: bool = True):
    try:
        update_job_progress(jid, "Starting similarity search…")
        with _state_lock:
            engine = _state["engine"]
            mol_map = _state["mol_map"]

        if engine is None:
            finish_job(jid, error="No database loaded.")
            return

        t0 = time.perf_counter()
        use_gpu_actual = use_gpu and metric == "tanimoto" and GPU_OK

        update_job_progress(jid, f"Searching (metric={metric}, threshold={threshold})…")
        try:
            if metric == "tversky":
                results = engine.tversky(
                    smiles, threshold, tversky_a, tversky_b,
                    n_workers=n_workers
                )
            elif use_gpu_actual:
                results = engine.similarity(smiles, threshold)
            else:
                results = engine.similarity(
                    smiles, threshold, metric=metric, n_workers=n_workers
                )
        except Exception as e:
            finish_job(jid, error=f"Search error: {e}")
            return

        elapsed = time.perf_counter() - t0
        update_job_progress(jid, f"Found {len(results):,} hits in {elapsed:.3f}s, building results…")

        hits = sorted(results, key=lambda x: x[1], reverse=True)
        if threshold_max < 1.0:
            hits = [h for h in hits if float(h[1]) <= threshold_max]
        if max_results > 0:
            hits = hits[:max_results]

        query_mol = Chem.MolFromSmiles(smiles) if (RDKIT_OK and do_highlight) else None
        rows = _build_result_rows(hits, mol_map, query_mol=query_mol, do_highlight=do_highlight)
        for r in rows:
            r["cluster_id"] = None
        finish_job(jid, result={"rows": rows, "total": len(results), "elapsed": elapsed})

    except Exception as exc:
        log.exception("Similarity search failed")
        finish_job(jid, error=str(exc))


def _run_substructure_search(jid: str, query: str, n_workers: int, max_results: int,
                             do_highlight: bool = True):
    try:
        update_job_progress(jid, "Starting substructure search…")
        with _state_lock:
            mol_map = dict(_state["mol_map"])
            sub_library = _state["sub_library"]
            sub_ids = _state["sub_ids"]

        if not mol_map:
            finish_job(jid, error="No database loaded.")
            return

        if not RDKIT_OK:
            finish_job(jid, error="RDKit not available for substructure search.")
            return

        if sub_library is None or not sub_ids:
            finish_job(jid, error="Substructure library not built. Reload the database.")
            return

        # Try SMILES first so aromaticity is perceived; fall back to SMARTS
        # for richer queries (atom lists, recursive SMARTS, etc.).
        # SMARTS-first is wrong for Kekulé SMILES queries: MolFromSmarts
        # takes atoms/bonds literally and won't match aromatic-perceived
        # database molecules.
        q = Chem.MolFromSmiles(query)
        if q is None:
            q = Chem.MolFromSmarts(query)
        if q is None:
            finish_job(jid, error="Could not parse query as SMILES or SMARTS.")
            return

        t0 = time.perf_counter()
        update_job_progress(jid, f"Searching {len(sub_ids):,} molecules…")

        # SubstructLibrary handles the FP screen + matching in C++, multi-threaded.
        # numThreads=-1 uses all available cores. maxResults caps the search early.
        cap = max_results if max_results > 0 else -1
        match_indices = sub_library.GetMatches(q, numThreads=-1, maxResults=cap)
        elapsed = time.perf_counter() - t0

        update_job_progress(
            jid,
            f"Found {len(match_indices):,} hits in {elapsed:.3f}s, building results…",
        )

        # Compute atom-level matches for the (few) hits, for highlighting.
        match_atoms = {}
        all_results = []
        for idx in match_indices:
            mid = sub_ids[idx]
            all_results.append((mid, 1.0))
            if do_highlight:
                try:
                    mol = sub_library.GetMol(idx)
                    atoms = mol.GetSubstructMatch(q)
                    if atoms:
                        match_atoms[mid] = list(atoms)
                except Exception:
                    pass

        rows = _build_result_rows(hits=all_results, mol_map=mol_map,
                                  match_atoms=match_atoms, do_highlight=do_highlight)
        for r in rows:
            r["cluster_id"] = None
        finish_job(jid, result={"rows": rows, "total": len(all_results), "elapsed": elapsed})

    except Exception as exc:
        log.exception("Substructure search failed")
        finish_job(jid, error=str(exc))


def _run_build_db(jid: str, input_path: str, output_path: str,
                  mol_format: str, fp_specs: list, name_prop: str):
    """Build one or more FPSim2 .h5 fingerprint databases from a source file.

    fp_specs: list of (fp_label, fp_type, fp_params). When the list contains
    more than one entry, sibling files are written as <base>.<tag>.h5 (one per
    FP), and each companion .smiles.json carries a "__meta__" → "siblings"
    list so the loader can discover the set."""
    tmp_path = None
    try:
        if not RDKIT_OK:
            finish_job(jid, error="RDKit is not installed.")
            return
        if not FPSIM2_OK:
            finish_job(jid, error="FPSim2 is not installed.")
            return
        if not fp_specs:
            finish_job(jid, error="At least one fingerprint type must be selected.")
            return

        update_job_progress(jid, "Reading molecules…")
        entries = []
        seq = 1

        if mol_format == "sdf":
            suppl = Chem.SDMolSupplier(input_path, sanitize=True, removeHs=True)
            for mol in suppl:
                if mol is None:
                    continue
                try:
                    smi = Chem.MolToSmiles(mol, canonical=True)
                except Exception:
                    continue
                name = ""
                if name_prop == "_Name":
                    name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
                elif name_prop and mol.HasProp(name_prop):
                    name = str(mol.GetProp(name_prop)).strip()
                if not name:
                    name = str(seq)
                entries.append((seq, smi, name))
                seq += 1
        else:
            with open(input_path, encoding="utf-8") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw or raw.startswith("#"):
                        continue
                    parts = raw.split()
                    if not parts:
                        continue
                    mol = Chem.MolFromSmiles(parts[0])
                    if mol is None:
                        continue
                    smi = Chem.MolToSmiles(mol, canonical=True)
                    name = parts[1] if len(parts) > 1 else str(seq)
                    entries.append((seq, smi, name))
                    seq += 1

        if not entries:
            finish_job(jid, error="No valid molecules found in the source file.")
            return

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".smi", delete=False, encoding="utf-8"
        ) as tmp:
            for seq_id, smi, _ in entries:
                tmp.write(f"{smi}\t{seq_id}\n")
            tmp_path = tmp.name

        multi = len(fp_specs) > 1
        out_p = Path(output_path)
        # For a single-FP build the user's exact filename is preserved
        # (legacy behavior). For a multi-FP build, the output becomes
        # <base>.fpset/<base>.<tag>.h5 so the alias is the directory.
        if out_p.suffix == ".h5":
            base_stem = out_p.with_suffix("").name
            parent_dir = out_p.parent
        else:
            base_stem = out_p.name
            parent_dir = out_p.parent

        if multi:
            set_dir = parent_dir / f"{base_stem}.fpset"
            set_dir.mkdir(parents=True, exist_ok=True)
        else:
            set_dir = None

        output_files = []  # (fp_label, full_path)
        for label, _ft, _fp in fp_specs:
            if multi:
                tag = _FP_TAGS.get(label, "fp")
                out_path = str(set_dir / f"{base_stem}.{tag}.h5")
            else:
                out_path = output_path
            output_files.append((label, out_path))

        sibling_filenames = [Path(p).name for _, p in output_files] if multi else []

        mol_map = {seq_id: {"smiles": smi, "name": name}
                   for seq_id, smi, name in entries}

        for idx, ((label, out_path), (_lbl, fp_type, fp_params)) in enumerate(zip(output_files, fp_specs), start=1):
            update_job_progress(
                jid,
                f"Writing fingerprint database {idx}/{len(output_files)}: {label}…"
            )
            create_db_file(tmp_path, out_path, "smi", fp_type, fp_params)

            companion = out_path + ".smiles.json"
            payload: dict = dict(mol_map)
            if multi:
                payload["__meta__"] = {"siblings": sibling_filenames}
            with open(companion, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)

        if multi:
            primary_filename = Path(output_files[0][1]).name
            manifest = {
                "version": 1,
                "name": base_stem,
                "primary": primary_filename,
                "variants": [
                    {"fp": label, "file": Path(p).name}
                    for label, p in output_files
                ],
            }
            with open(set_dir / "manifest.json", "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2)
            primary_path = str(set_dir)
        else:
            primary_path = output_files[0][1]

        update_job_progress(jid, f"Done — {len(entries):,} molecules indexed.")
        finish_job(jid, result={
            "path": primary_path,
            "count": len(entries),
            "files": [p for _, p in output_files],
        })

    except Exception as exc:
        log.exception("DB build failed")
        finish_job(jid, error=str(exc))
    finally:
        if tmp_path and Path(tmp_path).exists():
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

# ── Pydantic request models ────────────────────────────────────────────────────

from pydantic import BaseModel

class LoadDbRequest(BaseModel):
    path: str

class SwitchFpRequest(BaseModel):
    fp_name: str

class SearchRequest(BaseModel):
    smiles: str
    search_type: str = "similarity"
    fp: str = ""
    metric: str = "tanimoto"
    threshold: float = 0.7
    threshold_max: float = 1.0
    tversky_a: float = 0.5
    tversky_b: float = 0.5
    n_workers: int = 4
    use_gpu: bool = False
    max_results: int = 200
    highlight: bool = True

# ── FastAPI routes ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_TEMPLATE


@app.get("/ketcher/{path:path}")
def ketcher_static(path: str = ""):
    target = path if path else "index.html"
    full = KETCHER_DIR / target
    if not full.exists() or not full.is_file():
        raise HTTPException(status_code=404, detail="Ketcher file not found")
    return FileResponse(str(full))


@app.get("/api/status")
def api_status():
    cleanup_jobs()
    with _state_lock:
        return {
            "rdkit": RDKIT_OK,
            "fpsim2": FPSIM2_OK,
            "gpu": GPU_OK,
            "ketcher": (KETCHER_DIR / "index.html").exists(),
            "db_path": _state["db_path"],
            "set_path": _state.get("set_path"),
            "fp_name": _state["fp_name"],
            "fp_variants": list((_state.get("fp_engines") or {}).keys()),
            "mol_count": _state["mol_count"],
        }


def _perform_load_db(path: str) -> dict:
    """Load a DB into _state, including substructure cache. Returns result dict
    on success or raises. Used by both the /api/load_db endpoint and the
    startup auto-loader.
    """
    path = os.path.expanduser((path or "").strip())
    if not path:
        raise ValueError("path is required")
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not FPSIM2_OK:
        raise RuntimeError("FPSim2 is not installed")

    # `.fpset` directories are aliases — resolve to the primary sibling .h5.
    set_path: str | None = None
    p = Path(path)
    if p.is_dir() and p.name.endswith(".fpset"):
        manifest_path = p / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Set manifest missing: {manifest_path}")
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        primary = manifest.get("primary")
        if not primary:
            raise ValueError(f"Set manifest has no 'primary': {manifest_path}")
        set_path = str(p)
        path = str(p / primary)
        if not Path(path).exists():
            raise FileNotFoundError(f"Primary sibling missing inside set: {path}")

    engine = FPSim2Engine(path)
    companion = path + ".smiles.json"
    mol_map = {}
    sibling_filenames: list = []
    if Path(companion).exists():
        with open(companion, encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, dict):
            meta = raw.pop("__meta__", None) or {}
            siblings_field = meta.get("siblings")
            if isinstance(siblings_field, list):
                sibling_filenames = [str(s) for s in siblings_field]
        mol_map = {str(k): v for k, v in raw.items()}

    fp_name = _fp_name_from_engine(engine)

    fp_engines = {fp_name: engine}
    fp_files = {fp_name: path}
    if sibling_filenames:
        db_dir = Path(path).parent
        loaded_name = Path(path).name
        for sib_name in sibling_filenames:
            if sib_name == loaded_name:
                continue
            sib_path = str(db_dir / sib_name)
            if not Path(sib_path).exists():
                log.warning(f"Sibling FP file missing: {sib_path}")
                continue
            try:
                sib_engine = FPSim2Engine(sib_path)
            except Exception as exc:
                log.warning(f"Could not open sibling {sib_path}: {exc}")
                continue
            sib_fp = _fp_name_from_engine(sib_engine)
            if sib_fp in fp_engines:
                log.warning(f"Duplicate FP type '{sib_fp}' in sibling set; ignoring {sib_path}")
                continue
            fp_engines[sib_fp] = sib_engine
            fp_files[sib_fp] = sib_path

    sub_cache_source = "disk"
    sub_cache_bytes = 0
    t0 = time.perf_counter()
    cached = _load_substructure_cache(path)
    if cached is not None:
        sub_library, sub_ids = cached
    else:
        sub_cache_source = "build"
        sub_library, sub_ids = _build_substructure_library(mol_map)
        sub_cache_bytes = _save_substructure_cache(path, sub_library, sub_ids)
    sub_build_s = time.perf_counter() - t0
    log.info(
        f"Substructure library: {len(sub_ids):,}/{len(mol_map):,} mols "
        f"({sub_cache_source}) in {sub_build_s:.1f}s"
        + (f", saved {sub_cache_bytes / 1e6:.1f} MB" if sub_cache_bytes else "")
    )

    with _state_lock:
        _state["engine"] = engine
        _state["mol_map"] = mol_map
        _state["db_path"] = path
        _state["set_path"] = set_path
        _state["fp_name"] = fp_name
        _state["mol_count"] = len(mol_map)
        _state["fp_engines"] = fp_engines
        _state["fp_files"] = fp_files
        _state["sub_library"] = sub_library
        _state["sub_ids"] = sub_ids

    # Persist the alias path when loaded from a set — restart preserves the
    # alias view; the primary sibling is rediscovered on next load.
    _persist_last_db_path(set_path or path)

    return {
        "ok": True,
        "mol_count": len(mol_map),
        "fp_name": fp_name,
        "fp_variants": list(fp_engines.keys()),
        "path": path,
        "set_path": set_path,
        "sub_cache_count": len(sub_ids),
        "sub_cache_build_s": round(sub_build_s, 2),
        "sub_cache_source": sub_cache_source,
    }


@app.post("/api/load_db")
def api_load_db(req: LoadDbRequest):
    try:
        return _perform_load_db(req.path)
    except FileNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except (ValueError, RuntimeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        log.exception("load_db failed")
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/switch_fp")
def api_switch_fp(req: SwitchFpRequest):
    """Switch the active similarity-search engine to a different FP variant
    already loaded as a sibling. Substructure cache and mol_map are shared
    across siblings, so this is a near-instant pointer swap."""
    with _state_lock:
        engines = _state.get("fp_engines") or {}
        files = _state.get("fp_files") or {}
        if req.fp_name not in engines:
            return JSONResponse(
                {"error": f"FP '{req.fp_name}' not available. Loaded: {list(engines.keys())}"},
                status_code=400,
            )
        _state["engine"] = engines[req.fp_name]
        _state["fp_name"] = req.fp_name
        _state["db_path"] = files[req.fp_name]
        new_path = _state["db_path"]
        set_path = _state.get("set_path")
    # When loaded as a set, persist the alias path so restart restores it.
    _persist_last_db_path(set_path or new_path)
    return {"ok": True, "fp_name": req.fp_name, "path": new_path, "set_path": set_path}


@app.on_event("startup")
def _autoload_last_db():
    """If a DB was loaded in a previous run, restore it in a background thread."""
    last = _read_last_db_path()
    if not last:
        return

    def _do():
        try:
            log.info(f"Auto-loading last DB: {last}")
            _perform_load_db(last)
            log.info("Auto-load complete.")
        except Exception as exc:
            log.warning(f"Auto-load failed for {last}: {exc}")

    threading.Thread(target=_do, daemon=True).start()


@app.post("/api/search")
def api_search(req: SearchRequest):
    smiles = req.smiles.strip()
    if not smiles:
        return JSONResponse({"error": "smiles is required"}, status_code=400)

    fp_name = req.fp or list(FP_TYPES.keys())[0]
    fp_type, fp_params = FP_TYPES.get(fp_name, ("Morgan", {"radius": 2, "fpSize": 2048}))

    if req.search_type == "substructure":
        if not RDKIT_OK:
            return JSONResponse({"error": "RDKit not available"}, status_code=500)
        jid = new_job()
        threading.Thread(
            target=_run_substructure_search,
            args=(jid, smiles, req.n_workers, req.max_results, req.highlight),
            daemon=True,
        ).start()
    else:
        if not FPSIM2_OK:
            return JSONResponse({"error": "FPSim2 not available"}, status_code=500)
        jid = new_job()
        threading.Thread(
            target=_run_similarity_search,
            args=(jid, smiles, req.threshold, req.threshold_max, req.n_workers, req.use_gpu,
                  req.metric, req.tversky_a, req.tversky_b, req.max_results, req.highlight),
            daemon=True,
        ).start()

    return {"job_id": jid}


@app.get("/api/jobs/{jid}")
def api_get_job(jid: str):
    job = get_job(jid)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return job


@app.post("/api/build_db")
async def api_build_db(request: Request):
    content_type = request.headers.get("content-type", "")
    input_path = None
    tmp_upload = None

    if "multipart" in content_type or "form" in content_type:
        form = await request.form()
        output_path = (form.get("output_path") or "").strip()
        fp_labels_raw = form.getlist("fp_labels") if hasattr(form, "getlist") else []
        if not fp_labels_raw and form.get("fp"):
            fp_labels_raw = [form.get("fp")]
        mol_format = form.get("format") or "sdf"
        name_prop = form.get("name_prop") or ""
        upload: UploadFile | None = form.get("file")
        if upload and upload.filename:
            suffix = ".sdf" if mol_format == "sdf" else ".smi"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            contents = await upload.read()
            tmp.write(contents)
            tmp.close()
            input_path = tmp.name
            tmp_upload = tmp.name
        else:
            input_path = os.path.expanduser((form.get("input_path") or "").strip())
    else:
        data = await request.json()
        input_path = os.path.expanduser((data.get("input_path") or "").strip())
        output_path = os.path.expanduser((data.get("output_path") or "").strip())
        fp_labels_raw = data.get("fp_labels") or ([data["fp"]] if data.get("fp") else [])
        mol_format = data.get("format") or "sdf"
        name_prop = data.get("name_prop") or ""

    if not input_path:
        return JSONResponse({"error": "input_path or file upload required"}, status_code=400)
    if not output_path:
        return JSONResponse({"error": "output_path is required"}, status_code=400)
    if not Path(input_path).exists():
        return JSONResponse({"error": f"Input file not found: {input_path}"}, status_code=404)

    fp_labels = [l for l in (fp_labels_raw or []) if l in FP_TYPES]
    if not fp_labels:
        fp_labels = [list(FP_TYPES.keys())[0]]
    # Preserve user order while de-duplicating.
    seen = set()
    fp_labels = [l for l in fp_labels if not (l in seen or seen.add(l))]

    fp_specs = [(label, *FP_TYPES[label]) for label in fp_labels]

    jid = new_job()
    threading.Thread(
        target=_run_build_db,
        args=(jid, input_path, output_path, mol_format, fp_specs, name_prop),
        daemon=True,
    ).start()
    return {"job_id": jid}


class ClusterRequest(BaseModel):
    smiles: list[str]
    cutoff: float = 0.4

@app.post("/api/cluster")
def api_cluster(req: ClusterRequest):
    cluster_ids = assign_cluster_ids_from_smiles(req.smiles, cutoff=req.cutoff)
    return {"cluster_ids": cluster_ids}


# ── Lists endpoints ───────────────────────────────────────────────────────────

class ListCreateRequest(BaseModel):
    name: str
    mol_ids: list[int] = []
    smiles: list[str] = []
    identifiers: list[str] = []
    overwrite: bool = False


class ListCombineStep(BaseModel):
    op: str
    name: str


class ListCombineRequest(BaseModel):
    steps: list[ListCombineStep]
    highlight: bool = True
    max_results: int = 0  # 0 = unlimited


@app.get("/api/lists")
def api_lists_get():
    items = _lists_for_current_db()
    out = {
        name: {"count": len(entry.get("ids", [])), "created": entry.get("created", "")}
        for name, entry in items.items()
    }
    return {"lists": out, "db_path": _lists_db_key()}


@app.post("/api/lists")
def api_lists_create(req: ListCreateRequest):
    if not _lists_db_key():
        return JSONResponse({"error": "No database loaded."}, status_code=400)
    ids = list(req.mol_ids)
    unresolved: list = []
    if req.smiles:
        resolved, missing = _lists_resolve_smiles(req.smiles)
        ids.extend(resolved)
        unresolved.extend(missing)
    if req.identifiers:
        resolved, missing = _lists_resolve_identifiers(req.identifiers)
        ids.extend(resolved)
        unresolved.extend(missing)
    if not ids:
        msg = "No molecule IDs to save."
        if unresolved:
            msg += f" ({len(unresolved)} input(s) could not be resolved.)"
        return JSONResponse({"error": msg}, status_code=400)
    ok, msg = _lists_write_one(req.name, ids, req.overwrite)
    if not ok:
        return JSONResponse({"error": msg}, status_code=400)
    return {"ok": True, "message": msg, "count": len(set(ids)), "unresolved": unresolved}


@app.delete("/api/lists/{name}")
def api_lists_delete(name: str):
    ok, msg = _lists_delete_one(name)
    if not ok:
        return JSONResponse({"error": msg}, status_code=404)
    return {"ok": True, "message": msg}


@app.post("/api/lists/combine")
def api_lists_combine(req: ListCombineRequest):
    if not _lists_db_key():
        return JSONResponse({"error": "No database loaded."}, status_code=400)
    if not req.steps:
        return JSONResponse({"error": "No combine steps given."}, status_code=400)

    bucket = _lists_for_current_db()
    # Validate names + ops first so we fail fast and with a clear message.
    valid_ops = {"AND", "OR", "NOT", "XOR"}
    for s in req.steps:
        if s.op.upper() not in valid_ops:
            return JSONResponse({"error": f"Unknown operator: {s.op}"}, status_code=400)
        if s.name not in bucket:
            return JSONResponse({"error": f"List '{s.name}' not found."}, status_code=400)

    # Evaluate left-to-right.
    #   - First step's op is treated as the seed: AND/OR/XOR start from list; NOT starts from universe-list.
    #   - Subsequent NOT is interpreted as set-subtract.
    acc: set | None = None
    for s in req.steps:
        op = s.op.upper()
        members = {int(i) for i in bucket[s.name].get("ids", [])}
        if acc is None:
            acc = (_lists_universe() - members) if op == "NOT" else set(members)
            continue
        if op == "AND":
            acc &= members
        elif op == "OR":
            acc |= members
        elif op == "XOR":
            acc ^= members
        elif op == "NOT":
            acc -= members

    final_ids = sorted(acc or set())
    if req.max_results > 0:
        final_ids = final_ids[:req.max_results]

    with _state_lock:
        mol_map = dict(_state["mol_map"])
    hits = [(mid, 1.0) for mid in final_ids]
    rows = _build_result_rows(hits, mol_map, do_highlight=req.highlight)
    for r in rows:
        r["cluster_id"] = None
    return {"rows": rows, "total": len(acc or set()), "elapsed": 0.0}


@app.get("/api/mol_svg")
def api_mol_svg(smiles: str = Query(""), w: int = Query(300), h: int = Query(200)):
    svg = smiles_to_svg(smiles, width=w, height=h)
    return Response(content=svg, media_type="image/svg+xml")

_MOL_EXTS = {".sdf", ".smi", ".csv", ".txt", ".gz"}

@app.get("/api/fs")
def api_fs(path: str = Query(""), mode: str = Query("h5")):
    """List one directory.
    mode='h5'  → show .h5 files with MolDigger-ready badge (load DB)
    mode='mol' → show molecule input files (.sdf/.smi/.csv/.gz etc.)
    mode='save'→ show directories + existing .h5 as rename references
    """
    target = Path(os.path.expanduser(path)) if path else Path.home()
    try:
        target = target.resolve()
        entries = list(target.iterdir())
    except (PermissionError, OSError) as e:
        return JSONResponse({"error": str(e)}, status_code=403)

    dirs, files = [], []
    for e in sorted(entries, key=lambda x: (x.is_file(), x.name.lower())):
        try:
            is_set_dir = e.is_dir() and e.name.endswith(".fpset")
            if mode == "h5" and is_set_dir:
                manifest_path = e / "manifest.json"
                variants = []
                if manifest_path.exists():
                    try:
                        with open(manifest_path, encoding="utf-8") as fh:
                            m = json.load(fh)
                        variants = [v.get("fp", "?") for v in (m.get("variants") or [])]
                    except (OSError, json.JSONDecodeError):
                        variants = []
                files.append({
                    "name": e.name,
                    "path": str(e),
                    "kind": "set",
                    "ready": bool(variants),
                    "fp_count": len(variants),
                    "fp_names": variants,
                })
                continue
            if e.is_dir() and not e.name.startswith(".") and not is_set_dir:
                dirs.append({"name": e.name, "path": str(e)})
            elif e.is_file():
                n = e.name.lower()
                size_mb = round(e.stat().st_size / 1024 ** 2, 1)
                if mode == "h5":
                    if e.suffix == ".h5" and not n.endswith(".smiles.json"):
                        has_companion = Path(str(e) + ".smiles.json").exists()
                        files.append({"name": e.name, "path": str(e), "size_mb": size_mb, "ready": has_companion})
                elif mode == "mol":
                    if any(n.endswith(ext) for ext in _MOL_EXTS):
                        files.append({"name": e.name, "path": str(e), "size_mb": size_mb})
                elif mode == "save":
                    if e.suffix == ".h5" and not n.endswith(".smiles.json"):
                        files.append({"name": e.name, "path": str(e), "size_mb": size_mb})
        except OSError:
            continue

    parent = str(target.parent) if target != target.parent else None
    return {"path": str(target), "parent": parent, "dirs": dirs, "files": files}

# ── HTML Template ─────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MolDigger — Ultrafast Molecular Structure Searching & Clustering</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --primary: #2563eb;
    --primary-dark: #1d4ed8;
    --bg: #f1f5f9;
    --card-bg: #ffffff;
    --border: #e2e8f0;
    --text: #1e293b;
    --text-muted: #64748b;
    --success: #16a34a;
    --error: #dc2626;
    --warning: #d97706;
    --sidebar-w: 360px;
  }

  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  /* ── Header ── */
  .header {
    background: #0f172a;
    color: #f1f5f9;
    padding: 0 20px;
    height: 52px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
    box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    position: sticky;
    top: 0;
    z-index: 100;
  }
  .header-logo {
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.3px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .header-logo span.hex { color: #60a5fa; font-size: 22px; }
  .header-pills { display: flex; gap: 8px; align-items: center; }
  .pill {
    font-size: 11px;
    font-weight: 600;
    padding: 3px 9px;
    border-radius: 12px;
    text-transform: uppercase;
    letter-spacing: 0.4px;
  }
  .pill.ok  { background: #166534; color: #bbf7d0; }
  .pill.err { background: #7f1d1d; color: #fecaca; }
  .pill.warn { background: #78350f; color: #fde68a; }

  /* ── Layout ── */
  .layout {
    display: flex;
    flex: 1;
    min-height: 0;
  }

  /* ── Sidebar ── */
  .sidebar {
    width: var(--sidebar-w);
    min-width: var(--sidebar-w);
    background: var(--bg);
    border-right: 1px solid var(--border);
    overflow-y: auto;
    padding: 16px 12px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  /* ── Card ── */
  .card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
  }
  .card-header {
    padding: 10px 14px;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    background: #f8fafc;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    cursor: pointer;
    user-select: none;
  }
  .card-header .chevron {
    transition: transform 0.2s;
    font-size: 11px;
  }
  .card-header.collapsed .chevron { transform: rotate(-90deg); }
  .card-body { padding: 14px; display: flex; flex-direction: column; gap: 10px; }
  .card-body.hidden { display: none; }

  /* ── Form elements ── */
  label {
    font-size: 12px;
    font-weight: 500;
    color: var(--text-muted);
    display: block;
    margin-bottom: 4px;
  }
  input[type="text"],
  input[type="number"],
  select,
  textarea {
    width: 100%;
    padding: 7px 10px;
    font-size: 13px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: #fff;
    color: var(--text);
    outline: none;
    transition: border-color 0.15s;
  }
  input:focus, select:focus, textarea:focus {
    border-color: var(--primary);
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12);
  }
  textarea { resize: vertical; min-height: 60px; font-family: monospace; }
  .input-row { display: flex; gap: 6px; align-items: flex-end; }
  .input-row input { flex: 1; }

  /* ── Buttons ── */
  .btn {
    padding: 8px 14px;
    font-size: 13px;
    font-weight: 600;
    border: none;
    border-radius: 7px;
    cursor: pointer;
    transition: background 0.15s, transform 0.05s;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
  }
  .btn:active { transform: scale(0.98); }
  .btn:disabled { opacity: 0.55; cursor: not-allowed; transform: none; }
  .btn-primary { background: var(--primary); color: #fff; }
  .btn-primary:hover:not(:disabled) { background: var(--primary-dark); }
  .btn-sm { padding: 5px 10px; font-size: 12px; }
  .btn-ghost { background: transparent; color: var(--primary); border: 1px solid var(--primary); }
  .btn-ghost:hover:not(:disabled) { background: #eff6ff; }
  .btn-block { width: 100%; justify-content: center; }
  .btn-search {
    background: var(--primary);
    color: #fff;
    width: 100%;
    justify-content: center;
    padding: 10px;
    font-size: 14px;
    border-radius: 8px;
  }
  .btn-search:hover:not(:disabled) { background: var(--primary-dark); }

  /* ── Spinner ── */
  @keyframes spin { to { transform: rotate(360deg); } }
  .spinner {
    width: 14px; height: 14px;
    border: 2px solid rgba(255,255,255,0.35);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    display: none;
  }
  .searching .spinner { display: block; }

  /* ── Tabs ── */
  .tabs { display: flex; gap: 0; border-bottom: 2px solid var(--border); margin-bottom: 12px; }
  .tab-btn {
    flex: 1;
    padding: 8px;
    text-align: center;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    border: none;
    background: none;
    color: var(--text-muted);
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    transition: color 0.15s, border-color 0.15s;
  }
  .tab-btn.active { color: var(--primary); border-bottom-color: var(--primary); }
  .tab-btn:hover:not(.active) { color: var(--text); }

  /* ── DB info ── */
  .db-info {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 12px;
    color: #166534;
    display: none;
  }
  .db-info.visible { display: block; }
  .db-info strong { display: block; margin-bottom: 2px; }

  /* ── Slider ── */
  .slider-row { display: flex; align-items: center; gap: 8px; }
  .slider-row input[type="range"] { flex: 1; cursor: pointer; accent-color: var(--primary); }
  .slider-val {
    min-width: 38px;
    text-align: right;
    font-size: 13px;
    font-weight: 600;
    color: var(--text);
    font-variant-numeric: tabular-nums;
  }

  /* ── Status msg ── */
  .status-msg {
    font-size: 12px;
    padding: 6px 10px;
    border-radius: 6px;
    display: none;
    word-break: break-word;
  }
  .status-msg.info { display: block; background: #eff6ff; color: #1e40af; border: 1px solid #bfdbfe; }
  .status-msg.error { display: block; background: #fef2f2; color: var(--error); border: 1px solid #fecaca; }
  .status-msg.success { display: block; background: #f0fdf4; color: var(--success); border: 1px solid #bbf7d0; }

  /* ── Collapsible build section ── */
  .collapsible-header {
    font-size: 12px;
    font-weight: 600;
    color: var(--primary);
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 4px;
    user-select: none;
    padding: 4px 0;
  }
  .collapsible-header .chev { transition: transform 0.2s; }
  .collapsible-header.open .chev { transform: rotate(90deg); }
  .collapsible-body { display: none; padding-top: 10px; }
  .collapsible-body.open { display: flex; flex-direction: column; gap: 8px; }

  /* ── Tversky params ── */
  #tversky-params { display: none; }
  #tversky-params.visible { display: block; }

  /* ── Main content ── */
  .main {
    flex: 1;
    overflow-y: auto;
    padding: 16px 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  /* ── Results header ── */
  .results-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
  }
  .results-info {
    font-size: 14px;
    color: var(--text-muted);
  }
  .results-info strong { color: var(--text); }

  /* ── Placeholder ── */
  .placeholder {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    color: var(--text-muted);
    padding: 60px 20px;
    text-align: center;
  }
  .placeholder .big-icon { font-size: 56px; opacity: 0.3; }
  .placeholder p { font-size: 15px; max-width: 360px; line-height: 1.5; }

  /* ── Results table ── */
  .table-wrap {
    overflow-x: auto;
    border-radius: 10px;
    border: 1px solid var(--border);
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }
  thead tr {
    background: #f8fafc;
    border-bottom: 2px solid var(--border);
  }
  th {
    padding: 10px 12px;
    text-align: left;
    font-weight: 600;
    font-size: 12px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.4px;
    white-space: nowrap;
    user-select: none;
  }
  th.sortable { cursor: pointer; }
  .build-reopen { display: inline-block; font-size: 12px; color: var(--text-muted); text-decoration: none; margin-top: 4px; }
  .build-reopen:hover { color: var(--primary); text-decoration: underline; }
  .fp-checks { display: flex; flex-direction: column; gap: 4px; padding: 6px 8px; border: 1px solid var(--border); border-radius: 6px; background: var(--bg); max-height: 180px; overflow-y: auto; }
  .fp-checks label { display: flex; align-items: center; gap: 8px; font-size: 13px; font-weight: 400; cursor: pointer; padding: 2px 4px; border-radius: 4px; margin: 0; }
  .fp-checks label:hover { background: var(--border); }
  .fp-checks input[type="checkbox"] { margin: 0; }
  th.sortable:hover { color: var(--primary); }
  th .sort-icon { margin-left: 4px; opacity: 0.5; }
  th.sort-asc .sort-icon::after { content: "▲"; }
  th.sort-desc .sort-icon::after { content: "▼"; }
  td {
    padding: 8px 12px;
    vertical-align: middle;
    border-bottom: 1px solid var(--border);
  }
  tbody tr:last-child td { border-bottom: none; }
  tbody tr:hover { background: #f8fafc; }

  /* ── Score badges ── */
  .score-cell {
    font-variant-numeric: tabular-nums;
    font-weight: 700;
    font-size: 13px;
    padding: 3px 8px;
    border-radius: 5px;
    display: inline-block;
    min-width: 56px;
    text-align: center;
  }

  /* ── Structure cell ── */
  .struct-cell { width: 164px; padding: 6px 8px; }
  .struct-cell svg { display: block; border-radius: 4px; }

  /* ── SMILES cell ── */
  .smiles-cell {
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
    font-size: 11px;
    color: var(--text-muted);
    max-width: 220px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .smiles-actions { display: flex; gap: 4px; margin-top: 4px; }
  .action-btn {
    font-size: 11px;
    padding: 2px 7px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: #fff;
    cursor: pointer;
    color: var(--text-muted);
    transition: background 0.1s, color 0.1s;
  }
  .action-btn:hover { background: var(--primary); color: #fff; border-color: var(--primary); }

  /* ── Name cell ── */
  .name-cell { max-width: 160px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  /* ── Prop cells ── */
  .prop-cell { font-variant-numeric: tabular-nums; white-space: nowrap; }

  /* ── Ketcher modal ── */
  .modal-overlay {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.6);
    z-index: 200;
    align-items: center;
    justify-content: center;
  }
  .modal-overlay.open { display: flex; }
  .modal {
    background: var(--card-bg);
    border-radius: 12px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    width: 96vw;
    height: 92vh;
    max-width: 1100px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .modal-header {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .modal-title { font-weight: 700; font-size: 15px; }
  .modal-actions { display: flex; gap: 8px; }
  .modal-close { background: none; border: none; cursor: pointer; font-size: 16px; color: var(--text-muted); padding: 2px 6px; border-radius: 4px; }
  .modal-close:hover { background: var(--border); color: var(--text); }
  .fs-row { display: flex; align-items: center; gap: 8px; padding: 7px 10px; border-radius: 6px; cursor: pointer; font-size: 13px; margin-bottom: 2px; }
  .fs-row:hover { background: #f1f5f9; }
  .fs-dir { color: var(--primary); font-weight: 500; }
  .fs-file { color: var(--text); }
  .fs-file-dim { opacity: 0.5; cursor: default; }
  .fs-icon { flex-shrink: 0; }
  .modal-body { flex: 1; overflow: hidden; }
  .modal-body iframe {
    width: 100%;
    height: 100%;
    border: none;
  }

  /* ── Responsive ── */
  @media (max-width: 768px) {
    .layout { flex-direction: column; }
    .sidebar { width: 100%; min-width: unset; border-right: none; border-bottom: 1px solid var(--border); }
  }

  /* ── Row number ── */
  .row-num { color: var(--text-muted); font-size: 12px; min-width: 30px; }
  .sel-col { width: 28px; text-align: center; padding: 0 4px; }
  .sel-col input[type="checkbox"] { cursor: pointer; }
</style>
</head>
<body>

<!-- Header -->
<header class="header">
  <div class="header-logo">
    <span class="hex">⬡</span> MolDigger — Ultrafast Molecular Structure Searching &amp; Clustering
  </div>
  <div class="header-pills" id="status-pills">
    <span class="pill err" id="pill-rdkit">RDKit</span>
    <span class="pill err" id="pill-fpsim2">FPSim2</span>
    <span class="pill warn" id="pill-gpu">GPU</span>
  </div>
  <button class="btn btn-ghost btn-sm" onclick="document.getElementById('help-modal').classList.add('open')" style="margin-left:auto;">? Help</button>
</header>

<!-- Layout -->
<div class="layout">

  <!-- Sidebar -->
  <aside class="sidebar">

    <!-- Database card -->
    <div class="card">
      <div class="card-header" onclick="toggleCard(this)">
        Database <span class="chevron">▾</span>
      </div>
      <div class="card-body">
        <div>
          <label for="db-path">Database path (.h5)</label>
          <div class="input-row">
            <input type="text" id="db-path" placeholder="/path/to/database.h5">
            <button class="btn btn-secondary btn-sm" onclick="openDbBrowser()" title="Browse server for .h5 files">&#128193;</button>
            <button class="btn btn-primary btn-sm" onclick="loadDb()">Load</button>
          </div>
        </div>
        <div class="db-info" id="db-info">
          <strong id="db-info-count"></strong>
          <span id="db-info-fp"></span>
        </div>
        <div id="db-status" class="status-msg"></div>

        <a href="#" id="build-reopen" class="build-reopen" onclick="showBuildSection(event)" style="display:none;">+ Build new database</a>

        <!-- Build database collapsible -->
        <div id="build-section">
          <div class="collapsible-header" id="build-toggle" onclick="toggleBuild()">
            <span class="chev">▶</span> Build database&hellip;
          </div>
          <div class="collapsible-body" id="build-body">
            <div>
              <label for="build-input">Input file path</label>
              <div class="input-row">
                <input type="text" id="build-input" placeholder="/path/to/molecules.sdf">
                <button class="btn btn-secondary btn-sm" onclick="openFileBrowser('mol', function(p){document.getElementById('build-input').value=p;}, '', 'Select input file')" title="Browse">&#128193;</button>
              </div>
            </div>
            <div>
              <label for="build-output">Output path (.h5)</label>
              <div class="input-row">
                <input type="text" id="build-output" placeholder="/path/to/output.h5">
                <button class="btn btn-secondary btn-sm" onclick="openFileBrowser('save', function(p){document.getElementById('build-output').value=p;}, '', 'Choose output location')" title="Browse">&#128193;</button>
              </div>
            </div>
            <div>
              <label for="build-format">Format</label>
              <select id="build-format">
                <option value="sdf">SDF (.sdf / .sd)</option>
                <option value="smi">SMILES (.smi / .csv)</option>
              </select>
            </div>
            <div>
              <label>Fingerprints (tick one or more — each adds a sibling .h5 you can switch between at search time)</label>
              <div id="build-fp-options" class="fp-checks"></div>
            </div>
            <div>
              <label for="build-nameprop">Name property (SDF only, blank = sequential)</label>
              <input type="text" id="build-nameprop" placeholder="e.g. _Name or IDNUMBER">
            </div>
            <button class="btn btn-primary btn-block" onclick="startBuildDb()">
              Build Database
            </button>
            <div id="build-status" class="status-msg"></div>
          </div>
        </div>

      </div>
    </div>

    <!-- Search card -->
    <div class="card">
      <div class="card-header">
        Search
      </div>
      <div class="card-body">
        <!-- Tabs -->
        <div class="tabs">
          <button class="tab-btn active" data-type="similarity" onclick="setSearchType('similarity')">Similarity</button>
          <button class="tab-btn" data-type="substructure" onclick="setSearchType('substructure')">Substructure</button>
        </div>

        <!-- SMILES input -->
        <div>
          <label for="query-smiles">Query SMILES</label>
          <div class="input-row">
            <textarea id="query-smiles" rows="2" placeholder="Enter SMILES or use examples below"></textarea>
          </div>
          <div style="display:flex; gap:6px; margin-top:6px; flex-wrap:wrap; align-items:center;">
            <button class="btn btn-ghost btn-sm" id="draw-btn" onclick="openKetcher()" style="display:none;">
              ✏️ Draw
            </button>
            <select id="example-select" style="font-size:12px; flex:1;" onchange="loadExample()">
            </select>
          </div>
        </div>

        <!-- Similarity params -->
        <div id="sim-params">
          <div>
            <label for="fp-select">Fingerprint</label>
            <select id="fp-select"></select>
          </div>
          <div>
            <label for="metric-select">Metric</label>
            <select id="metric-select" onchange="onMetricChange()">
              <option value="tanimoto">Tanimoto</option>
              <option value="dice">Dice</option>
              <option value="tversky">Tversky</option>
            </select>
          </div>
          <div id="tversky-params">
            <div style="display:flex; gap:8px; margin-top:6px;">
              <div style="flex:1;">
                <label for="tv-alpha">Tversky α</label>
                <input type="number" id="tv-alpha" value="0.5" min="0" max="1" step="0.1" style="width:100%;">
              </div>
              <div style="flex:1;">
                <label for="tv-beta">Tversky β</label>
                <input type="number" id="tv-beta" value="0.5" min="0" max="1" step="0.1" style="width:100%;">
              </div>
            </div>
          </div>
          <div>
            <label>Min similarity: <span id="thresh-val">0.70</span></label>
            <div class="slider-row">
              <input type="range" id="threshold" min="0.01" max="1.0" step="0.01" value="0.7"
                     oninput="onThreshChange()">
            </div>
            <div id="thresh-max-row">
              <label>Max similarity: <span id="thresh-max-val">1.00</span></label>
              <div class="slider-row">
                <input type="range" id="threshold-max" min="0.01" max="1.0" step="0.01" value="1.0"
                       oninput="onThreshMaxChange()">
              </div>
            </div>
          </div>
        </div>

        <!-- Common params -->
        <div>
          <label>Max results: <span id="maxres-val">200</span></label>
          <div class="slider-row">
            <input type="range" id="max-results" min="10" max="2000" step="10" value="200"
                   oninput="document.getElementById('maxres-val').textContent=this.value">
          </div>
        </div>
        <div>
          <label for="n-workers">CPU workers</label>
          <input type="number" id="n-workers" value="4" min="1" max="64">
        </div>

        <!-- GPU checkbox -->
        <div id="gpu-row" style="display:none;">
          <label style="display:flex; align-items:center; gap:8px; flex-direction:row; margin:0;">
            <input type="checkbox" id="use-gpu"> Use GPU (CUDA)
          </label>
        </div>

        <!-- Highlight checkbox -->
        <div>
          <label style="display:flex; align-items:center; gap:8px; flex-direction:row; margin:0;">
            <input type="checkbox" id="highlight" checked> Highlight matching atoms
          </label>
        </div>


        <!-- Search button -->
        <button class="btn btn-search" id="search-btn" onclick="doSearch()">
          <span class="spinner" id="search-spinner"></span>
          <span id="search-btn-text">Search</span>
        </button>
        <div id="search-status" class="status-msg"></div>

      </div>
    </div>

    <!-- Lists card -->
    <div class="card">
      <div class="card-header" onclick="toggleListsCard()" style="cursor:pointer; display:flex; justify-content:space-between; align-items:center;">
        <span>Lists</span>
        <span id="lists-card-toggle" style="font-size:14px; color:var(--text-muted);">▾</span>
      </div>
      <div class="card-body" id="lists-card-body">

        <!-- Saved lists -->
        <div>
          <label>Saved lists</label>
          <div id="lists-saved" style="max-height:160px; overflow-y:auto; border:1px solid var(--border); border-radius:4px; padding:4px; background:var(--surface);">
            <div style="color:var(--text-muted); font-size:12px; padding:4px;">No lists yet.</div>
          </div>
        </div>

        <!-- Import list (SMILES or identifiers) -->
        <details style="margin-top:10px;">
          <summary style="cursor:pointer; font-size:13px;">Import list&hellip;</summary>
          <div style="margin-top:6px;">
            <div style="display:flex; gap:12px; font-size:12px; margin-bottom:4px;">
              <label style="display:inline-flex; align-items:center; gap:4px; font-weight:400; cursor:pointer;">
                <input type="radio" name="lists-import-kind" value="smiles" checked onchange="updateListsImportPlaceholder()"> SMILES
              </label>
              <label style="display:inline-flex; align-items:center; gap:4px; font-weight:400; cursor:pointer;">
                <input type="radio" name="lists-import-kind" value="identifiers" onchange="updateListsImportPlaceholder()"> Identifiers
              </label>
            </div>
            <textarea id="lists-import-smiles" rows="3" placeholder="One SMILES per line"></textarea>
            <div class="input-row" style="margin-top:6px;">
              <input type="text" id="lists-import-name" placeholder="List name">
              <button class="btn btn-primary btn-sm" onclick="importListFromText()">Import</button>
            </div>
            <div id="lists-import-status" class="status-msg"></div>
          </div>
        </details>

        <!-- Boolean combiner -->
        <div style="margin-top:14px; padding-top:10px; border-top:1px solid var(--border);">
          <label>Combine lists</label>
          <div id="combine-expr" style="min-height:30px; padding:4px; border:1px solid var(--border); border-radius:4px; background:var(--surface); margin-bottom:6px; font-size:13px;">
            <span style="color:var(--text-muted); font-size:12px;">Expression is empty</span>
          </div>
          <div class="input-row" style="gap:4px;">
            <select id="combine-op" style="flex:0 0 70px;">
              <option value="AND">AND</option>
              <option value="OR">OR</option>
              <option value="NOT">NOT</option>
              <option value="XOR">XOR</option>
            </select>
            <select id="combine-list" style="flex:1;"></select>
            <button class="btn btn-ghost btn-sm" onclick="combineExprAdd()">Add</button>
          </div>
          <div style="display:flex; gap:6px; margin-top:6px;">
            <button class="btn btn-primary btn-sm" onclick="combineRun()">Run</button>
            <button class="btn btn-ghost btn-sm" onclick="combineExprClear()">Clear</button>
          </div>
          <div id="combine-status" class="status-msg"></div>
        </div>

      </div>
    </div>

  </aside>

  <!-- Main content -->
  <main class="main">

    <!-- Results header -->
    <div class="results-header" id="results-header" style="display:none;">
      <div class="results-info">
        <strong id="hit-count">0</strong> hits
        &nbsp;<span id="elapsed-info" style="color:var(--text-muted)"></span>
        &nbsp;<span id="selection-info" style="color:var(--text-muted)"></span>
      </div>
      <div style="display:flex; gap:8px;">
        <button class="btn btn-ghost btn-sm" id="save-selected-btn" onclick="saveSelectedAsList()" style="display:none;">★ Save selected as list</button>
        <button class="btn btn-ghost btn-sm" onclick="saveAllAsList()">★ Save all as list</button>
        <button class="btn btn-ghost btn-sm" onclick="exportCsv()">⬇ Export CSV</button>
      </div>
    </div>

    <!-- Cluster bar — shown after results arrive -->
    <div id="cluster-bar" style="display:none; align-items:center; gap:10px; padding:6px 0; flex-wrap:wrap;">
      <span style="font-weight:600;">Cluster:</span>
      <label for="cluster-cutoff" title="Molecules with Tanimoto ≥ this value are in the same cluster. Higher = more/smaller clusters.">Min similarity</label>
      <input type="range" id="cluster-cutoff" min="0.1" max="0.9" step="0.05" value="0.4" style="width:120px;"
             oninput="document.getElementById('cluster-cutoff-val').textContent=parseFloat(this.value).toFixed(2)">
      <span id="cluster-cutoff-val" style="min-width:2.5em;">0.40</span>
      <button class="btn btn-primary btn-sm" onclick="doClustering()">Apply Clustering</button>
      <button class="btn btn-ghost btn-sm" onclick="clearClusters()">Clear</button>
    </div>

    <!-- Placeholder -->
    <div class="placeholder" id="placeholder">
      <div class="big-icon">⬡</div>
      <p>Load a database and enter a SMILES to start searching.</p>
    </div>

    <!-- Table -->
    <div class="table-wrap" id="table-wrap" style="display:none;">
      <table id="results-table">
        <thead>
          <tr>
            <th class="sel-col"><input type="checkbox" id="select-all-cb" onclick="toggleSelectAll(this)" title="Select all visible"></th>
            <th class="row-num">#</th>
            <th class="sortable cluster-col" style="display:none;" onclick="sortTable('cluster_id')" title="Butina cluster ID">Cluster <span class="sort-icon"></span></th>
            <th class="sortable" onclick="sortTable('name')">Name <span class="sort-icon"></span></th>
            <th>Structure</th>
            <th class="sortable score-col" id="score-header" onclick="sortTable('score')">Score <span class="sort-icon"></span></th>
            <th class="sortable" onclick="sortTable('mw')">MW <span class="sort-icon"></span></th>
            <th class="sortable" onclick="sortTable('clogp')">ClogP <span class="sort-icon"></span></th>
            <th>SMILES</th>
          </tr>
        </thead>
        <tbody id="results-tbody"></tbody>
      </table>
    </div>

  </main>
</div>

<!-- Generic file browser modal -->
<div class="modal-overlay" id="file-browser-modal" style="display:none" onclick="if(event.target===this)closeFileBrowser()">
  <div class="modal" style="max-width:580px;height:72vh;display:flex;flex-direction:column">
    <div class="modal-header">
      <span id="file-browser-title">Browse</span>
      <button class="modal-close" onclick="closeFileBrowser()">&#10005;</button>
    </div>
    <div id="file-browser-crumb" style="padding:8px 14px;background:#f8fafc;border-bottom:1px solid var(--border);font-size:12px;color:var(--text-muted);word-break:break-all"></div>
    <div id="file-browser-list" style="overflow-y:auto;flex:1;padding:8px 10px"></div>
    <!-- Save-as bar: only shown in save mode -->
    <div id="file-browser-savebar" style="display:none;padding:10px 14px;border-top:1px solid var(--border);gap:8px;align-items:center">
      <span style="font-size:12px;font-weight:600;color:var(--text-muted);white-space:nowrap">Filename:</span>
      <input type="text" id="file-browser-filename" placeholder="output.h5" style="flex:1;padding:6px 10px;border:1px solid var(--border);border-radius:6px;font-size:13px">
      <button class="btn btn-primary btn-sm" onclick="confirmSavePath()">Select</button>
    </div>
  </div>
</div>

<!-- Ketcher modal -->
<div class="modal-overlay" id="ketcher-modal">
  <div class="modal">
    <div class="modal-header">
      <span class="modal-title">✏️ Draw Structure — Ketcher</span>
      <div class="modal-actions">
        <button class="btn btn-ghost btn-sm" onclick="closeKetcher()">Cancel</button>
        <button class="btn btn-primary btn-sm" onclick="useKetcherSmiles()">Use This Structure</button>
      </div>
    </div>
    <div class="modal-body">
      <iframe id="ketcher-frame" src="/ketcher/" title="Ketcher structure editor"></iframe>
    </div>
  </div>
</div>

<!-- Help modal -->
<div class="modal-overlay" id="help-modal" onclick="if(event.target===this)this.classList.remove('open')">
  <div class="modal" style="max-width:700px;">
    <div class="modal-header">
      <span class="modal-title">MolDigger — Help</span>
      <button class="btn btn-ghost btn-sm" onclick="document.getElementById('help-modal').classList.remove('open')">Close</button>
    </div>
    <div class="modal-body" style="overflow-y:auto; max-height:70vh; padding:1.2rem 1.5rem; line-height:1.6;">

<h3>Quick Start</h3>
<ol>
  <li><strong>Database</strong> → load an existing <code>.h5</code> file, or build one from an SDF/SMILES file</li>
  <li><strong>Query</strong> → type a SMILES or SMARTS string (live 2D preview updates as you type), or click <strong>Draw Structure</strong> to use Ketcher</li>
  <li>Choose <strong>Similarity</strong> or <strong>Substructure</strong> search and set parameters</li>
  <li>Click <strong>Search</strong> — results appear sorted by score with 2D thumbnails; click again to stop</li>
  <li>Optionally click <strong>Apply Clustering</strong> above the results to group hits by structural similarity</li>
</ol>

<hr>

<h3>Search Types</h3>
<h4>Similarity</h4>
<p>Finds molecules with similar fingerprints to your query using a chosen metric:</p>
<table style="width:100%; border-collapse:collapse; font-size:0.9em;">
  <tr><th style="text-align:left; padding:4px 8px; border-bottom:1px solid #ddd;">Metric</th><th style="text-align:left; padding:4px 8px; border-bottom:1px solid #ddd;">Description</th></tr>
  <tr><td style="padding:4px 8px;"><strong>Tanimoto</strong></td><td style="padding:4px 8px;">Standard Jaccard similarity — most common in cheminformatics</td></tr>
  <tr><td style="padding:4px 8px;"><strong>Dice</strong></td><td style="padding:4px 8px;">2·|A∩B| / (|A|+|B|) — gives higher scores than Tanimoto</td></tr>
  <tr><td style="padding:4px 8px;"><strong>Tversky</strong></td><td style="padding:4px 8px;">Asymmetric; α=1, β=0 finds larger molecules containing your scaffold</td></tr>
</table>
<p>Set <strong>Min / Max</strong> threshold to control the score range. Results are colour-coded green (high) → red (low). The MCS between the query and each hit is highlighted.</p>

<h4>Substructure</h4>
<p>Finds all molecules containing the query as a substructure. Accepts SMILES or SMARTS:</p>
<ul>
  <li><code>c1ccccc1</code> — any benzene ring</li>
  <li><code>[#6]-C(=O)-[#7]</code> — amide bond</li>
  <li><code>[F,Cl,Br,I]</code> — any halogen</li>
  <li><code>[n;H1]</code> — NH in an aromatic ring</li>
</ul>

<hr>

<h3>Clustering</h3>
<p>After a search, click <strong>Apply Clustering</strong>. Adjust <strong>Min similarity</strong> and re-cluster without repeating the search. Always uses Morgan ECFP4 (best chemical groupings). Click <strong>Clear</strong> to restore score order.</p>

<hr>

<h3>Fingerprint Types</h3>
<table style="width:100%; border-collapse:collapse; font-size:0.9em;">
  <tr><th style="text-align:left; padding:4px 8px; border-bottom:1px solid #ddd;">Name</th><th style="text-align:left; padding:4px 8px; border-bottom:1px solid #ddd;">Notes</th></tr>
  <tr><td style="padding:4px 8px;">Morgan / ECFP4</td><td style="padding:4px 8px;">Most common for drug-like molecules — recommended</td></tr>
  <tr><td style="padding:4px 8px;">Morgan / ECFP6</td><td style="padding:4px 8px;">Larger neighbourhood</td></tr>
  <tr><td style="padding:4px 8px;">RDKit Topological</td><td style="padding:4px 8px;">Path-based</td></tr>
  <tr><td style="padding:4px 8px;">MACCS Keys</td><td style="padding:4px 8px;">166-bit, interpretable</td></tr>
  <tr><td style="padding:4px 8px;">Atom Pairs</td><td style="padding:4px 8px;">Encodes atom-pair types</td></tr>
  <tr><td style="padding:4px 8px;">Topological Torsion</td><td style="padding:4px 8px;">Encodes torsion angles</td></tr>
</table>
<p>The fingerprint type is fixed per <code>.h5</code> file at build time. To switch FP at search time, build the database with several types ticked — MolDigger collects the sibling <code>.h5</code> files into a <code>&lt;name&gt;.fpset</code> directory (with a manifest), which shows up in the file browser as a single 📦 entry. Loading the set opens all its FP engines, and the <strong>Fingerprint</strong> dropdown in the search panel becomes a live switcher across them.</p>

<hr>

<h3>Results Table</h3>
<p>Columns: <strong>#</strong> · <strong>Cluster</strong> (when active) · <strong>Name</strong> · <strong>Structure</strong> · <strong>Score</strong> · <strong>MW</strong> · <strong>ClogP</strong> · <strong>SMILES</strong>. Click any column header to sort. <strong>Export CSV</strong> saves all columns.</p>

<hr>

<h3>Installation</h3>
<pre style="background:#f4f4f4; padding:0.8rem; border-radius:4px; font-size:0.85em; overflow-x:auto;">conda create -n moldigger python=3.11
conda activate moldigger
conda install -c conda-forge rdkit
pip install fpsim2 fastapi uvicorn numpy tables
pip install cupy-cuda12x   # optional GPU (match your CUDA version)</pre>
<p>Run the web app:</p>
<pre style="background:#f4f4f4; padding:0.8rem; border-radius:4px; font-size:0.85em; overflow-x:auto;">python moldigger_web.py                       # binds to 0.0.0.0:8000
python moldigger_web.py --host 192.168.1.10   # specific interface
python moldigger_web.py --port 9000           # custom port</pre>

    </div>
  </div>
</div>

<script>
// ── State ──────────────────────────────────────────────────────────────────
let currentResults = [];
let selectedMolIds = new Set();
let currentSearchType = 'similarity';
let sortCol = 'score';
let sortDir = -1; // -1 desc, 1 asc
let pollTimer = null;

const FP_TYPES = [
  "Morgan / ECFP4  (radius=2, 2048 bits)",
  "Morgan / ECFP6  (radius=3, 2048 bits)",
  "RDKit Topological  (minPath=1, maxPath=7)",
  "MACCS Keys  (166 bits)",
  "Atom Pairs  (2048 bits)",
  "Topological Torsion  (2048 bits)",
];

const EXAMPLES = [
  ["— Quick examples —", ""],
  ["Benzene", "c1ccccc1"],
  ["Aspirin", "CC(=O)Oc1ccccc1C(=O)O"],
  ["Caffeine", "Cn1cnc2c1c(=O)n(c(=O)n2C)C"],
  ["Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"],
  ["Sildenafil", "CCCC1=NN(C2=CC(=C(C=C2)S(=O)(=O)N3CCN(CC3)C)OCC)C(=O)C4=C1NC(=NC4=O)C"],
];

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', function() {
  // Populate FP selects
  const fpSelect = document.getElementById('fp-select');
  const buildFpOptions = document.getElementById('build-fp-options');
  FP_TYPES.forEach(function(name, i) {
    fpSelect.appendChild(new Option(name, name));
    const id = 'build-fp-cb-' + i;
    const wrap = document.createElement('label');
    wrap.htmlFor = id;
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.id = id;
    cb.value = name;
    cb.className = 'build-fp-cb';
    if (i === 0) cb.checked = true;  // default: Morgan/ECFP4
    wrap.appendChild(cb);
    wrap.appendChild(document.createTextNode(' ' + name));
    buildFpOptions.appendChild(wrap);
  });
  fpSelect.addEventListener('change', onSearchFpChange);

  // Populate examples
  const exSel = document.getElementById('example-select');
  EXAMPLES.forEach(function(ex) {
    exSel.appendChild(new Option(ex[0], ex[1]));
  });

  loadStatus();
});

// ── Status ─────────────────────────────────────────────────────────────────
function loadStatus() {
  fetch('/api/status')
    .then(function(r) { return r.json(); })
    .then(function(d) {
      setPill('pill-rdkit', d.rdkit, 'RDKit');
      setPill('pill-fpsim2', d.fpsim2, 'FPSim2');
      setPill('pill-gpu', d.gpu, 'GPU', true);

      if (d.ketcher) {
        document.getElementById('draw-btn').style.display = 'inline-flex';
      }
      if (d.gpu) {
        document.getElementById('gpu-row').style.display = 'block';
      }
      if (d.db_path) {
        document.getElementById('db-path').value = d.db_path;
        showDbInfo(d.mol_count, d.fp_name, d.fp_variants);
        refreshLists();
      }
    })
    .catch(function(e) { console.error('status error', e); });
}

function setPill(id, ok, label, warn) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = label;
  if (ok) {
    el.className = 'pill ok';
  } else if (warn) {
    el.className = 'pill warn';
  } else {
    el.className = 'pill err';
  }
}

function showDbInfo(count, fpName, fpVariants) {
  const info = document.getElementById('db-info');
  document.getElementById('db-info-count').textContent = count.toLocaleString() + ' molecules';
  document.getElementById('db-info-fp').textContent = fpName || '';

  // Hide the Build section once a DB is loaded; expose a small re-open link.
  const buildSection = document.getElementById('build-section');
  const buildReopen = document.getElementById('build-reopen');
  if (buildSection) buildSection.style.display = 'none';
  if (buildReopen) buildReopen.style.display = 'inline-block';

  const sel = document.getElementById('fp-select');
  const variants = (fpVariants && fpVariants.length) ? fpVariants : (fpName ? [fpName] : FP_TYPES);
  sel.innerHTML = '';
  variants.forEach(function(v) { sel.appendChild(new Option(v, v)); });
  if (fpName) {
    for (let i = 0; i < sel.options.length; i++) {
      if (sel.options[i].value === fpName) { sel.selectedIndex = i; break; }
    }
  }
  // Only a real switcher when there's more than one variant.
  sel.disabled = variants.length <= 1;
  sel.title = variants.length > 1
    ? 'Switch active fingerprint (sibling .h5 files loaded with this DB)'
    : 'This database has a single fingerprint type. Build with multiple FPs to switch at runtime.';
  info.classList.add('visible');
}

function onSearchFpChange(ev) {
  const fp = ev.target.value;
  if (!fp) return;
  fetch('/api/switch_fp', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({fp_name: fp})
  })
    .then(function(r) { return r.json(); })
    .then(function(d) {
      if (d.error) {
        showStatus('db-status', 'error', d.error);
      } else {
        document.getElementById('db-info-fp').textContent = d.fp_name;
        showStatus('db-status', 'success', 'Active fingerprint: ' + d.fp_name);
      }
    })
    .catch(function(e) { showStatus('db-status', 'error', String(e)); });
}

// ── Card toggle ────────────────────────────────────────────────────────────
function toggleCard(header) {
  const body = header.nextElementSibling;
  header.classList.toggle('collapsed');
  body.classList.toggle('hidden');
}

// ── Collapsible build ──────────────────────────────────────────────────────
function toggleBuild() {
  const toggle = document.getElementById('build-toggle');
  const body = document.getElementById('build-body');
  toggle.classList.toggle('open');
  body.classList.toggle('open');
}

function showBuildSection(ev) {
  if (ev) ev.preventDefault();
  const buildSection = document.getElementById('build-section');
  const buildReopen = document.getElementById('build-reopen');
  if (buildSection) buildSection.style.display = '';
  if (buildReopen) buildReopen.style.display = 'none';
  // Expand the form so the user doesn't have to click again.
  const toggle = document.getElementById('build-toggle');
  const body = document.getElementById('build-body');
  if (toggle && !toggle.classList.contains('open')) toggle.classList.add('open');
  if (body && !body.classList.contains('open')) body.classList.add('open');
}

// ── Generic file browser ───────────────────────────────────────────────────
let _fbMode = 'h5';
let _fbCallback = null;
let _fbCurrentPath = '';

function openFileBrowser(mode, callback, startDir, title) {
  _fbMode = mode;
  _fbCallback = callback;
  document.getElementById('file-browser-title').textContent = title || 'Browse';
  const savebar = document.getElementById('file-browser-savebar');
  savebar.style.display = mode === 'save' ? 'flex' : 'none';
  document.getElementById('file-browser-modal').style.display = 'flex';
  const start = startDir || '';
  browseFileDir(start);
}

function openDbBrowser() {
  const cur = document.getElementById('db-path').value.trim();
  const startDir = cur ? cur.substring(0, cur.lastIndexOf('/')) : '';
  openFileBrowser('h5', function(p) {
    document.getElementById('db-path').value = p;
    loadDb();
  }, startDir, 'Select database');
}

function closeFileBrowser() {
  document.getElementById('file-browser-modal').style.display = 'none';
}

function browseFileDir(path) {
  _fbCurrentPath = path;
  const list = document.getElementById('file-browser-list');
  list.innerHTML = '<p style="padding:16px;color:var(--text-muted);font-size:13px">Loading…</p>';
  fetch(`/api/fs?path=${encodeURIComponent(path)}&mode=${_fbMode}`)
    .then(r => r.json())
    .then(d => {
      if (d.error) { list.innerHTML = `<p style="padding:16px;color:var(--error);font-size:13px">${escHtml(d.error)}</p>`; return; }
      _fbCurrentPath = d.path;
      document.getElementById('file-browser-crumb').textContent = d.path;
      let html = '';
      if (d.parent !== null) {
        html += `<div class="fs-row fs-dir" data-dir="${escHtml(d.parent)}">
          <span class="fs-icon">&#8593;</span><span>.. (up)</span></div>`;
      }
      d.dirs.forEach(dir => {
        html += `<div class="fs-row fs-dir" data-dir="${escHtml(dir.path)}">
          <span class="fs-icon">&#128193;</span><span>${escHtml(dir.name)}</span></div>`;
      });
      d.files.forEach(f => {
        if (f.kind === 'set') {
          const fps = (f.fp_names || []).join(', ');
          const badge = `<span style="margin-left:auto;font-size:10px;padding:2px 6px;border-radius:10px;background:#e0e7ff;color:#3730a3" title="${escHtml(fps)}">${f.fp_count} FPs</span>`;
          html += `<div class="fs-row fs-file" data-file="${escHtml(f.path)}">
            <span class="fs-icon">&#128230;</span>
            <span>${escHtml(f.name)}<span style="color:var(--text-muted);font-size:11px;margin-left:6px">${escHtml(fps)}</span></span>
            ${badge}</div>`;
          return;
        }
        let badge = '';
        if (_fbMode === 'h5') {
          badge = f.ready
            ? `<span style="margin-left:auto;font-size:10px;padding:2px 6px;border-radius:10px;background:#dcfce7;color:#166534">MolDigger DB</span>`
            : `<span style="margin-left:auto;font-size:10px;padding:2px 6px;border-radius:10px;background:#fef9c3;color:#854d0e">no companion</span>`;
        }
        const selectable = _fbMode !== 'h5' || f.ready;
        html += `<div class="fs-row fs-file${selectable ? '' : ' fs-file-dim'}"${selectable ? ` data-file="${escHtml(f.path)}"` : ''}>
          <span class="fs-icon">&#128196;</span>
          <span>${escHtml(f.name)}<span style="color:var(--text-muted);font-size:11px;margin-left:6px">${f.size_mb} MB</span></span>
          ${badge}</div>`;
      });
      if (!d.dirs.length && !d.files.length) {
        html += '<p style="padding:16px;color:var(--text-muted);font-size:13px">Empty directory.</p>';
      }
      list.innerHTML = html;
      list.querySelectorAll('[data-dir]').forEach(el =>
        el.addEventListener('click', () => browseFileDir(el.dataset.dir)));
      list.querySelectorAll('[data-file]').forEach(el =>
        el.addEventListener('click', () => {
          if (_fbMode === 'save') {
            // Clicking existing file pre-fills filename input
            document.getElementById('file-browser-filename').value = el.dataset.file.split('/').pop();
          } else {
            _fbCallback(el.dataset.file);
            closeFileBrowser();
          }
        }));
    })
    .catch(() => { list.innerHTML = '<p style="padding:16px;color:var(--error);font-size:13px">Failed to load directory.</p>'; });
}

function confirmSavePath() {
  const fname = document.getElementById('file-browser-filename').value.trim();
  if (!fname) { alert('Please enter a filename.'); return; }
  const full = _fbCurrentPath.replace(/\/$/, '') + '/' + (fname.endsWith('.h5') ? fname : fname + '.h5');
  _fbCallback(full);
  closeFileBrowser();
}

// ── Load DB ────────────────────────────────────────────────────────────────
function loadDb() {
  const path = document.getElementById('db-path').value.trim();
  if (!path) { showStatus('db-status', 'error', 'Please enter a database path.'); return; }
  showStatus('db-status', 'info', 'Loading…');
  fetch('/api/load_db', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({path: path})
  })
    .then(function(r) { return r.json(); })
    .then(function(d) {
      if (d.error) {
        showStatus('db-status', 'error', d.error);
      } else {
        showStatus('db-status', 'success', 'Database loaded.');
        showDbInfo(d.mol_count, d.fp_name, d.fp_variants);
        refreshLists();  // lists are per-DB; refresh after load
      }
    })
    .catch(function(e) { showStatus('db-status', 'error', String(e)); });
}

// ── Search type tabs ───────────────────────────────────────────────────────
function setSearchType(type) {
  currentSearchType = type;
  document.querySelectorAll('.tab-btn').forEach(function(btn) {
    btn.classList.toggle('active', btn.dataset.type === type);
  });
  document.getElementById('sim-params').style.display = type === 'similarity' ? 'flex' : 'none';
  document.getElementById('sim-params').style.flexDirection = 'column';
  document.getElementById('sim-params').style.gap = '10px';
}

// ── Metric change ──────────────────────────────────────────────────────────
function onMetricChange() {
  const m = document.getElementById('metric-select').value;
  document.getElementById('tversky-params').classList.toggle('visible', m === 'tversky');
  document.getElementById('thresh-max-row').style.display = m === 'tversky' ? 'none' : '';
}

// ── Threshold sliders ──────────────────────────────────────────────────────
function onThreshChange() {
  const min = parseFloat(document.getElementById('threshold').value);
  const maxEl = document.getElementById('threshold-max');
  if (parseFloat(maxEl.value) < min) { maxEl.value = min; onThreshMaxChange(); }
  document.getElementById('thresh-val').textContent = min.toFixed(2);
}

function onThreshMaxChange() {
  const max = parseFloat(document.getElementById('threshold-max').value);
  const minEl = document.getElementById('threshold');
  if (parseFloat(minEl.value) > max) { minEl.value = max; onThreshChange(); }
  document.getElementById('thresh-max-val').textContent = max.toFixed(2);
}

// ── Example picker ─────────────────────────────────────────────────────────
function loadExample() {
  const sel = document.getElementById('example-select');
  const smi = sel.value;
  if (smi) {
    document.getElementById('query-smiles').value = smi;
  }
  sel.selectedIndex = 0;
}

// ── Search ─────────────────────────────────────────────────────────────────
function doSearch() {
  const smiles = document.getElementById('query-smiles').value.trim();
  if (!smiles) { showStatus('search-status', 'error', 'Please enter a query SMILES.'); return; }

  const btn = document.getElementById('search-btn');
  btn.classList.add('searching');
  btn.disabled = true;
  document.getElementById('search-btn-text').textContent = 'Searching…';
  showStatus('search-status', 'info', 'Submitting search…');

  const payload = {
    smiles: smiles,
    search_type: currentSearchType,
    fp: document.getElementById('fp-select').value,
    metric: document.getElementById('metric-select').value,
    threshold: parseFloat(document.getElementById('threshold').value),
    threshold_max: parseFloat(document.getElementById('threshold-max').value),
    tversky_a: parseFloat(document.getElementById('tv-alpha').value),
    tversky_b: parseFloat(document.getElementById('tv-beta').value),
    n_workers: parseInt(document.getElementById('n-workers').value),
    use_gpu: document.getElementById('use-gpu').checked,
    max_results: parseInt(document.getElementById('max-results').value),
    highlight: document.getElementById('highlight').checked,
  };

  fetch('/api/search', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  })
    .then(function(r) { return r.json(); })
    .then(function(d) {
      if (d.error) {
        searchDone();
        showStatus('search-status', 'error', d.error);
      } else {
        pollJob(d.job_id, 'search');
      }
    })
    .catch(function(e) {
      searchDone();
      showStatus('search-status', 'error', String(e));
    });
}

function searchDone() {
  const btn = document.getElementById('search-btn');
  btn.classList.remove('searching');
  btn.disabled = false;
  document.getElementById('search-btn-text').textContent = 'Search';
}

// ── Job polling ────────────────────────────────────────────────────────────
function pollJob(jid, context) {
  if (pollTimer) clearTimeout(pollTimer);
  pollTimer = setTimeout(function() { _doPoll(jid, context); }, 300);
}

function _doPoll(jid, context) {
  fetch('/api/jobs/' + jid)
    .then(function(r) { return r.json(); })
    .then(function(d) {
      if (d.error) {
        if (context === 'search') { searchDone(); showStatus('search-status', 'error', d.error); }
        if (context === 'build') { showStatus('build-status', 'error', d.error); }
        return;
      }

      // Show latest progress
      const prog = d.progress || [];
      if (prog.length > 0) {
        const msg = prog[prog.length - 1];
        if (context === 'search') showStatus('search-status', 'info', msg);
        if (context === 'build') showStatus('build-status', 'info', msg);
      }

      if (d.status === 'running') {
        pollJob(jid, context);
      } else if (d.status === 'done') {
        if (context === 'search') {
          searchDone();
          renderResults(d.result);
          const elapsed = d.result.elapsed ? d.result.elapsed.toFixed(3) + 's' : '';
          showStatus('search-status', 'success',
            d.result.total.toLocaleString() + ' total hits' +
            (d.result.total > d.result.rows.length ? ' (showing ' + d.result.rows.length + ')' : '') +
            (elapsed ? ' — ' + elapsed : ''));
        }
        if (context === 'build') {
          const files = d.result.files || [d.result.path];
          const filesMsg = files.length > 1
            ? ' across ' + files.length + ' FP variants: ' + files.map(function(p) {
                const parts = p.split('/'); return parts[parts.length - 1];
              }).join(', ')
            : ' at ' + d.result.path;
          showStatus('build-status', 'success',
            'Done — ' + (d.result.count || 0).toLocaleString() + ' molecules indexed' + filesMsg);
          // Offer to load the newly built (primary) DB.
          document.getElementById('db-path').value = d.result.path;
        }
      } else if (d.status === 'error') {
        if (context === 'search') { searchDone(); showStatus('search-status', 'error', d.error); }
        if (context === 'build') { showStatus('build-status', 'error', d.error); }
      }
    })
    .catch(function(e) {
      if (context === 'search') { searchDone(); showStatus('search-status', 'error', String(e)); }
      if (context === 'build') { showStatus('build-status', 'error', String(e)); }
    });
}

// ── Render results ─────────────────────────────────────────────────────────
function renderResults(data) {
  currentResults = data.rows || [];
  selectedMolIds = new Set();  // fresh result set ⇒ clear selection
  _updateSelectionInfo();
  const total = data.total || 0;
  let metricLabel;
  if (currentSearchType === 'substructure') metricLabel = 'Substructure';
  else if (currentSearchType === 'list') metricLabel = 'List';
  else metricLabel = document.getElementById('metric-select').options[document.getElementById('metric-select').selectedIndex].text;
  const scoreHeader = document.getElementById('score-header');
  if (scoreHeader) scoreHeader.childNodes[0].textContent = metricLabel + ' ';

  // Show cluster column only when results have cluster IDs
  const hasClusters = currentResults.length > 0 && currentResults[0].cluster_id !== null;
  const clusterDisplay = hasClusters ? '' : 'none';
  document.querySelectorAll('.cluster-col').forEach(el => el.style.display = clusterDisplay);

  // Hide score column when scores aren't meaningful (lists, substructure hits)
  const scoreColDisplay = (currentSearchType === 'list' || currentSearchType === 'substructure') ? 'none' : '';
  document.querySelectorAll('.score-col').forEach(el => el.style.display = scoreColDisplay);

  document.getElementById('placeholder').style.display = 'none';
  document.getElementById('results-header').style.display = 'flex';
  document.getElementById('cluster-bar').style.display = currentResults.length > 0 ? 'flex' : 'none';
  document.getElementById('table-wrap').style.display = 'block';

  document.getElementById('hit-count').textContent = total.toLocaleString();
  if (data.elapsed) {
    document.getElementById('elapsed-info').textContent = '— ' + data.elapsed.toFixed(3) + 's';
  }

  _renderTable();
}

function _hasClusters() {
  return currentResults.length > 0 && currentResults.some(function(r) {
    return r.cluster_id !== null && r.cluster_id !== undefined;
  });
}

function _sortedResults() {
  const hasClusters = _hasClusters();
  return currentResults.slice().sort(function(a, b) {
    if (sortCol === 'cluster_id') {
      const ca = (a.cluster_id !== null && a.cluster_id !== undefined) ? a.cluster_id : 999999;
      const cb = (b.cluster_id !== null && b.cluster_id !== undefined) ? b.cluster_id : 999999;
      return sortDir * (ca - cb);
    }
    if (hasClusters) {
      const ca = (a.cluster_id !== null && a.cluster_id !== undefined) ? a.cluster_id : 999999;
      const cb = (b.cluster_id !== null && b.cluster_id !== undefined) ? b.cluster_id : 999999;
      if (ca !== cb) return ca - cb;
    }
    let av = a[sortCol], bv = b[sortCol];
    if (av === null || av === undefined) av = '';
    if (bv === null || bv === undefined) bv = '';
    if (typeof av === 'string') return sortDir * av.localeCompare(bv);
    return sortDir * (av - bv);
  });
}

function _renderTable() {
  const sorted = _sortedResults();

  // Update sort icons
  document.querySelectorAll('th.sortable').forEach(function(th) {
    th.classList.remove('sort-asc', 'sort-desc');
    const col = th.getAttribute('onclick').replace("sortTable('", '').replace("')", '');
    if (col === sortCol) {
      th.classList.add(sortDir === -1 ? 'sort-desc' : 'sort-asc');
    }
  });

  const tbody = document.getElementById('results-tbody');
  tbody.innerHTML = '';

  // Hide score column for searches where the score isn't meaningful
  // (lists, substructure hits — all 1.00). Applied per-row because rows
  // are rebuilt fresh on every render/sort.
  const scoreColStyle = (currentSearchType === 'list' || currentSearchType === 'substructure') ? 'display:none;' : '';

  sorted.forEach(function(row, i) {
    const tr = document.createElement('tr');
    const scoreHtml = '<span class="score-cell" style="' + scoreStyle(row.score) + '">' + row.score.toFixed(2) + '</span>';

    const clusterDisplay = (row.cluster_id !== null && row.cluster_id !== undefined) ? '' : 'none';
    const checked = selectedMolIds.has(row.mol_id) ? ' checked' : '';
    tr.innerHTML =
      '<td class="sel-col"><input type="checkbox" data-mol-id="' + row.mol_id + '" onclick="toggleRowSelect(this)"' + checked + '></td>' +
      '<td class="row-num">' + (i + 1) + '</td>' +
      '<td class="cluster-col prop-cell" style="display:' + clusterDisplay + ';">' + (row.cluster_id !== null && row.cluster_id !== undefined ? row.cluster_id : '') + '</td>' +
      '<td class="name-cell" title="' + escHtml(row.name || '') + '">' + escHtml(truncate(row.name || '', 24)) + '</td>' +
      '<td class="struct-cell">' + (row.svg || '') + '</td>' +
      '<td class="score-col" style="' + scoreColStyle + '">' + scoreHtml + '</td>' +
      '<td class="prop-cell">' + (row.mw !== null && row.mw !== undefined ? row.mw.toFixed(1) : '—') + '</td>' +
      '<td class="prop-cell">' + (row.clogp !== null && row.clogp !== undefined ? row.clogp.toFixed(2) : '—') + '</td>' +
      '<td>' +
        '<div class="smiles-cell" title="' + escHtml(row.smiles || '') + '">' + escHtml(truncate(row.smiles || '', 40)) + '</div>' +
        '<div class="smiles-actions">' +
          '<button class="action-btn" onclick="copySmiles(' + i + ')">Copy</button>' +
          '<button class="action-btn" onclick="useAsQuery(' + i + ')">Use as query</button>' +
        '</div>' +
      '</td>';

    tbody.appendChild(tr);
  });

  _updateSelectAllCheckbox();
}

// ── Selection ──────────────────────────────────────────────────────────────
function toggleRowSelect(cb) {
  const id = parseInt(cb.getAttribute('data-mol-id'), 10);
  if (cb.checked) selectedMolIds.add(id);
  else selectedMolIds.delete(id);
  _updateSelectionInfo();
  _updateSelectAllCheckbox();
}

function toggleSelectAll(cb) {
  if (cb.checked) {
    currentResults.forEach(function(r) { selectedMolIds.add(r.mol_id); });
  } else {
    currentResults.forEach(function(r) { selectedMolIds.delete(r.mol_id); });
  }
  // Refresh row checkboxes without rebuilding the whole table.
  document.querySelectorAll('#results-tbody input[type="checkbox"][data-mol-id]').forEach(function(box) {
    const id = parseInt(box.getAttribute('data-mol-id'), 10);
    box.checked = selectedMolIds.has(id);
  });
  _updateSelectionInfo();
}

function _updateSelectAllCheckbox() {
  const cb = document.getElementById('select-all-cb');
  if (!cb || !currentResults.length) { if (cb) { cb.checked = false; cb.indeterminate = false; } return; }
  const selectedInResults = currentResults.filter(function(r) { return selectedMolIds.has(r.mol_id); }).length;
  cb.checked = selectedInResults === currentResults.length;
  cb.indeterminate = selectedInResults > 0 && selectedInResults < currentResults.length;
}

function _updateSelectionInfo() {
  const n = selectedMolIds.size;
  document.getElementById('selection-info').textContent = n > 0 ? ('— ' + n + ' selected') : '';
  document.getElementById('save-selected-btn').style.display = n > 0 ? '' : 'none';
}

function scoreStyle(score) {
  const hue = Math.round(score * 120);
  return 'background:hsl(' + hue + ',100%,67%);color:hsl(' + hue + ',80%,22%)';
}

// ── Clustering ─────────────────────────────────────────────────────────────
function doClustering() {
  if (!currentResults.length) return;
  const cutoff = parseFloat(document.getElementById('cluster-cutoff').value);
  const smiles = currentResults.map(function(r) { return r.smiles || ''; });
  const btn = document.querySelector('#cluster-bar .btn-primary');
  btn.disabled = true;
  btn.textContent = 'Clustering…';
  fetch('/api/cluster', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({smiles: smiles, cutoff: cutoff})
  })
    .then(function(r) { return r.json(); })
    .then(function(d) {
      if (d.error) { alert('Clustering failed: ' + d.error); return; }
      const ids = d.cluster_ids;
      // Assign cluster_ids back to currentResults in their current order
      currentResults.forEach(function(row, i) { row.cluster_id = ids[i] !== null ? ids[i] : null; });
      // Sort by cluster then score
      currentResults.sort(function(a, b) {
        const ca = a.cluster_id !== null ? a.cluster_id : 999999;
        const cb = b.cluster_id !== null ? b.cluster_id : 999999;
        if (ca !== cb) return ca - cb;
        return b.score - a.score;
      });
      const hasClusters = ids.some(function(id) { return id !== null; });
      document.querySelectorAll('.cluster-col').forEach(function(el) {
        el.style.display = hasClusters ? '' : 'none';
      });
      _renderTable();
    })
    .catch(function(e) { alert('Clustering error: ' + e); })
    .finally(function() { btn.disabled = false; btn.textContent = 'Apply Clustering'; });
}

function clearClusters() {
  currentResults.forEach(function(r) { r.cluster_id = null; });
  document.querySelectorAll('.cluster-col').forEach(function(el) { el.style.display = 'none'; });
  _renderTable();
}

// ── Sort ───────────────────────────────────────────────────────────────────
function sortTable(col) {
  if (sortCol === col) {
    sortDir *= -1;
  } else {
    sortCol = col;
    sortDir = (col === 'name' || col === 'cluster_id') ? 1 : -1;
  }
  _renderTable();
}

// ── SMILES actions ─────────────────────────────────────────────────────────
function copySmiles(i) {
  const sorted = _sortedResults();
  const smi = sorted[i] ? sorted[i].smiles : '';
  if (smi && navigator.clipboard) {
    navigator.clipboard.writeText(smi).catch(function() {});
  }
}

function useAsQuery(i) {
  const sorted = _sortedResults();
  const smi = sorted[i] ? sorted[i].smiles : '';
  if (smi) {
    document.getElementById('query-smiles').value = smi;
    document.getElementById('query-smiles').scrollIntoView({behavior: 'smooth'});
  }
}

// ── Export CSV ─────────────────────────────────────────────────────────────
function exportCsv() {
  if (!currentResults.length) return;
  const sorted = _sortedResults();
  const hasClusters = _hasClusters();
  const header = hasClusters
    ? ['Index', 'Cluster', 'Name', 'Score', 'MW', 'ClogP', 'SMILES']
    : ['Index', 'Name', 'Score', 'MW', 'ClogP', 'SMILES'];
  const rows = sorted.map(function(r, i) {
    const cluster = (r.cluster_id !== null && r.cluster_id !== undefined) ? r.cluster_id : '';
    const cells = [
      i + 1,
      csvEsc(r.name || ''),
      r.score.toFixed(2),
      r.mw !== null && r.mw !== undefined ? r.mw.toFixed(1) : '',
      r.clogp !== null && r.clogp !== undefined ? r.clogp.toFixed(2) : '',
      csvEsc(r.smiles || ''),
    ];
    if (hasClusters) cells.splice(1, 0, cluster);
    return cells.join(',');
  });
  const csv = [header.join(',')].concat(rows).join('\n');
  const blob = new Blob([csv], {type: 'text/csv'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'moldigger_results.csv';
  a.click();
  URL.revokeObjectURL(url);
}

// ── Lists feature ──────────────────────────────────────────────────────────
let combineExpr = []; // [{op, name}, ...]

function toggleListsCard() {
  const body = document.getElementById('lists-card-body');
  const tog = document.getElementById('lists-card-toggle');
  if (body.style.display === 'none') {
    body.style.display = '';
    tog.textContent = '▾';
  } else {
    body.style.display = 'none';
    tog.textContent = '▸';
  }
}

function refreshLists() {
  fetch('/api/lists')
    .then(function(r) { return r.json(); })
    .then(function(d) {
      const lists = d.lists || {};
      const names = Object.keys(lists).sort();
      const savedEl = document.getElementById('lists-saved');
      const sel = document.getElementById('combine-list');
      if (names.length === 0) {
        savedEl.innerHTML = '<div style="color:var(--text-muted); font-size:12px; padding:4px;">No lists yet.</div>';
        sel.innerHTML = '<option value="">— no lists —</option>';
        return;
      }
      savedEl.innerHTML = names.map(function(name) {
        const meta = lists[name];
        const nameArg = JSON.stringify(name).replace(/"/g, '&quot;');
        return '<div style="display:flex; justify-content:space-between; align-items:center; gap:6px; padding:3px 4px; font-size:13px; border-bottom:1px solid var(--border);">'
          + '<span style="flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="' + escHtml(name) + '">'
          + escHtml(name) + ' <span style="color:var(--text-muted); font-size:11px;">(' + meta.count.toLocaleString() + ')</span></span>'
          + '<button class="btn btn-ghost btn-sm" style="padding:1px 8px; font-size:11px;" onclick="loadList(' + nameArg + ')" title="Load this list as results">Load</button>'
          + '<button class="action-btn" onclick="deleteList(' + nameArg + ')" title="Delete">✕</button>'
          + '</div>';
      }).join('');
      sel.innerHTML = names.map(function(name) {
        return '<option value="' + escHtml(name) + '">' + escHtml(name) + ' (' + lists[name].count + ')</option>';
      }).join('');
    })
    .catch(function() {});
}

function saveAllAsList() {
  if (!currentResults.length) { alert('No results to save.'); return; }
  const name = prompt('Save all ' + currentResults.length + ' results as a list. Name?');
  if (!name) return;
  const ids = currentResults.map(function(r) { return r.mol_id; });
  _postList(name, ids);
}

function saveSelectedAsList() {
  if (selectedMolIds.size === 0) { alert('No rows selected.'); return; }
  const name = prompt('Save ' + selectedMolIds.size + ' selected rows as a list. Name?');
  if (!name) return;
  const ids = Array.from(selectedMolIds);
  _postList(name, ids);
}

function _postList(name, ids) {
  fetch('/api/lists', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name: name, mol_ids: ids, overwrite: false})
  })
    .then(function(r) { return r.json(); })
    .then(function(d) {
      if (d.error) {
        if (d.error.indexOf('already exists') >= 0 && confirm(d.error + ' Overwrite?')) {
          fetch('/api/lists', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name: name, mol_ids: ids, overwrite: true})
          }).then(function(r) { return r.json(); }).then(function(d2) {
            if (d2.error) alert('Save failed: ' + d2.error);
            else { refreshLists(); }
          });
        } else {
          alert('Save failed: ' + d.error);
        }
      } else {
        refreshLists();
      }
    })
    .catch(function(e) { alert('Save failed: ' + e); });
}

function updateListsImportPlaceholder() {
  const kind = document.querySelector('input[name="lists-import-kind"]:checked').value;
  const ta = document.getElementById('lists-import-smiles');
  ta.placeholder = kind === 'identifiers'
    ? 'One identifier (molecule name) per line'
    : 'One SMILES per line';
}

function importListFromText() {
  const name = document.getElementById('lists-import-name').value.trim();
  const raw = document.getElementById('lists-import-smiles').value;
  const kind = document.querySelector('input[name="lists-import-kind"]:checked').value;
  if (!name) { showStatus('lists-import-status', 'error', 'Name is required.'); return; }
  const lines = raw.split(/\r?\n/).map(function(s) { return s.trim(); }).filter(function(s) { return s.length > 0; });
  if (lines.length === 0) {
    showStatus('lists-import-status', 'error',
      kind === 'identifiers' ? 'No identifiers provided.' : 'No SMILES provided.');
    return;
  }
  const baseBody = {name: name};
  if (kind === 'identifiers') baseBody.identifiers = lines;
  else baseBody.smiles = lines;

  function post(overwrite, cb) {
    fetch('/api/lists', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(Object.assign({}, baseBody, {overwrite: overwrite}))
    }).then(function(r) { return r.json(); }).then(cb);
  }
  post(false, function(d) {
    if (d.error) {
      if (d.error.indexOf('already exists') >= 0 && confirm(d.error + ' Overwrite?')) {
        post(true, function(d2) {
          if (d2.error) showStatus('lists-import-status', 'error', d2.error);
          else { showStatus('lists-import-status', 'success', d2.message + (d2.unresolved && d2.unresolved.length ? ' — ' + d2.unresolved.length + ' unresolved.' : '')); refreshLists(); }
        });
      } else {
        showStatus('lists-import-status', 'error', d.error);
      }
    } else {
      showStatus('lists-import-status', 'success', d.message + (d.unresolved && d.unresolved.length ? ' — ' + d.unresolved.length + ' unresolved.' : ''));
      refreshLists();
    }
  });
}

function deleteList(name) {
  if (!confirm('Delete list "' + name + '"?')) return;
  fetch('/api/lists/' + encodeURIComponent(name), {method: 'DELETE'})
    .then(function(r) {
      if (!r.ok && r.status !== 404) {
        return r.text().then(function(t) { throw new Error('HTTP ' + r.status + ': ' + t); });
      }
      return r.json();
    })
    .then(function(d) {
      if (d.error) alert('Delete failed: ' + d.error);
      else refreshLists();
    })
    .catch(function(e) { alert('Delete failed: ' + e.message); });
}

function loadList(name) {
  // Run a single-step combine ("OR <name>") to load this list as results.
  fetch('/api/lists/combine', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({steps: [{op: 'OR', name: name}], highlight: true})
  })
    .then(function(r) { return r.json(); })
    .then(function(d) {
      if (d.error) { alert(d.error); return; }
      currentSearchType = 'list';
      renderResults(d);
      showStatus('search-status', 'success', 'Loaded list "' + name + '" — ' + d.total + ' molecules.');
    });
}

function combineExprAdd() {
  const op = document.getElementById('combine-op').value;
  const name = document.getElementById('combine-list').value;
  if (!name) return;
  combineExpr.push({op: op, name: name});
  _renderCombineExpr();
}

function combineExprClear() {
  combineExpr = [];
  _renderCombineExpr();
  document.getElementById('combine-status').textContent = '';
}

function _renderCombineExpr() {
  const el = document.getElementById('combine-expr');
  if (combineExpr.length === 0) {
    el.innerHTML = '<span style="color:var(--text-muted); font-size:12px;">Expression is empty</span>';
    return;
  }
  el.innerHTML = combineExpr.map(function(step, i) {
    return '<span style="display:inline-block; padding:2px 6px; margin:2px; border-radius:3px; background:var(--bg);">'
      + '<strong>' + step.op + '</strong> ' + escHtml(step.name)
      + ' <a href="#" onclick="combineExprRemove(' + i + '); return false;" style="color:var(--error); text-decoration:none;">✕</a>'
      + '</span>';
  }).join(' ');
}

function combineExprRemove(i) {
  combineExpr.splice(i, 1);
  _renderCombineExpr();
}

function combineRun() {
  if (combineExpr.length === 0) { showStatus('combine-status', 'error', 'Expression is empty.'); return; }
  showStatus('combine-status', 'info', 'Running…');
  fetch('/api/lists/combine', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({steps: combineExpr, highlight: true})
  })
    .then(function(r) { return r.json(); })
    .then(function(d) {
      if (d.error) { showStatus('combine-status', 'error', d.error); return; }
      currentSearchType = 'list';
      renderResults(d);
      showStatus('combine-status', 'success', 'Combined ' + combineExpr.length + ' lists → ' + d.total + ' molecules.');
    })
    .catch(function(e) { showStatus('combine-status', 'error', String(e)); });
}

function csvEsc(s) {
  if (s.indexOf(',') >= 0 || s.indexOf('"') >= 0 || s.indexOf('\n') >= 0) {
    return '"' + s.replace(/"/g, '""') + '"';
  }
  return s;
}

// ── Build DB ───────────────────────────────────────────────────────────────
function startBuildDb() {
  const inputPath = document.getElementById('build-input').value.trim();
  const outputPath = document.getElementById('build-output').value.trim();
  const fps = Array.from(document.querySelectorAll('#build-fp-options input.build-fp-cb:checked'))
    .map(function(c) { return c.value; });
  const fmt = document.getElementById('build-format').value;
  const nameProp = document.getElementById('build-nameprop').value.trim();

  if (!inputPath) { showStatus('build-status', 'error', 'Input file path is required.'); return; }
  if (!outputPath) { showStatus('build-status', 'error', 'Output path is required.'); return; }
  if (fps.length === 0) { showStatus('build-status', 'error', 'Select at least one fingerprint type.'); return; }

  const noun = fps.length > 1 ? (fps.length + ' fingerprint variants') : 'fingerprints';
  showStatus('build-status', 'info', 'Submitting build job (' + noun + ')…');

  fetch('/api/build_db', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      input_path: inputPath,
      output_path: outputPath,
      fp_labels: fps,
      format: fmt,
      name_prop: nameProp,
    })
  })
    .then(function(r) { return r.json(); })
    .then(function(d) {
      if (d.error) {
        showStatus('build-status', 'error', d.error);
      } else {
        pollJob(d.job_id, 'build');
      }
    })
    .catch(function(e) { showStatus('build-status', 'error', String(e)); });
}

// ── Ketcher ────────────────────────────────────────────────────────────────
function openKetcher() {
  document.getElementById('ketcher-modal').classList.add('open');
  const smiles = document.getElementById('query-smiles').value.trim();
  if (smiles) {
    setTimeout(function() {
      try {
        const frame = document.getElementById('ketcher-frame');
        if (frame && frame.contentWindow && frame.contentWindow.ketcher) {
          frame.contentWindow.ketcher.setMolecule(smiles);
        }
      } catch(e) { /* ignore cross-origin or timing issues */ }
    }, 500);
  }
}

function closeKetcher() {
  document.getElementById('ketcher-modal').classList.remove('open');
}

function useKetcherSmiles() {
  try {
    const frame = document.getElementById('ketcher-frame');
    if (frame && frame.contentWindow && frame.contentWindow.ketcher) {
      frame.contentWindow.ketcher.getSmiles().then(function(smi) {
        if (smi) {
          document.getElementById('query-smiles').value = smi;
        }
        closeKetcher();
      }).catch(function(e) {
        console.error('Ketcher getSmiles error:', e);
        closeKetcher();
      });
    } else {
      closeKetcher();
    }
  } catch(e) {
    console.error('Ketcher error:', e);
    closeKetcher();
  }
}

// Close modal on overlay click
document.getElementById('ketcher-modal').addEventListener('click', function(e) {
  if (e.target === this) closeKetcher();
});

// ── Helpers ────────────────────────────────────────────────────────────────
function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function truncate(s, n) {
  if (!s) return '';
  return s.length > n ? s.slice(0, n) + '…' : s;
}

function showStatus(id, type, msg) {
  const el = document.getElementById(id);
  if (!el) return;
  el.className = 'status-msg ' + type;
  el.textContent = msg;
}

// Keyboard shortcut: Enter in SMILES textarea triggers search
document.addEventListener('DOMContentLoaded', function() {
  document.getElementById('query-smiles').addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      doSearch();
    }
  });
});
</script>
</body>
</html>
"""

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MolDigger Web — Browser-based molecular search")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0 — all interfaces). "
                             "Pass a specific IP to restrict access, e.g. --host 192.168.1.10")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    args = parser.parse_args()

    if not RDKIT_OK:
        log.warning("RDKit not found — structure rendering disabled.")
    if not FPSIM2_OK:
        log.warning("FPSim2 not found — search disabled.")
    if GPU_OK:
        log.info("CUDA GPU support available.")
    if not (KETCHER_DIR / "index.html").exists():
        log.info("Ketcher not found — draw button disabled. "
                 "Download ketcher-standalone from https://github.com/epam/ketcher/releases "
                 "and extract to ~/.moldigger/ketcher/")

    import socket
    display_host = socket.getfqdn() if args.host in ("0.0.0.0", "::") else args.host
    log.info(f"Starting MolDigger Web at http://{display_host}:{args.port}")
    uvicorn.run(
        "moldigger_web:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
