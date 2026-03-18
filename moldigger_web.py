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
    "Morgan / FCFP4  (feature, radius=2)":         ("Morgan", {"radius": 2, "fpSize": 2048, "includeChirality": False}),
    "RDKit Topological  (minPath=1, maxPath=7)":   ("RDKit",  {"minPath": 1, "maxPath": 7, "fpSize": 2048}),
    "MACCS Keys  (166 bits)":                      ("MACCSKeys", {}),
    "Atom Pairs  (2048 bits)":                     ("AtomPair", {"fpSize": 2048}),
    "Topological Torsion  (2048 bits)":            ("TopologicalTorsion", {"fpSize": 2048}),
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
    "fp_name": None,
    "mol_count": 0,
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
    """Infer display name from engine fp_type and fp_params."""
    try:
        fp_type = engine.fp_type
        fp_params = engine.fp_params
    except AttributeError:
        return "Unknown"
    for name, (t, p) in FP_TYPES.items():
        if t != fp_type:
            continue
        if fp_type == "Morgan":
            if p.get("radius") == fp_params.get("radius"):
                return name
        else:
            return name
    return f"{fp_type}"

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

        if not mol_map:
            finish_job(jid, error="No database loaded.")
            return

        if not RDKIT_OK:
            finish_job(jid, error="RDKit not available for substructure search.")
            return

        t0 = time.perf_counter()
        update_job_progress(jid, f"Searching {len(mol_map):,} molecules…")

        items = list(mol_map.items())

        def match_chunk(chunk):
            q = Chem.MolFromSmarts(query)
            if q is None:
                q = Chem.MolFromSmiles(query)
            local_results = []
            local_atoms = {}
            for mol_id, entry in chunk:
                smiles = entry.get("smiles", "") if isinstance(entry, dict) else str(entry)
                if not smiles:
                    continue
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                match = mol.GetSubstructMatch(q)
                if match:
                    local_results.append((mol_id, 1.0))
                    local_atoms[mol_id] = list(match)
            return local_results, local_atoms

        n = max(1, n_workers)
        chunk_size = max(1, (len(items) + n - 1) // n)
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

        all_results = []
        match_atoms = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
            for local_results, local_atoms in pool.map(match_chunk, chunks):
                all_results.extend(local_results)
                match_atoms.update(local_atoms)

        elapsed = time.perf_counter() - t0
        update_job_progress(jid, f"Found {len(all_results):,} hits in {elapsed:.3f}s, building results…")

        hits = all_results
        if max_results > 0:
            hits = hits[:max_results]

        rows = _build_result_rows(hits, mol_map, match_atoms=match_atoms, do_highlight=do_highlight)
        for r in rows:
            r["cluster_id"] = None
        finish_job(jid, result={"rows": rows, "total": len(all_results), "elapsed": elapsed})

    except Exception as exc:
        log.exception("Substructure search failed")
        finish_job(jid, error=str(exc))


def _run_build_db(jid: str, input_path: str, output_path: str,
                  mol_format: str, fp_type: str, fp_params: dict, name_prop: str):
    tmp_path = None
    try:
        if not RDKIT_OK:
            finish_job(jid, error="RDKit is not installed.")
            return
        if not FPSIM2_OK:
            finish_job(jid, error="FPSim2 is not installed.")
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

        update_job_progress(jid, f"Loaded {len(entries):,} molecules. Writing fingerprint database…")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".smi", delete=False, encoding="utf-8"
        ) as tmp:
            for seq_id, smi, _ in entries:
                tmp.write(f"{smi}\t{seq_id}\n")
            tmp_path = tmp.name

        create_db_file(tmp_path, output_path, "smi", fp_type, fp_params)

        companion = output_path + ".smiles.json"
        mol_map = {seq_id: {"smiles": smi, "name": name}
                   for seq_id, smi, name in entries}
        with open(companion, "w", encoding="utf-8") as fh:
            json.dump(mol_map, fh)

        update_job_progress(jid, f"Done — {len(entries):,} molecules indexed.")
        finish_job(jid, result={"path": output_path, "count": len(entries)})

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
            "fp_name": _state["fp_name"],
            "mol_count": _state["mol_count"],
        }


@app.post("/api/load_db")
def api_load_db(req: LoadDbRequest):
    path = os.path.expanduser(req.path.strip())
    if not path:
        return JSONResponse({"error": "path is required"}, status_code=400)
    if not Path(path).exists():
        return JSONResponse({"error": f"File not found: {path}"}, status_code=404)
    if not FPSIM2_OK:
        return JSONResponse({"error": "FPSim2 is not installed"}, status_code=500)

    try:
        engine = FPSim2Engine(path)
        companion = path + ".smiles.json"
        mol_map = {}
        if Path(companion).exists():
            with open(companion, encoding="utf-8") as fh:
                raw = json.load(fh)
            mol_map = {str(k): v for k, v in raw.items()}

        fp_name = _fp_name_from_engine(engine)
        with _state_lock:
            _state["engine"] = engine
            _state["mol_map"] = mol_map
            _state["db_path"] = path
            _state["fp_name"] = fp_name
            _state["mol_count"] = len(mol_map)

        return {"ok": True, "mol_count": len(mol_map), "fp_name": fp_name, "path": path}
    except Exception as exc:
        log.exception("load_db failed")
        return JSONResponse({"error": str(exc)}, status_code=500)


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
        fp_label = form.get("fp") or list(FP_TYPES.keys())[0]
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
        fp_label = data.get("fp") or list(FP_TYPES.keys())[0]
        mol_format = data.get("format") or "sdf"
        name_prop = data.get("name_prop") or ""

    if not input_path:
        return JSONResponse({"error": "input_path or file upload required"}, status_code=400)
    if not output_path:
        return JSONResponse({"error": "output_path is required"}, status_code=400)
    if not Path(input_path).exists():
        return JSONResponse({"error": f"Input file not found: {input_path}"}, status_code=404)

    fp_type, fp_params = FP_TYPES.get(fp_label, ("Morgan", {"radius": 2, "fpSize": 2048}))

    jid = new_job()
    threading.Thread(
        target=_run_build_db,
        args=(jid, input_path, output_path, mol_format, fp_type, fp_params, name_prop),
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
            if e.is_dir() and not e.name.startswith("."):
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
  .score-sub  { background: #dbeafe; color: #1e40af; }

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

        <!-- Build database collapsible -->
        <div>
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
              <label for="build-fp">Fingerprint</label>
              <select id="build-fp"></select>
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

  </aside>

  <!-- Main content -->
  <main class="main">

    <!-- Results header -->
    <div class="results-header" id="results-header" style="display:none;">
      <div class="results-info">
        <strong id="hit-count">0</strong> hits
        &nbsp;<span id="elapsed-info" style="color:var(--text-muted)"></span>
      </div>
      <button class="btn btn-ghost btn-sm" onclick="exportCsv()">⬇ Export CSV</button>
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
            <th class="row-num">#</th>
            <th class="cluster-col" style="display:none;" onclick="sortTable('cluster_id')" title="Butina cluster ID">Cluster</th>
            <th class="sortable" onclick="sortTable('name')">Name <span class="sort-icon"></span></th>
            <th>Structure</th>
            <th class="sortable" id="score-header" onclick="sortTable('score')">Score <span class="sort-icon"></span></th>
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
  <tr><td style="padding:4px 8px;">Morgan / FCFP4</td><td style="padding:4px 8px;">Feature-based, pharmacophore-aware</td></tr>
  <tr><td style="padding:4px 8px;">RDKit Topological</td><td style="padding:4px 8px;">Path-based</td></tr>
  <tr><td style="padding:4px 8px;">MACCS Keys</td><td style="padding:4px 8px;">166-bit, interpretable</td></tr>
  <tr><td style="padding:4px 8px;">Atom Pairs</td><td style="padding:4px 8px;">Encodes atom-pair types</td></tr>
  <tr><td style="padding:4px 8px;">Topological Torsion</td><td style="padding:4px 8px;">Encodes torsion angles</td></tr>
</table>
<p>The fingerprint type is fixed at database build time and auto-detected on load.</p>

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
let currentSearchType = 'similarity';
let sortCol = 'score';
let sortDir = -1; // -1 desc, 1 asc
let pollTimer = null;

const FP_TYPES = [
  "Morgan / ECFP4  (radius=2, 2048 bits)",
  "Morgan / ECFP6  (radius=3, 2048 bits)",
  "Morgan / FCFP4  (feature, radius=2)",
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
  const buildFp = document.getElementById('build-fp');
  FP_TYPES.forEach(function(name) {
    fpSelect.appendChild(new Option(name, name));
    buildFp.appendChild(new Option(name, name));
  });

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
        showDbInfo(d.mol_count, d.fp_name);
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

function showDbInfo(count, fpName) {
  const info = document.getElementById('db-info');
  document.getElementById('db-info-count').textContent = count.toLocaleString() + ' molecules';
  document.getElementById('db-info-fp').textContent = fpName || '';
  if (fpName) {
    const sel = document.getElementById('fp-select');
    for (let i = 0; i < sel.options.length; i++) {
      if (sel.options[i].value === fpName) { sel.selectedIndex = i; break; }
    }
  }
  info.classList.add('visible');
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
        showDbInfo(d.mol_count, d.fp_name);
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
          showStatus('build-status', 'success',
            'Done — ' + (d.result.count || 0).toLocaleString() + ' molecules indexed at ' + d.result.path);
          // Offer to load the newly built DB
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
  const total = data.total || 0;
  const metricLabel = currentSearchType === 'substructure' ? 'Substructure'
    : (document.getElementById('metric-select').options[document.getElementById('metric-select').selectedIndex].text);
  const scoreHeader = document.getElementById('score-header');
  if (scoreHeader) scoreHeader.childNodes[0].textContent = metricLabel + ' ';

  // Show cluster column only when results have cluster IDs
  const hasClusters = currentResults.length > 0 && currentResults[0].cluster_id !== null;
  const clusterDisplay = hasClusters ? '' : 'none';
  document.querySelectorAll('.cluster-col').forEach(el => el.style.display = clusterDisplay);

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

function _renderTable() {
  const sorted = currentResults.slice().sort(function(a, b) {
    let av = a[sortCol], bv = b[sortCol];
    if (av === null || av === undefined) av = '';
    if (bv === null || bv === undefined) bv = '';
    if (typeof av === 'string') return sortDir * av.localeCompare(bv);
    return sortDir * (av - bv);
  });

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

  sorted.forEach(function(row, i) {
    const tr = document.createElement('tr');
    const isSub = currentSearchType === 'substructure';

    let scoreHtml;
    if (isSub) {
      scoreHtml = '<span class="score-cell score-sub">match</span>';
    } else {
      scoreHtml = '<span class="score-cell" style="' + scoreStyle(row.score) + '">' + row.score.toFixed(2) + '</span>';
    }

    const clusterDisplay = (row.cluster_id !== null && row.cluster_id !== undefined) ? '' : 'none';
    tr.innerHTML =
      '<td class="row-num">' + (i + 1) + '</td>' +
      '<td class="cluster-col prop-cell" style="display:' + clusterDisplay + ';">' + (row.cluster_id !== null && row.cluster_id !== undefined ? row.cluster_id : '') + '</td>' +
      '<td class="name-cell" title="' + escHtml(row.name || '') + '">' + escHtml(truncate(row.name || '', 24)) + '</td>' +
      '<td class="struct-cell">' + (row.svg || '') + '</td>' +
      '<td>' + scoreHtml + '</td>' +
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
    sortDir = col === 'name' ? 1 : -1;
  }
  _renderTable();
}

// ── SMILES actions ─────────────────────────────────────────────────────────
function copySmiles(i) {
  const sorted = currentResults.slice().sort(function(a, b) {
    let av = a[sortCol], bv = b[sortCol];
    if (av === null || av === undefined) av = '';
    if (bv === null || bv === undefined) bv = '';
    if (typeof av === 'string') return sortDir * av.localeCompare(bv);
    return sortDir * (av - bv);
  });
  const smi = sorted[i] ? sorted[i].smiles : '';
  if (smi && navigator.clipboard) {
    navigator.clipboard.writeText(smi).catch(function() {});
  }
}

function useAsQuery(i) {
  const sorted = currentResults.slice().sort(function(a, b) {
    let av = a[sortCol], bv = b[sortCol];
    if (av === null || av === undefined) av = '';
    if (bv === null || bv === undefined) bv = '';
    if (typeof av === 'string') return sortDir * av.localeCompare(bv);
    return sortDir * (av - bv);
  });
  const smi = sorted[i] ? sorted[i].smiles : '';
  if (smi) {
    document.getElementById('query-smiles').value = smi;
    document.getElementById('query-smiles').scrollIntoView({behavior: 'smooth'});
  }
}

// ── Export CSV ─────────────────────────────────────────────────────────────
function exportCsv() {
  if (!currentResults.length) return;
  const header = ['Index', 'Name', 'Score', 'MW', 'ClogP', 'SMILES'];
  const rows = currentResults.map(function(r, i) {
    return [
      i + 1,
      csvEsc(r.name || ''),
      r.score.toFixed(2),
      r.mw !== null && r.mw !== undefined ? r.mw.toFixed(1) : '',
      r.clogp !== null && r.clogp !== undefined ? r.clogp.toFixed(2) : '',
      csvEsc(r.smiles || ''),
    ].join(',');
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
  const fp = document.getElementById('build-fp').value;
  const fmt = document.getElementById('build-format').value;
  const nameProp = document.getElementById('build-nameprop').value.trim();

  if (!inputPath) { showStatus('build-status', 'error', 'Input file path is required.'); return; }
  if (!outputPath) { showStatus('build-status', 'error', 'Output path is required.'); return; }

  showStatus('build-status', 'info', 'Submitting build job…');

  fetch('/api/build_db', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      input_path: inputPath,
      output_path: outputPath,
      fp: fp,
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
