#!/usr/bin/env python3
"""
MolDigger — Ultrafast Molecular Structure Search
===============================================
FPSim2-based tool with GPU acceleration and a PyQt5 GUI.
Supports Tanimoto similarity with multiple fingerprint types.

Install dependencies:
    pip install rdkit fpsim2 PyQt5 numpy tables

GPU acceleration (optional — requires NVIDIA CUDA):
    pip install cupy-cuda12x   # match your installed CUDA version
    # FPSim2's CudaEngine is then automatically available

Quick start:
    1. Run:  python moldigger.py
    2. Go to the "Database" tab → create or load a .h5 FPSim2 database
    3. Return to "Similarity Search" → enter SMILES → click Search

Author: generated with Claude Code
License: MIT
"""

import sys
import os
import io
import json
import time
import logging
import tempfile
from pathlib import Path

os.environ.setdefault("QT_XCB_GL_INTEGRATION", "xcb_egl")
os.environ.setdefault("LIBGL_ALWAYS_INDIRECT", "0")
os.environ.setdefault("GALLIUM_DRIVER", "d3d12")
os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")
os.environ.setdefault(
    "QTWEBENGINE_CHROMIUM_FLAGS",
    "--disable-gpu --no-sandbox --use-gl=egl"
)

import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QLineEdit, QTextEdit, QComboBox,
    QSlider, QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar,
    QGroupBox, QFormLayout, QSplitter, QMessageBox, QSizePolicy,
    QFrame, QScrollArea, QDialog,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QFont, QColor, QPalette, QIcon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("moldigger")

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
    import cupy  # noqa: F401 — just check availability
    GPU_OK = True
except Exception:
    pass

WEBENGINE_OK = False
_webengine_import_ok = False
try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineScript
    from PyQt5.QtCore import QUrl
    _webengine_import_ok = True
except Exception:
    pass

_WEBENGINE_CACHE = Path.home() / ".moldigger" / "webengine_ok"

def _probe_webengine() -> bool:
    """Probe WebEngine in a subprocess to catch SIGABRT before it kills the main app.
    Result is cached to disk so the slow probe only runs when the environment changes.
    """
    if not _webengine_import_ok:
        return False

    # Return cached result if available
    if _WEBENGINE_CACHE.exists():
        return _WEBENGINE_CACHE.read_text().strip() == "1"

    import subprocess
    env_setup = "".join(
        f"os.environ[{k!r}] = {v!r}\n"
        for k, v in os.environ.items()
        if k.startswith(("QT_", "QTWEBENGINE_", "GALLIUM_", "MESA_", "LIBGL_", "EGL_", "WAYLAND_"))
    )
    probe = (
        "import os, sys\n"
        + env_setup
        + "from PyQt5.QtWidgets import QApplication\n"
        "from PyQt5.QtWebEngineWidgets import QWebEngineView\n"
        "from PyQt5.QtCore import QTimer\n"
        "app = QApplication(sys.argv)\n"
        "v = QWebEngineView()\n"
        "v.show()\n"
        "QTimer.singleShot(5000, app.quit)\n"
        "app.exec_()\n"
        "print('OK')\n"
    )
    try:
        r = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True, timeout=12, text=True,
        )
        ok = "OK" in r.stdout
    except Exception:
        ok = False

    # Cache result
    _WEBENGINE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _WEBENGINE_CACHE.write_text("1" if ok else "0")
    return ok

# Polyfill for JS APIs missing in Qt WebEngine 5.15 (Chromium 87)
_WEBENGINE_POLYFILLS = """
// Object.hasOwn — added Chrome 93, missing in Chromium 87
if (typeof Object.hasOwn === 'undefined') {
    Object.hasOwn = function(obj, prop) {
        return Object.prototype.hasOwnProperty.call(obj, prop);
    };
}
// Array.prototype.at — added Chrome 92
if (typeof Array.prototype.at === 'undefined') {
    Array.prototype.at = function(n) {
        n = Math.trunc(n) || 0;
        if (n < 0) n += this.length;
        if (n < 0 || n >= this.length) return undefined;
        return this[n];
    };
}
// String.prototype.at
if (typeof String.prototype.at === 'undefined') {
    String.prototype.at = Array.prototype.at;
}
"""

KETCHER_DIR = Path.home() / ".moldigger" / "ketcher"

# ── Constants ─────────────────────────────────────────────────────────────────

# Each entry: display_name → (fp_type_str, fp_params_dict)
FP_TYPES: dict = {
    "Morgan / ECFP4  (radius=2, 2048 bits)":       ("Morgan", {"radius": 2, "fpSize": 2048}),
    "Morgan / ECFP6  (radius=3, 2048 bits)":       ("Morgan", {"radius": 3, "fpSize": 2048}),
    "Morgan / FCFP4  (feature, radius=2)":         ("Morgan", {"radius": 2, "fpSize": 2048, "includeChirality": False}),
    "RDKit Topological  (minPath=1, maxPath=7)":   ("RDKit",  {"minPath": 1, "maxPath": 7, "fpSize": 2048}),
    "MACCS Keys  (166 bits)":                      ("MACCSKeys", {}),
    "Atom Pairs  (2048 bits)":                     ("AtomPair", {"fpSize": 2048}),
    "Topological Torsion  (2048 bits)":            ("TopologicalTorsion", {"fpSize": 2048}),
}

def _fp_name_from_engine(fp_type: str, fp_params: dict) -> str | None:
    """Return the FP_TYPES display name that best matches an engine's fp_type/fp_params."""
    for name, (t, p) in FP_TYPES.items():
        if t != fp_type:
            continue
        # For Morgan, match on radius to distinguish ECFP4/ECFP6
        if fp_type == "Morgan":
            if p.get("radius") == fp_params.get("radius"):
                return name
        else:
            return name  # non-Morgan types are unique
    return None


MOL_FORMATS: dict = {
    "SDF (.sdf / .sd)":        "sdf",
    "SMILES (.smi / .csv)":    "smi",
}

RESULT_IMG_SIZE = 130   # px — thumbnail size in the results table

EXAMPLE_SMILES = [
    ("— Quick examples —",              ""),
    ("Benzene",                          "c1ccccc1"),
    ("Aspirin",                          "CC(=O)Oc1ccccc1C(=O)O"),
    ("Caffeine",                         "Cn1cnc2c1c(=O)n(c(=O)n2C)C"),
    ("Paracetamol",                      "CC(=O)Nc1ccc(O)cc1"),
    ("Ibuprofen",                        "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("Sildenafil",                       "CCCC1=NN(C2=CC(=C(C=C2)S(=O)(=O)N3CCN(CC3)C)OCC)C(=O)C4=C1NC(=NC4=O)C"),
    ("Penicillin G",                     "CC1(C(N2C(S1)C(C2=O)NC(=O)Cc3ccccc3)C(=O)O)C"),
    ("Adenine",                          "Nc1ncnc2[nH]cnc12"),
]

# ── Worker: database creation ─────────────────────────────────────────────────

class DBCreationWorker(QThread):
    """
    Background worker that:
      1. Iterates the source molecule file with RDKit
      2. Assigns sequential integer IDs to all valid molecules
      3. Optionally reads a user-chosen SDF property as the display name
      4. Writes a clean SMILES temp file for FPSim2
      5. Calls create_db_file() to build the HDF5 fingerprint database
      6. Saves a companion JSON {mol_id: {"smiles": ..., "name": ...}}
    """

    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)   # success, db_path or error_message

    def __init__(self, input_file: str, output_file: str,
                 mol_format: str, fp_type: str, fp_params: dict,
                 name_prop: str = ""):
        super().__init__()
        self.input_file = input_file
        self.output_file = output_file
        self.mol_format = mol_format
        self.fp_type = fp_type
        self.fp_params = fp_params
        self.name_prop = name_prop   # SDF property to use as display name; "" = sequential

    # ── helpers ───────────────────────────────────────────────────────────────

    def _read_entries(self) -> list[tuple[int, str, str]]:
        """Return [(seq_id, canonical_smiles, display_name), ...]."""
        entries: list[tuple[int, str, str]] = []
        seq = 1

        if self.mol_format == "sdf":
            suppl = Chem.SDMolSupplier(self.input_file, sanitize=True, removeHs=True)
            for mol in suppl:
                if mol is None:
                    continue
                try:
                    smi = Chem.MolToSmiles(mol, canonical=True)
                except Exception:
                    continue

                # Resolve display name
                name = ""
                if self.name_prop == "_Name":
                    name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
                elif self.name_prop and mol.HasProp(self.name_prop):
                    name = str(mol.GetProp(self.name_prop)).strip()
                if not name:
                    name = str(seq)

                entries.append((seq, smi, name))
                seq += 1
        else:
            # SMILES file: SMILES[\tID_or_name]
            with open(self.input_file, encoding="utf-8") as fh:
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
        return entries

    # ── QThread.run ───────────────────────────────────────────────────────────

    def run(self):
        tmp_path = None
        try:
            if not RDKIT_OK:
                raise RuntimeError("RDKit is not installed.")
            if not FPSIM2_OK:
                raise RuntimeError("FPSim2 is not installed.")

            self.progress.emit("Reading molecules…")
            entries = self._read_entries()
            if not entries:
                raise RuntimeError("No valid molecules found in the source file.")

            self.progress.emit(f"Loaded {len(entries):,} molecules. Writing fingerprint database…")

            # Write temp SMILES file: "SMILES\tseq_id\n" — FPSim2 expects this format
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".smi", delete=False, encoding="utf-8"
            ) as tmp:
                for seq_id, smi, _ in entries:
                    tmp.write(f"{smi}\t{seq_id}\n")
                tmp_path = tmp.name

            # Build the FPSim2 HDF5 database
            create_db_file(
                tmp_path,
                self.output_file,
                "smi",
                self.fp_type,
                self.fp_params,
            )

            # Save companion JSON: {seq_id: {"smiles": ..., "name": ...}}
            companion = self.output_file + ".smiles.json"
            mol_map = {seq_id: {"smiles": smi, "name": name}
                       for seq_id, smi, name in entries}
            with open(companion, "w", encoding="utf-8") as fh:
                json.dump(mol_map, fh)

            self.progress.emit(f"Done — {len(entries):,} molecules indexed.")
            self.finished.emit(True, self.output_file)

        except Exception as exc:
            log.exception("DB creation failed")
            self.finished.emit(False, str(exc))
        finally:
            if tmp_path and Path(tmp_path).exists():
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


# ── Worker: similarity search ─────────────────────────────────────────────────

class SearchWorker(QThread):
    """Runs FPSim2 similarity search in a background thread."""

    finished = pyqtSignal(object, float)   # results_array, elapsed_seconds
    error    = pyqtSignal(str)

    def __init__(self, engine, query: str, threshold: float,
                 n_workers: int, use_gpu: bool,
                 metric: str = "tanimoto", tversky_a: float = 0.5, tversky_b: float = 0.5):
        super().__init__()
        self.engine    = engine
        self.query     = query
        self.threshold = threshold
        self.n_workers = n_workers
        self.use_gpu   = use_gpu
        self.metric    = metric
        self.tversky_a = tversky_a
        self.tversky_b = tversky_b

    def run(self):
        try:
            t0 = time.perf_counter()
            # CudaEngine only supports similarity(); fall back to CPU for tversky/dice
            use_gpu = self.use_gpu and self.metric == "tanimoto"
            if self.metric == "tversky":
                results = self.engine.tversky(
                    self.query, self.threshold, self.tversky_a, self.tversky_b,
                    n_workers=self.n_workers
                )
            elif use_gpu:
                results = self.engine.similarity(self.query, self.threshold)
            else:
                results = self.engine.similarity(
                    self.query, self.threshold, metric=self.metric,
                    n_workers=self.n_workers
                )
            elapsed = time.perf_counter() - t0
            self.finished.emit(results, elapsed)
        except Exception as exc:
            log.exception("Search failed")
            self.error.emit(str(exc))


class SubstructureSearchWorker(QThread):
    """Runs RDKit substructure search over the mol_map using multiple threads."""

    finished = pyqtSignal(object, object, float)  # [(mol_id, 1.0)], {mol_id: [atoms]}, elapsed
    error    = pyqtSignal(str)

    def __init__(self, query_smarts: str, mol_map: dict, n_workers: int = 4):
        super().__init__()
        self.query_smarts = query_smarts
        self.mol_map      = mol_map
        self.n_workers    = n_workers
        self._cancelled   = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        import concurrent.futures

        try:
            t0 = time.perf_counter()
            # Try SMARTS first (richer queries), fall back to SMILES
            query = Chem.MolFromSmarts(self.query_smarts)
            if query is None:
                query = Chem.MolFromSmiles(self.query_smarts)
            if query is None:
                self.error.emit(f"Could not parse query as SMARTS or SMILES:\n{self.query_smarts}")
                return

            # Pre-extract items so workers share the same query mol object
            items = list(self.mol_map.items())
            query_smarts_str = self.query_smarts  # send string; each thread re-parses once

            def match_chunk(chunk):
                """Parse query once per thread, then match all molecules in chunk."""
                q = Chem.MolFromSmarts(query_smarts_str)
                if q is None:
                    q = Chem.MolFromSmiles(query_smarts_str)
                local_results = []
                local_atoms   = {}
                for mol_id, entry in chunk:
                    if self._cancelled:
                        return local_results, local_atoms
                    smiles = entry.get("smiles", "") if isinstance(entry, dict) else entry
                    if not smiles:
                        continue
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        continue
                    match = mol.GetSubstructMatch(q)
                    if match:
                        local_results.append((int(mol_id), 1.0))
                        local_atoms[int(mol_id)] = list(match)
                return local_results, local_atoms

            # Split work into equal chunks
            n = self.n_workers
            chunk_size = max(1, (len(items) + n - 1) // n)
            chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

            results     = []
            match_atoms = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
                for local_results, local_atoms in pool.map(match_chunk, chunks):
                    if self._cancelled:
                        return
                    results.extend(local_results)
                    match_atoms.update(local_atoms)

            elapsed = time.perf_counter() - t0
            self.finished.emit(results, match_atoms, elapsed)
        except Exception as exc:
            log.exception("Substructure search failed")
            self.error.emit(str(exc))


# ── Ketcher download worker ───────────────────────────────────────────────────

class KetcherDownloadWorker(QThread):
    """Downloads and extracts the Ketcher standalone bundle from GitHub."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)   # success, message

    def run(self):
        import urllib.request, zipfile, json as _j, shutil

        try:
            # Fetch latest release metadata
            self.progress.emit("Fetching latest Ketcher release info…")
            api_url = "https://api.github.com/repos/epam/ketcher/releases/latest"
            req = urllib.request.Request(api_url, headers={"User-Agent": "moldigger/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = _j.loads(resp.read())

            # Find the standalone zip asset
            dl_url = None
            for asset in data.get("assets", []):
                name = asset["name"]
                if "standalone" in name and name.endswith(".zip"):
                    dl_url = asset["browser_download_url"]
                    break
            if not dl_url:
                self.finished.emit(False, "No standalone zip found in the latest Ketcher release.")
                return

            self.progress.emit(f"Downloading {Path(dl_url).name}…")
            KETCHER_DIR.mkdir(parents=True, exist_ok=True)
            zip_path = KETCHER_DIR.parent / "_ketcher_download.zip"
            urllib.request.urlretrieve(dl_url, zip_path)

            # Extract
            self.progress.emit("Extracting…")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(KETCHER_DIR)
            zip_path.unlink()

            # If everything landed in a subdirectory, lift it up
            html = list(KETCHER_DIR.rglob("index.html"))
            if not html:
                self.finished.emit(False, "index.html not found after extraction.")
                return
            src_dir = html[0].parent
            if src_dir != KETCHER_DIR:
                for item in src_dir.iterdir():
                    dest = KETCHER_DIR / item.name
                    if dest.exists():
                        shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
                    shutil.move(str(item), str(dest))
                try:
                    src_dir.rmdir()
                except OSError:
                    pass

            self.finished.emit(True, str(KETCHER_DIR / "index.html"))

        except Exception as exc:
            log.exception("Ketcher download failed")
            self.finished.emit(False, str(exc))


# ── Ketcher drawing dialog ────────────────────────────────────────────────────

class KetcherDialog(QDialog):
    """
    Embeds Ketcher (MIT-licensed web-based structure editor) in a QWebEngineView.
    Requires PyQt5.QtWebEngineWidgets (conda install -c conda-forge pyqtwebengine).
    Ketcher standalone is downloaded on first use to ~/.molsim/ketcher/.
    """

    smiles_accepted = pyqtSignal(str)

    def __init__(self, parent=None, initial_smiles: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Draw Query Structure — Ketcher")
        self.resize(980, 720)
        self._initial_smiles = initial_smiles
        self._smiles = initial_smiles
        self._ketcher_ready = False
        self._dl_worker = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # Stack: setup page or Ketcher view
        self._stack = QWidget()
        self._stack_layout = QVBoxLayout(self._stack)
        self._stack_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._stack, stretch=1)

        # Bottom button bar
        btn_row = QHBoxLayout()
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color: #666; font-style: italic;")
        btn_row.addWidget(self._status_lbl)
        btn_row.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)
        self._btn_use = QPushButton("Use This Structure")
        self._btn_use.setStyleSheet("font-weight: bold;")
        self._btn_use.setEnabled(False)
        self._btn_use.clicked.connect(self._fetch_smiles)
        btn_row.addWidget(self._btn_use)
        layout.addLayout(btn_row)

        # Decide whether to show setup or load Ketcher
        if (KETCHER_DIR / "index.html").exists():
            self._load_ketcher_view()
        else:
            self._show_setup_page()

    # ── setup page ────────────────────────────────────────────────────────────

    def _show_setup_page(self):
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setAlignment(Qt.AlignCenter)
        vl.setSpacing(16)

        title = QLabel("Ketcher Structure Editor — First-time Setup")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        vl.addWidget(title)

        info = QLabel(
            "MolDigger uses <b>Ketcher</b> (MIT license, by EPAM) for interactive structure drawing.<br>"
            "The standalone bundle (~15 MB) needs to be downloaded once and is stored in<br>"
            f"<code>{KETCHER_DIR}</code>"
        )
        info.setAlignment(Qt.AlignCenter)
        info.setTextFormat(Qt.RichText)
        info.setWordWrap(True)
        vl.addWidget(info)

        self._dl_btn = QPushButton("Download Ketcher Now")
        self._dl_btn.setStyleSheet("font-size: 13px; padding: 8px 20px; font-weight: bold;")
        self._dl_btn.clicked.connect(self._start_download)
        vl.addWidget(self._dl_btn, alignment=Qt.AlignCenter)

        self._dl_progress = QProgressBar()
        self._dl_progress.setRange(0, 0)
        self._dl_progress.hide()
        vl.addWidget(self._dl_progress)

        self._dl_status = QLabel("")
        self._dl_status.setAlignment(Qt.AlignCenter)
        vl.addWidget(self._dl_status)

        manual = QLabel(
            "Or download manually from "
            "<a href='https://github.com/epam/ketcher/releases'>github.com/epam/ketcher/releases</a>"
            f" → extract into <code>{KETCHER_DIR}</code>"
        )
        manual.setTextFormat(Qt.RichText)
        manual.setOpenExternalLinks(True)
        manual.setAlignment(Qt.AlignCenter)
        vl.addWidget(manual)

        self._stack_layout.addWidget(w)

    def _start_download(self):
        self._dl_btn.setEnabled(False)
        self._dl_progress.show()
        self._dl_status.setText("Starting download…")

        self._dl_worker = KetcherDownloadWorker()
        self._dl_worker.progress.connect(self._dl_status.setText)
        self._dl_worker.finished.connect(self._on_download_done)
        self._dl_worker.start()

    def _on_download_done(self, success: bool, msg: str):
        self._dl_progress.hide()
        if success:
            self._dl_status.setText("Download complete — loading editor…")
            # Clear setup page, show Ketcher
            for i in reversed(range(self._stack_layout.count())):
                w = self._stack_layout.itemAt(i).widget()
                if w:
                    w.deleteLater()
            self._load_ketcher_view()
        else:
            self._dl_status.setText(f"Download failed: {msg}")
            self._dl_btn.setEnabled(True)

    # ── Ketcher WebEngine view ─────────────────────────────────────────────────

    def _load_ketcher_view(self):
        self._view = QWebEngineView()
        self._view.loadFinished.connect(self._on_load_finished)

        # Inject polyfills for APIs missing in Chromium 87 (Qt WebEngine 5.15)
        script = QWebEngineScript()
        script.setName("molsearch_polyfills")
        script.setSourceCode(_WEBENGINE_POLYFILLS)
        script.setInjectionPoint(QWebEngineScript.DocumentCreation)
        script.setWorldId(QWebEngineScript.MainWorld)
        script.setRunsOnSubFrames(False)
        self._view.page().scripts().insert(script)

        self._stack_layout.addWidget(self._view)

        html_path = KETCHER_DIR / "index.html"
        self._view.load(QUrl.fromLocalFile(str(html_path)))
        self._status_lbl.setText("Loading Ketcher…")

    def _on_load_finished(self, ok: bool):
        if ok:
            self._status_lbl.setText("Waiting for editor…")
            self._poll_ready()
        else:
            self._status_lbl.setText("Failed to load Ketcher editor.")

    def _poll_ready(self):
        """Poll until ketcher JS object is fully initialised."""
        self._view.page().runJavaScript(
            "typeof ketcher !== 'undefined' && typeof ketcher.getSmiles === 'function'",
            self._on_ready_check,
        )

    def _on_ready_check(self, ready):
        if ready:
            self._ketcher_ready = True
            self._status_lbl.setText("")
            self._btn_use.setEnabled(True)
            if self._initial_smiles:
                smi = self._initial_smiles.replace("\\", "\\\\").replace("'", "\\'")
                self._view.page().runJavaScript(f"ketcher.setMolecule('{smi}')")
        else:
            QTimer.singleShot(250, self._poll_ready)

    # ── Get SMILES ─────────────────────────────────────────────────────────────

    def _fetch_smiles(self):
        """Ask Ketcher for the current SMILES, then accept the dialog."""
        if not self._ketcher_ready:
            self.accept()
            return

        # ketcher.getSmiles() returns a Promise; Qt WebEngine 5.15 does NOT await Promises
        # in runJavaScript callbacks — resolve it in JS first, store in a global, then read back.
        self._view.page().runJavaScript(
            "ketcher.getSmiles().then(function(s){ window.__molsearch_smiles = s; })"
        )
        t = QTimer()
        t.setSingleShot(True)
        t.setInterval(400)
        t.timeout.connect(self._read_smiles_result)
        t.setParent(self)
        self.__smiles_timer = t  # keep reference
        t.start()

    def _read_smiles_result(self):
        self._view.page().runJavaScript(
            "window.__molsearch_smiles || ''",
            self._on_smiles_ready
        )

    def _on_smiles_ready(self, smiles):
        if smiles:
            self._smiles = smiles
        self.accept()

    def get_smiles(self) -> str:
        return self._smiles


# ── Molecule 2D viewer ────────────────────────────────────────────────────────

class MolViewer(QLabel):
    """
    QLabel subclass that renders a 2D molecule image using RDKit.
    Prefers MolDraw2DCairo (PNG, high quality); falls back to PIL.
    """

    def __init__(self, w: int = 280, h: int = 180, parent=None):
        super().__init__(parent)
        self._w, self._h = w, h
        self.setFixedSize(w, h)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "border: 1px solid #c0c0c0; border-radius: 4px; background: #ffffff;"
        )
        self._clear()

    def _clear(self):
        self.clear()
        self.setText('<span style="color:#b0b0b0; font-style:italic;">No structure</span>')

    def render_smiles(self, smiles: str):
        if not RDKIT_OK or not smiles.strip():
            self._clear()
            return
        mol = Chem.MolFromSmiles(smiles.strip())
        self.render_mol(mol)

    def render_mol(self, mol, query_mol=None, highlight_atoms=None):
        if mol is None:
            self._clear()
            return
        try:
            if mol.GetNumConformers() == 0:
                AllChem.Compute2DCoords(mol)

            # Direct atom highlights (substructure mode) take priority over MCS
            if highlight_atoms is not None:
                h_atoms = highlight_atoms
                match_set = set(h_atoms)
                h_bonds = [
                    b.GetIdx() for b in mol.GetBonds()
                    if b.GetBeginAtomIdx() in match_set and b.GetEndAtomIdx() in match_set
                ]
                png_bytes = self._mol_to_png(mol, h_atoms, h_bonds)
                if png_bytes:
                    pix = QPixmap()
                    pix.loadFromData(png_bytes)
                    self.setPixmap(pix)
                    self.setText("")
                else:
                    self._clear()
                return

            # Compute MCS highlight atoms/bonds against the query
            h_atoms, h_bonds = [], []
            if query_mol is not None and RDKIT_OK:
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
                            match_set = set(match)
                            h_bonds = [
                                b.GetIdx() for b in mol.GetBonds()
                                if b.GetBeginAtomIdx() in match_set
                                and b.GetEndAtomIdx() in match_set
                            ]
                except Exception:
                    pass

            png_bytes = self._mol_to_png(mol, h_atoms, h_bonds)
            if png_bytes:
                pix = QPixmap()
                pix.loadFromData(png_bytes)
                self.setPixmap(pix)
                self.setText("")
            else:
                self._clear()
        except Exception as exc:
            log.debug("MolViewer render error: %s", exc)
            self._clear()

    def _mol_to_png(self, mol, h_atoms=(), h_bonds=()) -> bytes | None:
        # Preferred path: RDKit Cairo renderer → PNG bytes
        try:
            drawer = rdMolDraw2D.MolDraw2DCairo(self._w, self._h)
            opts = drawer.drawOptions()
            opts.addStereoAnnotation = True
            opts.padding = 0.1
            if h_atoms:
                drawer.DrawMolecule(mol,
                                    highlightAtoms=h_atoms,
                                    highlightBonds=h_bonds)
            else:
                drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            return drawer.GetDrawingText()
        except Exception:
            pass

        # Fallback: RDKit PIL image
        try:
            img = Draw.MolToImage(mol, size=(self._w, self._h))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            pass

        return None


# ── Query input widget ────────────────────────────────────────────────────────

class QueryWidget(QGroupBox):
    """
    SMILES text input with:
      - live 2D structure preview
      - validity indicator
      - quick example molecules
      - load from MOL/SDF file
    """

    smiles_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__("Query Structure", parent)
        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)
        self._debounce.timeout.connect(self._update_preview)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # 2D preview
        self.viewer = MolViewer(w=280, h=190)
        layout.addWidget(self.viewer, alignment=Qt.AlignCenter)

        # Validity badge
        self.validity_lbl = QLabel("")
        self.validity_lbl.setAlignment(Qt.AlignCenter)
        self.validity_lbl.setFixedHeight(18)
        layout.addWidget(self.validity_lbl)

        # SMILES row
        smiles_row = QHBoxLayout()
        smiles_row.addWidget(QLabel("SMILES:"))
        self.smiles_edit = QLineEdit()
        self.smiles_edit.setPlaceholderText("e.g.  c1ccccc1   or   CC(=O)Oc1ccccc1C(=O)O")
        self.smiles_edit.textChanged.connect(self._debounce.start)
        smiles_row.addWidget(self.smiles_edit)
        btn_clear = QPushButton("✕")
        btn_clear.setFixedWidth(28)
        btn_clear.setToolTip("Clear SMILES")
        btn_clear.clicked.connect(self.smiles_edit.clear)
        smiles_row.addWidget(btn_clear)
        layout.addLayout(smiles_row)

        # Example picker + load from file + draw
        btn_row = QHBoxLayout()
        self.examples_combo = QComboBox()
        for name, smi in EXAMPLE_SMILES:
            self.examples_combo.addItem(name, smi)
        self.examples_combo.activated.connect(self._pick_example)
        btn_row.addWidget(self.examples_combo)
        btn_file = QPushButton("Load from MOL/SDF…")
        btn_file.clicked.connect(self._load_from_file)
        btn_row.addWidget(btn_file)
        layout.addLayout(btn_row)

        # Draw button (requires PyQt5 WebEngine + Ketcher)
        draw_row = QHBoxLayout()
        self._btn_draw = QPushButton("✏  Draw Structure…")
        self._btn_draw.setStyleSheet("font-weight: bold; padding: 5px;")
        self._btn_draw.setToolTip(
            "Open Ketcher interactive structure editor\n"
            "(requires: conda install -c conda-forge pyqtwebengine)"
            if not WEBENGINE_OK else
            "Open Ketcher interactive structure editor"
        )
        self._btn_draw.setEnabled(True)
        self._btn_draw.clicked.connect(self._open_ketcher)
        draw_row.addWidget(self._btn_draw)
        if not WEBENGINE_OK:
            draw_row.addWidget(QLabel(
                '<span style="color:#888; font-size:10px;">'
                '(browser mode)</span>'
            ))
        layout.addLayout(draw_row)

    # ── internal ──────────────────────────────────────────────────────────────

    def _update_preview(self):
        text = self.smiles_edit.text().strip()
        if not RDKIT_OK or not text:
            self.viewer._clear()
            self.validity_lbl.setText("")
            self.smiles_changed.emit(text)
            return

        mol = Chem.MolFromSmiles(text)
        if mol:
            self.validity_lbl.setText(
                '<span style="color:#1a8a1a; font-weight:bold;">✓ Valid SMILES</span>'
            )
        else:
            # Try SMARTS (valid for substructure queries)
            mol = Chem.MolFromSmarts(text)
            if mol:
                self.validity_lbl.setText(
                    '<span style="color:#2255cc; font-weight:bold;">✓ Valid SMARTS</span>'
                )
            else:
                self.validity_lbl.setText(
                    '<span style="color:#cc2200; font-weight:bold;">✗ Invalid SMILES/SMARTS</span>'
                )
        self.viewer.render_mol(mol)
        self.smiles_changed.emit(text)

    def _pick_example(self, index: int):
        smi = self.examples_combo.itemData(index)
        if smi:
            self.smiles_edit.setText(smi)
        self.examples_combo.setCurrentIndex(0)

    def _load_from_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Query Molecule", "",
            "MOL / SDF files (*.mol *.sdf *.sd);;All files (*)"
        )
        if not path:
            return
        if not RDKIT_OK:
            QMessageBox.warning(self, "RDKit missing", "RDKit is required to load MOL files.")
            return
        mol = Chem.MolFromMolFile(path, sanitize=True)
        if mol:
            self.smiles_edit.setText(Chem.MolToSmiles(mol))
        else:
            QMessageBox.warning(self, "Load Error", f"Could not parse molecule from:\n{path}")

    def _open_ketcher(self):
        if WEBENGINE_OK:
            dlg = KetcherDialog(parent=self, initial_smiles=self.smiles_edit.text().strip())
            if dlg.exec_() == QDialog.Accepted:
                smiles = dlg.get_smiles()
                if smiles:
                    self.smiles_edit.setText(smiles)
        else:
            self._open_ketcher_browser()

    def _download_ketcher_then_open(self):
        """Download Ketcher in the background, then open the browser once done."""
        reply = QMessageBox.question(
            self, "Download Ketcher",
            "Ketcher (~15 MB) needs to be downloaded once.\nDownload now?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._btn_draw.setEnabled(False)
        self._btn_draw.setText("Downloading Ketcher…")

        self._dl_worker = KetcherDownloadWorker()
        self._dl_worker.progress.connect(lambda msg: self._btn_draw.setText(msg[:40]))
        self._dl_worker.finished.connect(self._on_ketcher_download_done)
        self._dl_worker.start()

    def _on_ketcher_download_done(self, success: bool, msg: str):
        self._btn_draw.setEnabled(True)
        self._btn_draw.setText("✏  Draw Structure…")
        if success:
            self._open_ketcher_browser()
        else:
            QMessageBox.critical(self, "Download Failed", msg)

    def _open_ketcher_browser(self):
        """Fallback: serve Ketcher via a local HTTP server, open in the default browser.
        A wrapper page adds a 'Use This Structure' button that POSTs SMILES back.
        A QTimer polls the server from the main thread and updates the SMILES field.
        """
        import threading
        import webbrowser
        import socket
        import urllib.parse
        from http.server import BaseHTTPRequestHandler, HTTPServer

        ketcher_dir = KETCHER_DIR
        if not ketcher_dir.exists() or not (ketcher_dir / "index.html").exists():
            self._download_ketcher_then_open()
            return

        # Try a fixed port so the URL is predictable; fall back to random if busy
        PREFERRED_PORT = 18920
        try:
            with socket.socket() as s:
                s.bind(("localhost", PREFERRED_PORT))
            port = PREFERRED_PORT
        except OSError:
            with socket.socket() as s:
                s.bind(("localhost", 0))
                port = s.getsockname()[1]

        smiles_box = [None]  # shared between server thread and Qt timer

        wrapper_html = (
            "<!DOCTYPE html><html><head><style>"
            "body{margin:0;font-family:sans-serif;display:flex;flex-direction:column;height:100vh}"
            "#bar{padding:8px 12px;background:#f0f0f0;display:flex;gap:10px;align-items:center}"
            "button{padding:6px 18px;font-size:14px;cursor:pointer;border-radius:4px}"
            "#btn{background:#2a7ae2;color:#fff;border:none;font-weight:bold}"
            "#st{color:#555;font-size:13px}"
            "iframe{flex:1;border:none}"
            "</style></head><body>"
            "<div id='bar'>"
            f"<button id='btn' onclick='send()'>&#10003; Use This Structure</button>"
            "<span id='st'></span>"
            "</div>"
            f"<iframe id='f' src='http://localhost:{port}/ketcher/'></iframe>"
            "<script>"
            "async function send(){"
            "const w=document.getElementById('f').contentWindow;"
            "try{"
            "const s=await w.ketcher.getSmiles();"
            "await fetch('/submit',{method:'POST',"
            "headers:{'Content-Type':'application/x-www-form-urlencoded'},"
            "body:'s='+encodeURIComponent(s)});"
            "document.getElementById('st').textContent='Sent to MolDigger';}"
            "catch(e){document.getElementById('st').textContent='Error: '+e;}}"
            "</script></body></html>"
        )

        kdir = ketcher_dir

        class _H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def do_GET(self):
                if self.path in ("/", ""):
                    self._serve_text(wrapper_html.encode(), "text/html")
                    return
                rel = self.path.lstrip("/")
                # strip leading "ketcher/" prefix
                if rel.startswith("ketcher/"):
                    rel = rel[len("ketcher/"):]
                if not rel:
                    rel = "index.html"
                fpath = kdir / rel
                if fpath.is_file():
                    ctype = (
                        "application/javascript" if rel.endswith(".js") else
                        "text/css" if rel.endswith(".css") else
                        "text/html"
                    )
                    self._serve_text(fpath.read_bytes(), ctype)
                else:
                    self.send_response(404); self.end_headers()

            def do_POST(self):
                if self.path == "/submit":
                    n = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(n).decode()
                    params = urllib.parse.parse_qs(body)
                    smiles_box[0] = urllib.parse.unquote_plus(
                        params.get("s", [""])[0]
                    )
                    self._serve_text(b"OK", "text/plain")
                else:
                    self.send_response(404); self.end_headers()

            def _serve_text(self, data, ctype):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        srv = HTTPServer(("localhost", port), _H)
        srv.allow_reuse_address = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()

        webbrowser.open(f"http://localhost:{port}/")

        # Poll from the Qt main thread — safe UI updates.
        # Keep server alive so the user can submit updated structures multiple times.
        # Shut down any previous browser session first.
        if hasattr(self, "_browser_srv") and self._browser_srv:
            try:
                self._browser_srv.shutdown()
            except Exception:
                pass
        self._browser_srv = srv

        timer = QTimer(self)
        timer.setInterval(500)

        def _check():
            if smiles_box[0] is not None:
                smiles = smiles_box[0]
                smiles_box[0] = None  # reset so next submission is detected
                if smiles:
                    self.smiles_edit.setText(smiles)

        timer.timeout.connect(_check)
        timer.start()
        self._browser_timer = timer  # keep reference

    # ── public API ────────────────────────────────────────────────────────────

    def get_smiles(self) -> str:
        return self.smiles_edit.text().strip()

    def set_smiles(self, smiles: str):
        self.smiles_edit.setText(smiles)


# ── Search parameters widget ──────────────────────────────────────────────────

class SearchParamsWidget(QGroupBox):
    """Tanimoto threshold, fingerprint type, worker count, GPU toggle."""

    def __init__(self, parent=None):
        super().__init__("Search Parameters", parent)
        self._build_ui()

    def _build_ui(self):
        form = QFormLayout(self)
        form.setSpacing(8)

        # Search type (signal connected after all widgets are built)
        self.search_type = QComboBox()
        self.search_type.addItems(["Similarity", "Substructure"])
        form.addRow("Search type:", self.search_type)

        # Fingerprint type
        self.fp_combo = QComboBox()
        for name in FP_TYPES:
            self.fp_combo.addItem(name)
        form.addRow("Fingerprint:", self.fp_combo)

        # Similarity metric (signal connected after all widgets are built)
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["Tanimoto", "Dice", "Tversky"])
        form.addRow("Metric:", self.metric_combo)

        # Tversky α / β (hidden unless Tversky selected)
        tversky_row = QHBoxLayout()
        self.tversky_a_spin = QDoubleSpinBox()
        self.tversky_a_spin.setRange(0.0, 1.0)
        self.tversky_a_spin.setSingleStep(0.1)
        self.tversky_a_spin.setDecimals(2)
        self.tversky_a_spin.setValue(0.5)
        self.tversky_a_spin.setMaximumWidth(65)
        self.tversky_a_spin.setToolTip("α  — weight on query bits not in target (0=scaffold, 1=strict)")
        self.tversky_b_spin = QDoubleSpinBox()
        self.tversky_b_spin.setRange(0.0, 1.0)
        self.tversky_b_spin.setSingleStep(0.1)
        self.tversky_b_spin.setDecimals(2)
        self.tversky_b_spin.setValue(0.5)
        self.tversky_b_spin.setMaximumWidth(65)
        self.tversky_b_spin.setToolTip("β  — weight on target bits not in query")
        tversky_row.addWidget(QLabel("α"))
        tversky_row.addWidget(self.tversky_a_spin)
        tversky_row.addSpacing(8)
        tversky_row.addWidget(QLabel("β"))
        tversky_row.addWidget(self.tversky_b_spin)
        tversky_row.addStretch()
        self._tversky_label = QLabel("α / β weights:")
        self._tversky_widget = QWidget()
        self._tversky_widget.setLayout(tversky_row)
        self._tversky_row_idx = form.rowCount()
        form.addRow(self._tversky_label, self._tversky_widget)
        self._tversky_label.hide()
        self._tversky_widget.hide()

        # Threshold
        thresh_row = QHBoxLayout()
        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setRange(1, 100)
        self.thresh_slider.setValue(70)
        self.thresh_spin = QDoubleSpinBox()
        self.thresh_spin.setRange(0.01, 1.0)
        self.thresh_spin.setSingleStep(0.01)
        self.thresh_spin.setDecimals(2)
        self.thresh_spin.setValue(0.70)
        self.thresh_spin.setMaximumWidth(72)
        # Keep slider ↔ spinbox in sync
        self.thresh_slider.valueChanged.connect(
            lambda v: self.thresh_spin.blockSignals(True)
            or self.thresh_spin.setValue(v / 100)
            or self.thresh_spin.blockSignals(False)
        )
        self.thresh_spin.valueChanged.connect(
            lambda v: self.thresh_slider.blockSignals(True)
            or self.thresh_slider.setValue(int(v * 100))
            or self.thresh_slider.blockSignals(False)
        )
        thresh_row.addWidget(self.thresh_slider)
        thresh_row.addWidget(self.thresh_spin)
        self._thresh_label = QLabel("Threshold ≥")
        form.addRow(self._thresh_label, thresh_row)

        # Max threshold (upper bound)
        thresh_max_row = QHBoxLayout()
        self.thresh_max_slider = QSlider(Qt.Horizontal)
        self.thresh_max_slider.setRange(1, 100)
        self.thresh_max_slider.setValue(100)
        self.thresh_max_spin = QDoubleSpinBox()
        self.thresh_max_spin.setRange(0.01, 1.0)
        self.thresh_max_spin.setSingleStep(0.01)
        self.thresh_max_spin.setDecimals(2)
        self.thresh_max_spin.setValue(1.00)
        self.thresh_max_spin.setMaximumWidth(72)
        self.thresh_max_slider.valueChanged.connect(
            lambda v: self.thresh_max_spin.blockSignals(True)
            or self.thresh_max_spin.setValue(v / 100)
            or self.thresh_max_spin.blockSignals(False)
            or self._clamp_thresh_min()
        )
        self.thresh_max_spin.valueChanged.connect(
            lambda v: self.thresh_max_slider.blockSignals(True)
            or self.thresh_max_slider.setValue(int(v * 100))
            or self.thresh_max_slider.blockSignals(False)
            or self._clamp_thresh_min()
        )
        self.thresh_spin.valueChanged.connect(lambda v: self._clamp_thresh_max())
        thresh_max_row.addWidget(self.thresh_max_slider)
        thresh_max_row.addWidget(self.thresh_max_spin)
        self._thresh_max_label = QLabel("Threshold ≤")
        self._thresh_max_widget = QWidget()
        self._thresh_max_widget.setLayout(thresh_max_row)
        self._thresh_max_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        form.addRow(self._thresh_max_label, self._thresh_max_widget)

        # Max results to display
        self.max_results = QSpinBox()
        self.max_results.setRange(1, 500_000)
        self.max_results.setValue(2000)
        self.max_results.setSuffix("  molecules")
        form.addRow("Max results:", self.max_results)

        # CPU workers
        self.n_workers = QSpinBox()
        cpu_max = os.cpu_count() or 4
        self.n_workers.setRange(1, cpu_max)
        self.n_workers.setValue(min(4, cpu_max))
        self.n_workers.setToolTip("Number of parallel CPU threads (ignored in GPU mode)")
        form.addRow("CPU workers:", self.n_workers)

        # GPU
        if GPU_OK:
            gpu_label = "Use GPU"
        else:
            gpu_label = "GPU unavailable"
        self.gpu_check = QCheckBox(gpu_label)
        self.gpu_check.setEnabled(GPU_OK)
        form.addRow("Acceleration:", self.gpu_check)

        # Connect after all widgets exist so the signal handler can reference them safely
        self.search_type.currentTextChanged.connect(self._on_search_type_changed)
        self.metric_combo.currentTextChanged.connect(self._on_metric_changed)

    def _clamp_thresh_min(self):
        """Ensure min threshold never exceeds max."""
        mx = self.thresh_max_spin.value()
        if self.thresh_spin.value() > mx:
            self.thresh_spin.setValue(mx)

    def _clamp_thresh_max(self):
        """Ensure max threshold never goes below min."""
        mn = self.thresh_spin.value()
        if self.thresh_max_spin.value() < mn:
            self.thresh_max_spin.setValue(mn)

    def set_fingerprint(self, name: str):
        """Select the fingerprint combo entry matching name (no-op if not found)."""
        idx = self.fp_combo.findText(name)
        if idx >= 0:
            self.fp_combo.setCurrentIndex(idx)

    def _on_search_type_changed(self, stype: str):
        similarity = (stype == "Similarity")
        self.fp_combo.setEnabled(similarity)
        self.metric_combo.setEnabled(similarity)
        self.thresh_slider.setEnabled(similarity)
        self.thresh_spin.setEnabled(similarity)
        self.thresh_max_slider.setEnabled(similarity)
        self.thresh_max_spin.setEnabled(similarity)
        self.gpu_check.setEnabled(similarity and GPU_OK)
        if similarity:
            self._on_metric_changed(self.metric_combo.currentText())
        else:
            self._tversky_label.hide()
            self._tversky_widget.hide()
            self._thresh_max_label.hide()
            self._thresh_max_widget.hide()

    def _on_metric_changed(self, metric: str):
        is_tversky = (metric == "Tversky")
        self._tversky_label.setVisible(is_tversky)
        self._tversky_widget.setVisible(is_tversky)
        self._thresh_max_label.setVisible(not is_tversky)
        self._thresh_max_widget.setVisible(not is_tversky)
        self._thresh_label.setText("Threshold ≥")

    def get_params(self) -> dict:
        fp_name = self.fp_combo.currentText()
        fp_type, fp_params = FP_TYPES[fp_name]
        metric = self.metric_combo.currentText().lower()
        return {
            "search_type": self.search_type.currentText(),
            "fp_type":     fp_type,
            "fp_params":   fp_params,
            "metric":      metric,
            "tversky_a":   self.tversky_a_spin.value(),
            "tversky_b":   self.tversky_b_spin.value(),
            "threshold":     self.thresh_spin.value(),
            "threshold_max": self.thresh_max_spin.value(),
            "max_results":   self.max_results.value(),
            "n_workers":   self.n_workers.value(),
            "use_gpu":     self.gpu_check.isChecked() and GPU_OK,
        }


# ── Results table ─────────────────────────────────────────────────────────────

class ResultsTable(QTableWidget):
    """
    Displays similarity search results.
    Columns: Rank | Mol ID | Tanimoto | SMILES | Structure | MW | ClogP
    """

    use_as_query = pyqtSignal(str)   # emitted with SMILES when "Use as Query" is chosen

    _COLS = ["#", "Name", "Structure", "Score", "MW", "ClogP", "SMILES"]
    _COL_SMILES = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_appearance()
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def _setup_appearance(self):
        self.setColumnCount(len(self._COLS))
        self.setHorizontalHeaderLabels(self._COLS)
        hh = self.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)   # rank
        hh.setSectionResizeMode(1, QHeaderView.Interactive)        # name
        hh.setSectionResizeMode(2, QHeaderView.Fixed)              # structure
        hh.setSectionResizeMode(3, QHeaderView.ResizeToContents)   # score
        hh.setSectionResizeMode(4, QHeaderView.ResizeToContents)   # MW
        hh.setSectionResizeMode(5, QHeaderView.ResizeToContents)   # ClogP
        hh.setSectionResizeMode(6, QHeaderView.Stretch)            # SMILES
        self.setColumnWidth(1, 160)
        self.setColumnWidth(2, RESULT_IMG_SIZE + 6)
        self.verticalHeader().setDefaultSectionSize(RESULT_IMG_SIZE + 6)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setSortingEnabled(True)

    def _smiles_for_row(self, row: int) -> str:
        item = self.item(row, self._COL_SMILES)
        return item.text() if item else ""

    def _show_context_menu(self, pos):
        row = self.rowAt(pos.y())
        if row < 0:
            return
        smiles = self._smiles_for_row(row)
        if not smiles:
            return

        from PyQt5.QtWidgets import QMenu
        menu = QMenu(self)
        act_copy  = menu.addAction("Copy SMILES")
        act_query = menu.addAction("Use as Query")
        action = menu.exec_(self.viewport().mapToGlobal(pos))
        if action == act_copy:
            QApplication.clipboard().setText(smiles)
        elif action == act_query:
            self.use_as_query.emit(smiles)

    def populate(self, results, mol_map: dict, query_mol=None, highlight_map=None, metric: str = "Score"):
        """
        results       : numpy structured array with (mol_id, coeff) OR list of (mol_id, score)
        mol_map       : {int(mol_id): {"smiles": str, "name": str}}
        query_mol     : optional RDKit Mol — used to highlight MCS in hit thumbnails
        highlight_map : optional {int(mol_id): [atom_indices]} — direct substructure highlights
                        (takes priority over MCS when provided)
                  OR legacy {int(mol_id): smiles_str}
        metric        : label shown in the Score column header
        """
        self.setHorizontalHeaderItem(3, QTableWidgetItem(metric))
        self.setSortingEnabled(False)
        self.clearContents()
        self.setRowCount(0)

        if results is None or len(results) == 0:
            return

        sorted_res = sorted(results, key=lambda r: float(r[1]), reverse=True)

        for rank, row_data in enumerate(sorted_res, start=1):
            mol_id = int(row_data[0])
            score  = float(row_data[1])
            row    = self.rowCount()
            self.insertRow(row)
            self.setRowHeight(row, RESULT_IMG_SIZE + 6)

            # Resolve entry — support both new dict format and legacy str format
            entry  = mol_map.get(mol_id, {})
            if isinstance(entry, dict):
                smiles = entry.get("smiles", "")
                name   = entry.get("name", str(mol_id))
            else:
                smiles = entry
                name   = str(mol_id)

            # --- Rank ---
            rank_item = QTableWidgetItem()
            rank_item.setData(Qt.DisplayRole, rank)
            rank_item.setTextAlignment(Qt.AlignCenter)
            self.setItem(row, 0, rank_item)

            # --- Name / identifier ---
            self.setItem(row, 1, QTableWidgetItem(name))

            # --- Structure thumbnail (col 2) ---
            if smiles and RDKIT_OK:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    thumb = MolViewer(RESULT_IMG_SIZE, RESULT_IMG_SIZE)
                    h_atoms = highlight_map.get(mol_id) if highlight_map else None
                    if h_atoms is not None:
                        thumb.render_mol(mol, highlight_atoms=h_atoms)
                    else:
                        thumb.render_mol(mol, query_mol=query_mol)
                    self.setCellWidget(row, 2, thumb)

                    mw_item = QTableWidgetItem()
                    mw_item.setData(Qt.DisplayRole, round(Descriptors.MolWt(mol), 2))
                    mw_item.setTextAlignment(Qt.AlignCenter)
                    self.setItem(row, 4, mw_item)

                    clogp_item = QTableWidgetItem()
                    clogp_item.setData(Qt.DisplayRole, round(Descriptors.MolLogP(mol), 2))
                    clogp_item.setTextAlignment(Qt.AlignCenter)
                    self.setItem(row, 5, clogp_item)

            # --- Score (col 3) ---
            score_item = QTableWidgetItem()
            if highlight_map is not None:
                score_item.setData(Qt.DisplayRole, "match")
                score_item.setBackground(QColor.fromHsvF(1/3, 0.65, 1.0))  # green
            else:
                score_item.setData(Qt.DisplayRole, f"{score:.2f}")
                score_item.setBackground(QColor.fromHsvF(score / 3.0, 0.65, 1.0))
            score_item.setTextAlignment(Qt.AlignCenter)
            self.setItem(row, 3, score_item)

            # --- SMILES (col 6) ---
            self.setItem(row, 6, QTableWidgetItem(smiles))

        self.setSortingEnabled(True)
        self.horizontalHeader().setSectionResizeMode(6, QHeaderView.Stretch)

    def export_csv(self, path: str):
        import csv
        skip_cols = {2}   # skip the structure image column
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                self.horizontalHeaderItem(c).text()
                for c in range(self.columnCount()) if c not in skip_cols
            )
            for r in range(self.rowCount()):
                writer.writerow(
                    (self.item(r, c).text() if self.item(r, c) else "")
                    for c in range(self.columnCount()) if c not in skip_cols
                )


# ── Database panel ────────────────────────────────────────────────────────────

class DatabasePanel(QWidget):
    """
    Two sections:
      - Load an existing FPSim2 .h5 database
      - Create a new database from an SDF or SMILES file
    """

    db_loaded = pyqtSignal(str)   # emits the validated db_path

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: DBCreationWorker | None = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # ── Load existing ──────────────────────────────────────────────────
        load_box = QGroupBox("Load Existing FPSim2 Database")
        load_layout = QHBoxLayout(load_box)
        self.db_path_edit = QLineEdit()
        self.db_path_edit.setPlaceholderText("Path to .h5 / .hdf5 fingerprint database…")
        load_layout.addWidget(self.db_path_edit)
        btn_browse_db = QPushButton("Browse…")
        btn_browse_db.clicked.connect(self._browse_db)
        load_layout.addWidget(btn_browse_db)
        btn_load = QPushButton("Load Database")
        btn_load.setStyleSheet("font-weight: bold;")
        btn_load.clicked.connect(self._load_db)
        load_layout.addWidget(btn_load)
        layout.addWidget(load_box)

        # ── Create new ─────────────────────────────────────────────────────
        create_box = QGroupBox("Create New FPSim2 Database from Molecule File")
        create_form = QFormLayout(create_box)
        create_form.setSpacing(8)

        # Input file
        in_row = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Source molecules (.sdf or .smi)")
        in_row.addWidget(self.input_edit)
        btn_in = QPushButton("Browse…")
        btn_in.clicked.connect(self._browse_input)
        in_row.addWidget(btn_in)
        create_form.addRow("Molecule file:", in_row)

        # Output file
        out_row = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Output .h5 database path")
        out_row.addWidget(self.output_edit)
        btn_out = QPushButton("Browse…")
        btn_out.clicked.connect(self._browse_output)
        out_row.addWidget(btn_out)
        create_form.addRow("Output database:", out_row)

        # Format
        self.format_combo = QComboBox()
        for k in MOL_FORMATS:
            self.format_combo.addItem(k)
        create_form.addRow("File format:", self.format_combo)

        # Fingerprint type
        self.create_fp_combo = QComboBox()
        for name in FP_TYPES:
            self.create_fp_combo.addItem(name)
        create_form.addRow("Fingerprint type:", self.create_fp_combo)

        # Molecule identifier property (SDF only)
        self.name_prop_combo = QComboBox()
        self.name_prop_combo.addItem("Sequential  (#1, #2, …)", "")
        self.name_prop_combo.setToolTip(
            "SDF property to use as the molecule name/identifier in results.\n"
            "Select the input file first to populate this list."
        )
        self.format_combo.currentIndexChanged.connect(
            lambda: self._populate_name_prop_combo(self.input_edit.text())
        )
        create_form.addRow("Molecule ID property:", self.name_prop_combo)

        # Create button
        btn_create = QPushButton("Create Database")
        btn_create.setStyleSheet("font-weight: bold; padding: 6px 12px;")
        btn_create.clicked.connect(self._create_db)
        create_form.addRow("", btn_create)

        # Progress bar (hidden until creation starts)
        self.create_progress = QProgressBar()
        self.create_progress.setRange(0, 0)   # indeterminate
        self.create_progress.hide()
        create_form.addRow("", self.create_progress)

        # Status label
        self.create_status = QLabel("")
        self.create_status.setWordWrap(True)
        create_form.addRow("", self.create_status)

        layout.addWidget(create_box)

        # ── Overall status ─────────────────────────────────────────────────
        self.status_lbl = QLabel("No database loaded.")
        self.status_lbl.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.status_lbl)
        layout.addStretch()

    # ── Load ──────────────────────────────────────────────────────────────────

    def _browse_db(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open FPSim2 Database", "",
            "HDF5 files (*.h5 *.hdf5);;All files (*)"
        )
        if path:
            self.db_path_edit.setText(path)

    def _load_db(self):
        path = self.db_path_edit.text().strip()
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "Not Found", "Please select a valid .h5 database file.")
            return
        self.status_lbl.setText(f"Database loaded: {Path(path).name}")
        self.status_lbl.setStyleSheet("color: #1a8a1a; font-style: normal; font-weight: bold;")
        self.db_loaded.emit(path)

    # ── Create ────────────────────────────────────────────────────────────────

    def _browse_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Molecule File", "",
            "Molecule files (*.sdf *.sd *.smi *.csv *.txt);;All files (*)"
        )
        if path:
            self.input_edit.setText(path)
            if not self.output_edit.text():
                self.output_edit.setText(str(Path(path).with_suffix(".h5")))
            self._populate_name_prop_combo(path)

    def _populate_name_prop_combo(self, sdf_path: str):
        """Read the first molecule from the SDF and populate the ID property combo."""
        self.name_prop_combo.blockSignals(True)
        self.name_prop_combo.clear()
        self.name_prop_combo.addItem("Sequential  (#1, #2, …)", "")

        fmt_key = self.format_combo.currentText()
        if MOL_FORMATS.get(fmt_key) != "sdf" or not RDKIT_OK:
            self.name_prop_combo.blockSignals(False)
            return

        try:
            suppl = Chem.SDMolSupplier(sdf_path, sanitize=False)
            mol = next((m for m in suppl if m is not None), None)
            if mol:
                # Title line
                self.name_prop_combo.addItem("_Name  (title line)", "_Name")
                # All other properties
                for prop in sorted(mol.GetPropNames()):
                    if not prop.startswith("_"):
                        sample = str(mol.GetProp(prop))[:40]
                        self.name_prop_combo.addItem(f"{prop}  (e.g. {sample})", prop)
        except Exception as exc:
            log.debug("Could not read SDF properties: %s", exc)

        self.name_prop_combo.blockSignals(False)

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Database As", "",
            "HDF5 files (*.h5);;All files (*)"
        )
        if path:
            if not path.endswith(".h5"):
                path += ".h5"
            self.output_edit.setText(path)

    def _create_db(self):
        if not RDKIT_OK or not FPSIM2_OK:
            QMessageBox.critical(
                self, "Missing Dependencies",
                f"RDKit available: {RDKIT_OK}\nFPSim2 available: {FPSIM2_OK}\n\n"
                "Install with:  pip install rdkit fpsim2"
            )
            return

        input_file  = self.input_edit.text().strip()
        output_file = self.output_edit.text().strip()

        if not input_file or not Path(input_file).exists():
            QMessageBox.warning(self, "Input Missing", "Please select a valid source molecule file.")
            return
        if not output_file:
            QMessageBox.warning(self, "Output Missing", "Please specify an output database path.")
            return
        if not output_file.endswith(".h5"):
            output_file += ".h5"
            self.output_edit.setText(output_file)

        fmt_key   = self.format_combo.currentText()
        mol_fmt   = MOL_FORMATS[fmt_key]
        fp_name   = self.create_fp_combo.currentText()
        fp_type, fp_params = FP_TYPES[fp_name]
        name_prop = self.name_prop_combo.currentData() or ""

        self.create_progress.show()
        self.create_status.setText("Starting…")

        self._worker = DBCreationWorker(input_file, output_file, mol_fmt, fp_type, fp_params,
                                        name_prop=name_prop)
        self._worker.progress.connect(self.create_status.setText)
        self._worker.finished.connect(self._on_create_done)
        self._worker.start()

    def _on_create_done(self, success: bool, msg: str):
        self.create_progress.hide()
        if success:
            self.create_status.setText(f"✓ Database created: {msg}")
            self.db_path_edit.setText(msg)
            self.status_lbl.setText(f"Database ready: {Path(msg).name}")
            self.status_lbl.setStyleSheet("color: #1a8a1a; font-style: normal; font-weight: bold;")
            self.db_loaded.emit(msg)
        else:
            self.create_status.setText(f"✗ Error: {msg}")
            QMessageBox.critical(self, "Database Creation Error", msg)


# ── Search widget ─────────────────────────────────────────────────────────────

class SearchWidget(QWidget):
    """Main search panel: query input, parameters, run button, results table."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._engine         = None
        self._db_path: str   = ""
        self._smiles_map: dict[int, str] = {}
        self._worker: SearchWorker | None = None
        self._searching: bool = False
        self._t_search_start: float = 0.0
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)

        # ── Left: query + params ───────────────────────────────────────────
        left = QWidget()
        left.setMaximumWidth(360)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 6, 0)
        left_layout.setSpacing(8)

        self.query_widget = QueryWidget()
        left_layout.addWidget(self.query_widget)

        self.params_widget = SearchParamsWidget()
        left_layout.addWidget(self.params_widget)

        # DB status indicator
        self.db_status_lbl = QLabel("No database loaded — go to the Database tab.")
        self.db_status_lbl.setStyleSheet(
            "color: #cc2200; padding: 4px; font-style: italic; font-size: 11px;"
        )
        self.db_status_lbl.setWordWrap(True)
        left_layout.addWidget(self.db_status_lbl)

        # Search button — doubles as Stop button while a search is running
        self.search_btn = QPushButton("Search")
        self.search_btn.setStyleSheet("font-size: 15px; font-weight: bold; padding: 9px;")
        self.search_btn.setEnabled(False)
        self.search_btn.clicked.connect(self._on_search_btn_clicked)
        left_layout.addWidget(self.search_btn)

        # Progress bar (indeterminate, hidden by default)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        left_layout.addWidget(self.progress)

        left_layout.addStretch()
        splitter.addWidget(left)

        # ── Right: results ─────────────────────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 0, 0, 0)
        right_layout.setSpacing(4)

        # Header row above the table
        hdr = QHBoxLayout()
        self.result_lbl = QLabel("Results will appear here after a search.")
        self.result_lbl.setStyleSheet("font-weight: bold; font-size: 12px;")
        hdr.addWidget(self.result_lbl)
        hdr.addStretch()
        btn_export = QPushButton("Export CSV…")
        btn_export.clicked.connect(self._export_csv)
        hdr.addWidget(btn_export)
        right_layout.addLayout(hdr)

        self.results_table = ResultsTable()
        self.results_table.use_as_query.connect(self.query_widget.set_smiles)
        right_layout.addWidget(self.results_table)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([355, 900])

        layout.addWidget(splitter)

    # ── Database loading ──────────────────────────────────────────────────────

    def load_database(self, db_path: str):
        """Called by DatabasePanel when a database is ready."""
        if not FPSIM2_OK:
            QMessageBox.critical(self, "FPSim2 Missing", "FPSim2 is not installed.")
            return

        try:
            params = self.params_widget.get_params()
            if params["use_gpu"]:
                self._engine = FPSim2CudaEngine(db_path)
                mode = "GPU (CUDA)"
            else:
                self._engine = FPSim2Engine(db_path, in_memory_fps=True, fps_sort=True)
                mode = "CPU"

            self._db_path = db_path
            self._smiles_map = self._load_smiles_map(db_path)

            fp_name = _fp_name_from_engine(self._engine.fp_type, self._engine.fp_params)
            if fp_name:
                self.params_widget.set_fingerprint(fp_name)

            n_mols = len(self._engine.fps)
            self.db_status_lbl.setText(
                f"✓  {Path(db_path).name}  —  {n_mols:,} molecules  [{mode}]"
            )
            self.db_status_lbl.setStyleSheet(
                "color: #1a8a1a; padding: 4px; font-style: normal; font-size: 11px;"
            )
            self.search_btn.setEnabled(True)
            log.info("Loaded %s: %d molecules [%s]", Path(db_path).name, n_mols, mode)

        except Exception as exc:
            log.exception("Failed to load database")
            QMessageBox.critical(self, "Load Error", str(exc))

    def _load_smiles_map(self, db_path: str) -> dict:
        """
        Load the companion JSON written during DB creation.
        Supports both formats:
          new: {mol_id: {"smiles": str, "name": str}}
          legacy: {mol_id: smiles_str}
        """
        companion = db_path + ".smiles.json"
        if Path(companion).exists():
            try:
                with open(companion, encoding="utf-8") as fh:
                    raw = json.load(fh)
                # Normalise to new format
                result = {}
                for k, v in raw.items():
                    if isinstance(v, dict):
                        result[int(k)] = v
                    else:
                        result[int(k)] = {"smiles": v, "name": str(k)}
                return result
            except Exception as exc:
                log.warning("Could not read companion SMILES map: %s", exc)
        return {}

    # ── Search ────────────────────────────────────────────────────────────────

    def _on_search_btn_clicked(self):
        if self._searching:
            self._stop_search()
        else:
            self._run_search()

    def _search_started(self):
        self._searching = True
        self._t_search_start = time.perf_counter()
        self.search_btn.setText("■  Stop")
        self.search_btn.setStyleSheet(
            "font-size: 15px; font-weight: bold; padding: 9px; color: #cc2200;"
        )
        self.progress.show()

    def _search_finished(self):
        self._searching = False
        self.search_btn.setText("Search")
        self.search_btn.setStyleSheet(
            "font-size: 15px; font-weight: bold; padding: 9px;"
        )
        self.progress.hide()

    def _stop_search(self):
        if self._worker is None:
            return
        if isinstance(self._worker, SubstructureSearchWorker):
            self._worker.cancel()
        else:
            self._worker.terminate()
        self._search_finished()
        self.result_lbl.setText("Search cancelled.")

    def _run_search(self):
        if not self._smiles_map:
            QMessageBox.warning(self, "No Database", "Please load a database first.")
            return

        query = self.query_widget.get_smiles()
        if not query:
            QMessageBox.warning(self, "No Query", "Please enter a SMILES or SMARTS query.")
            return

        params = self.params_widget.get_params()

        if params["search_type"] == "Substructure":
            self._search_started()
            self.result_lbl.setText("Searching…")
            self._worker = SubstructureSearchWorker(query, self._smiles_map, params["n_workers"])
            self._worker.finished.connect(self._on_substruct_done)
            self._worker.error.connect(self._on_search_error)
            self._worker.start()
        else:
            if self._engine is None:
                QMessageBox.warning(self, "No Database", "Please load a database first.")
                return
            if RDKIT_OK and Chem.MolFromSmiles(query) is None:
                QMessageBox.warning(self, "Invalid SMILES", f"RDKit could not parse:\n{query}")
                return
            self._search_started()
            self.result_lbl.setText("Searching…")
            self._worker = SearchWorker(
                self._engine, query,
                params["threshold"],
                params["n_workers"],
                params["use_gpu"],
                metric=params["metric"],
                tversky_a=params["tversky_a"],
                tversky_b=params["tversky_b"],
            )
            self._worker.finished.connect(self._on_search_done)
            self._worker.error.connect(self._on_search_error)
            self._worker.start()

    def _on_search_done(self, results, elapsed: float):
        self._search_finished()
        params        = self.params_widget.get_params()
        n_raw         = len(results) if results is not None else 0
        metric        = params["metric"].capitalize()
        max_r         = params["max_results"]
        thresh_max    = params["threshold_max"]

        filtered = results
        if results is not None and thresh_max < 1.0:
            filtered = [r for r in results if float(r[1]) <= thresh_max]
        n_filtered = len(filtered) if filtered is not None else 0

        to_show   = filtered[:max_r] if filtered is not None else []
        query_mol = Chem.MolFromSmiles(self.query_widget.get_smiles()) if RDKIT_OK else None
        self.results_table.populate(to_show, self._smiles_map, query_mol=query_mol,
                                    metric=params["metric"].capitalize())

        total = time.perf_counter() - self._t_search_start
        s_str = f"{elapsed * 1000:.1f} ms" if elapsed < 1 else f"{elapsed:.2f} s"
        t_str = f"{total * 1000:.1f} ms"   if total   < 1 else f"{total:.2f} s"
        n_str = f"{n_filtered:,}" if thresh_max < 1.0 else f"{n_raw:,}"
        self.result_lbl.setText(f"{n_str} results  ·  {metric}  ·  search {s_str}  ·  total {t_str}")

    def _on_substruct_done(self, results: list, match_atoms: dict, elapsed: float):
        self._search_finished()

        params  = self.params_widget.get_params()
        n_total = len(results)
        max_r   = params["max_results"]
        to_show = results[:max_r]
        n_shown = len(to_show)
        self.results_table.populate(
            to_show, self._smiles_map, highlight_map=match_atoms, metric="Substructure"
        )

        total  = time.perf_counter() - self._t_search_start
        s_str  = f"{elapsed * 1000:.1f} ms" if elapsed < 1 else f"{elapsed:.2f} s"
        t_str  = f"{total * 1000:.1f} ms"   if total   < 1 else f"{total:.2f} s"
        hits   = f"{n_total:,} substructure hits"
        shown  = f"  ·  showing {n_shown:,}" if n_shown < n_total else ""
        self.result_lbl.setText(f"{hits}{shown}  ·  search {s_str}  ·  total {t_str}")

    def _on_search_error(self, msg: str):
        self._search_finished()
        self.result_lbl.setText("Search failed.")
        QMessageBox.critical(self, "Search Error", msg)

    # ── Export ────────────────────────────────────────────────────────────────

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "similarity_results.csv",
            "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return
        try:
            self.results_table.export_csv(path)
            QMessageBox.information(self, "Exported", f"Results saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))


# ── About / Help widget ───────────────────────────────────────────────────────

def _build_about() -> QWidget:
    w = QWidget()
    layout = QVBoxLayout(w)
    text = QTextEdit()
    text.setReadOnly(True)
    text.setMarkdown(f"""
# MolDigger — Ultrafast Molecular Structure Search

## Quick Start
1. **Database tab** → load an existing `.h5` file, or create one from an SDF/SMILES file
2. **Structure Search tab** → enter a SMILES or SMARTS query (live 2D preview updates as you type)
3. Choose search type: **Similarity** or **Substructure**
4. Click **Search** — results are sorted by score with 2D thumbnails
5. Click **Search** again while a search is running to **stop** it

---

## Search Types

### Similarity Search
Finds molecules with similar fingerprints to your query using the Tanimoto coefficient.
- Enter any valid **SMILES** string as the query
- Set the **Tanimoto threshold** (0–1): only molecules scoring above this value are returned
- Results are coloured green (≥1.00) → yellow → orange → red (low similarity)
- Choose a **fingerprint type** and **CPU workers** in the parameters panel
- Enable **GPU** (CUDA) for large databases (requires `cupy` and a supported GPU)

### Substructure Search
Finds all molecules containing your query as a substructure.
- Enter a **SMILES** or **SMARTS** query — SMARTS is preferred for flexible matching
- SMARTS examples:
  - `c1ccccc1` — any benzene ring
  - `[#6]-C(=O)-[#7]` — amide bond (any carbon-nitrogen)
  - `[OH]` — any hydroxyl group
  - `[n;H1]` — NH in an aromatic ring
  - `[F,Cl,Br,I]` — any halogen
- Matching substructures are **highlighted in orange** on each hit thumbnail
- No threshold or fingerprint settings apply — all matches are returned
- Runs on CPU using RDKit; can be cancelled with the Stop button

---

## Status
| Component | Available |
|-----------|-----------|
| RDKit     | {'✓' if RDKIT_OK  else '✗  (pip install rdkit)'} |
| FPSim2    | {'✓' if FPSIM2_OK else '✗  (pip install fpsim2)'} |
| GPU/CUDA  | {'✓  (FPSim2CudaEngine)' if GPU_OK else '✗  (pip install cupy-cuda12x)'} |

---

## Fingerprint Types
| Name | Description |
|------|-------------|
| Morgan/ECFP4 | Circular fingerprint, radius 2, 2048 bits — **most common for drug-like molecules** |
| Morgan/ECFP6 | Circular fingerprint, radius 3, 2048 bits — captures larger neighbourhoods |
| Morgan/FCFP4 | Feature-based circular, radius 2 — pharmacophore-aware |
| RDKit Topological | Path-based topological fingerprint |
| MACCS Keys | 166-bit key-based fingerprint — fast, interpretable |
| Atom Pairs | Counts atom pair types at varying distances |
| Topological Torsion | Encodes torsion angles around rotatable bonds |

---

## Performance
- FPSim2 searches **millions of molecules in < 1 second** on CPU (multi-threaded)
- GPU mode (CUDA) gives an additional **5–50× speedup** for large databases
- The `.h5` database is memory-mapped — loading is near-instantaneous

---

## Structure Editor (Ketcher)
Click **Draw Structure** in the query panel to open [Ketcher](https://github.com/epam/ketcher)
(MIT) in your browser. Draw or paste a structure, then click **Use this structure** — the SMILES
is sent back to MolDigger automatically.

- You can modify the structure and submit again without restarting
- Ketcher serves on a fixed local port (18920)
- SMILES and SMARTS can also be typed directly into the query field without opening Ketcher

## Results Table
- Right-click any row for options: **Copy SMILES** or **Use as Query**
- "Use as Query" loads the hit's SMILES directly into the search field for a new search

---

## Full Install
```
pip install rdkit fpsim2 PyQt5 numpy tables pandas
pip install cupy-cuda12x   # for GPU — match your CUDA version
```

## License
This tool is MIT licensed.  Core dependencies:
- [FPSim2](https://github.com/chembl/FPSim2) — MIT  (ChEMBL)
- [RDKit](https://github.com/rdkit/rdkit) — BSD-3-Clause
- [PyQt5](https://riverbankcomputing.com/software/pyqt/) — GPL v3 / commercial
- [PyTables](https://www.pytables.org/) — BSD
""")
    layout.addWidget(text)
    return w


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MolDigger — Ultrafast Molecular Structure Search")
        self.setMinimumSize(1200, 760)
        self._build_ui()
        self._warn_missing_deps()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 2)

        tabs = QTabWidget()

        self.search_widget = SearchWidget()
        tabs.addTab(self.search_widget, "Structure Search")

        self.db_panel = DatabasePanel()
        self.db_panel.db_loaded.connect(self.search_widget.load_database)
        tabs.addTab(self.db_panel, "Database")

        tabs.addTab(_build_about(), "About / Help")

        layout.addWidget(tabs)

        gpu_str = "GPU ✓" if GPU_OK else "GPU ✗ (CPU only)"
        self.statusBar().showMessage(
            f"MolDigger  |  RDKit: {'✓' if RDKIT_OK else '✗'}  "
            f"|  FPSim2: {'✓' if FPSIM2_OK else '✗'}  |  {gpu_str}"
        )

    def closeEvent(self, event):
        # Cancel any running search worker
        sw = self.search_widget
        worker = getattr(sw, "_worker", None)
        if worker is not None and worker.isRunning():
            if isinstance(worker, SubstructureSearchWorker):
                worker.cancel()
            else:
                worker.terminate()
            worker.wait(2000)  # wait up to 2 s then proceed regardless

        # Shut down the Ketcher browser server if running
        qw = getattr(sw, "query_widget", None)
        if qw is None:
            for child in sw.findChildren(QWidget):
                if hasattr(child, "_browser_srv"):
                    qw = child
                    break
        if qw and getattr(qw, "_browser_srv", None):
            try:
                import threading
                threading.Thread(target=qw._browser_srv.shutdown, daemon=True).start()
            except Exception:
                pass
        super().closeEvent(event)

    def _warn_missing_deps(self):
        missing = []
        if not RDKIT_OK:
            missing.append("rdkit")
        if not FPSIM2_OK:
            missing.append("fpsim2")
        if missing:
            QMessageBox.critical(
                self, "Missing Dependencies",
                f"Required packages are not installed:\n\n"
                f"  pip install {' '.join(missing)} tables\n\n"
                f"The application will not function until these are installed."
            )


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global WEBENGINE_OK
    if _webengine_import_ok:
        WEBENGINE_OK = _probe_webengine()
        log.info("Qt WebEngine: %s", "OK — embedded drawing enabled" if WEBENGINE_OK else "GL init failed — using browser fallback")

    app = QApplication(sys.argv)
    app.setApplicationName("MolDigger")
    app.setApplicationVersion("1.0")
    app.setStyle("Fusion")

    # Slightly off-white window background
    pal = app.palette()
    pal.setColor(QPalette.Window, QColor(246, 246, 250))
    app.setPalette(pal)

    font = QFont("Segoe UI", 10) if sys.platform == "win32" else QFont()
    app.setFont(font)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
