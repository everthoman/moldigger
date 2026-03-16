# MolDigger — Ultrafast Molecular Structure Searching & Clustering

MolDigger is a molecular structure search and clustering tool available as both a **PyQt5 desktop app** and a **browser-based web app**. It uses [FPSim2](https://github.com/chembl/FPSim2) for fingerprint-based similarity search (screening millions of compounds in milliseconds) and [RDKit](https://www.rdkit.org/) for substructure matching, Butina clustering, 2D depiction, and property calculation.

![MolDigger screenshot](moldigger.png)

---

## Features

- **Similarity search** — Tanimoto, Dice, and Tversky metrics with adjustable min/max threshold range
- **Substructure search** — SMILES or SMARTS queries with multi-threaded RDKit matching
- **Clustering** — Butina clustering of search results with adjustable similarity cutoff; cluster ID column added to results table; always uses Morgan ECFP4 for best chemical groupings
- **Atom highlighting** — MCS highlighted in similarity hits; matched atoms highlighted in substructure hits (optional, uses RDKit default highlight colour)
- **GPU acceleration** — NVIDIA CUDA via FPSim2's CudaEngine (Tanimoto only)
- **Multiple fingerprint types** — Morgan/ECFP4, ECFP6, FCFP4, RDKit Topological, MACCS Keys, Atom Pairs, Topological Torsion
- **Auto-detects fingerprint type** from the loaded database file
- **Structure editor** — [Ketcher](https://github.com/epam/ketcher) launched in browser; drawn structures sent back to the app automatically
- **Results table** — sortable, with 2D thumbnails, MW, ClogP; right-click to copy SMILES or use hit as new query
- **Stop button** — cancel any running search mid-way
- **Database builder** — create FPSim2 `.h5` databases from SDF or SMILES files within the app
- **Export** — save results to CSV

---

## Apps

### Desktop app — `moldigger.py`

Full-featured PyQt5 GUI. Run with:

```bash
conda activate moldigger
python moldigger.py
```

### Web app — `moldigger_web.py`

Browser-based interface built with FastAPI and uvicorn. Useful on headless servers or when a native GUI is not available.

```bash
conda activate moldigger
python moldigger_web.py
# opens http://localhost:8000 in your browser
```

Both apps share the same database format and support the same search, clustering, and highlighting features.

---

## Installation

### Requirements

- Python 3.10+
- Conda (recommended) or pip

### Conda (recommended)

```bash
conda create -n moldigger python=3.11
conda activate moldigger
conda install -c conda-forge rdkit
pip install fpsim2 PyQt5 numpy tables
```

### Web app only (no GUI dependencies needed)

```bash
pip install fastapi uvicorn
```

### GPU support (optional)

Requires an NVIDIA GPU with CUDA installed. Match the `cupy` version to your CUDA installation:

```bash
pip install cupy-cuda12x   # for CUDA 12.x
# or
pip install cupy-cuda11x   # for CUDA 11.x
```

GPU availability is detected automatically at startup. If available, a **Use GPU** checkbox appears in the search parameters panel.

---

## Quick Start

1. **Database tab** → load an existing `.h5` file, or build one from an SDF/SMILES file
2. **Structure Search tab** → type a SMILES or SMARTS query (live 2D preview updates as you type)
3. Choose **Search type** (Similarity or Substructure), fingerprint, metric, and threshold range
4. Click **Search** — results appear sorted by score with 2D thumbnails
5. Optionally click **Apply Clustering** above the results table to group hits by structural similarity
6. Click **Search** again while running to **stop** it

---

## Building a Database

1. Go to the **Database** tab
2. Select an input file (SDF or SMILES)
3. Choose fingerprint type and output path
4. Click **Create Database**

The app writes a `.h5` FPSim2 database and a companion `.h5.smiles.json` file that stores SMILES strings and molecule names for display in the results table.

---

## Search Types

### Similarity Search

Finds molecules with similar fingerprints to the query using a chosen metric:

| Metric | Description |
|--------|-------------|
| **Tanimoto** | Standard Jaccard similarity — most common in cheminformatics |
| **Dice** | 2·&#124;A∩B&#124; / (&#124;A&#124;+&#124;B&#124;) — gives higher scores than Tanimoto |
| **Tversky** | Asymmetric; α=1, β=0 finds larger molecules containing your scaffold |

Set the **Min** and **Max** threshold sliders to control the score range returned. Results are colour-coded green (score = 1.00) → yellow → orange → red (low similarity). The MCS (maximum common substructure) between the query and each hit is highlighted in the 2D thumbnail.

### Substructure Search

Finds all molecules containing the query as a substructure. Accepts:
- **SMILES** — exact substructure match
- **SMARTS** — flexible pattern matching, e.g.:
  - `c1ccccc1` — any benzene ring
  - `[#6]-C(=O)-[#7]` — amide bond
  - `[F,Cl,Br,I]` — any halogen
  - `[n;H1]` — NH in an aromatic ring

Matched atoms are highlighted in the hit thumbnails. Runs on CPU using all configured worker threads.

---

## Clustering

After a search, a **Cluster** toolbar appears above the results table. Clustering is decoupled from search — you can adjust the cutoff and re-cluster without repeating the database search:

1. Set **Min similarity** (0.10–0.90, default 0.40) — molecules with ≥ this similarity will tend to be grouped together
2. Click **Apply Clustering** — a **Cluster** column appears and results are sorted by cluster ID
3. Click **Clear** to remove clustering and restore score order

Clustering always uses **Morgan ECFP4** fingerprints (radius=2, 2048 bits) regardless of the fingerprint type used for the similarity search, as Morgan ECFP4 gives the best chemical groupings for diverse compound sets.

The underlying algorithm is Butina clustering (`rdkit.ML.Cluster.Butina`), which is a standard single-pass, sphere-exclusion method widely used in cheminformatics.

---

## Fingerprint Types

| Name | FPSim2 type | Notes |
|------|-------------|-------|
| Morgan / ECFP4 | Morgan, radius=2 | Most common for drug-like molecules |
| Morgan / ECFP6 | Morgan, radius=3 | Larger neighbourhood |
| Morgan / FCFP4 | Morgan, radius=2 | Feature-based (pharmacophore-aware) |
| RDKit Topological | RDKit | Path-based |
| MACCS Keys | MACCSKeys | 166-bit, interpretable |
| Atom Pairs | AtomPair | Counts atom-pair types |
| Topological Torsion | TopologicalTorsion | Encodes torsion angles |

The fingerprint type is automatically detected from the loaded `.h5` file.

---

## Performance

- FPSim2 screens **millions of molecules in < 1 second** on CPU (multi-threaded)
- GPU mode (CUDA) provides an additional **5–50× speedup** for large databases
- Substructure search is parallelised across all CPU workers using `ThreadPoolExecutor`
- The `.h5` database is memory-mapped — loading is near-instantaneous
- The results label reports both the **search time** (FPSim2/RDKit computation) and **total time** (including 2D rendering)

---

## Structure Editor (Ketcher)

MolDigger integrates [Ketcher](https://github.com/epam/ketcher) (MIT) as a structure editor. Click **Draw Structure** to open Ketcher in your browser. Draw or paste a structure, then click **Use this structure** — the SMILES is sent back to MolDigger automatically. You can modify and resubmit without restarting.

> **Note for WSL2 users:** Qt WebEngine (embedded browser) does not work reliably under WSL2 due to OpenGL/GLX limitations. MolDigger automatically detects this and falls back to launching Ketcher in your system browser on a local port (18920). On native Linux or Windows this limitation does not apply.

---

## Dependencies

| Package | Purpose | License |
|---------|---------|---------|
| [FPSim2](https://github.com/chembl/FPSim2) | Fingerprint similarity search | MIT |
| [RDKit](https://www.rdkit.org/) | Cheminformatics, substructure search, clustering, depiction | BSD-3-Clause |
| [PyQt5](https://riverbankcomputing.com/software/pyqt/) | Desktop GUI framework | GPL v3 / commercial |
| [FastAPI](https://fastapi.tiangolo.com/) | Web app framework | MIT |
| [uvicorn](https://www.uvicorn.org/) | ASGI server for web app | BSD |
| [PyTables](https://www.pytables.org/) | HDF5 I/O | BSD |
| [NumPy](https://numpy.org/) | Array operations | BSD |
| [CuPy](https://cupy.dev/) *(optional)* | GPU array library for CUDA | MIT |
| [Ketcher](https://github.com/epam/ketcher) *(optional)* | Structure editor | MIT |

---

## License

MIT
