# MolDigger — Ultrafast Molecular Structure Searching & Clustering

MolDigger is a molecular structure search and clustering tool available as both a **PyQt5 desktop app** and a **browser-based web app**. It uses [FPSim2](https://github.com/chembl/FPSim2) for fingerprint-based similarity search (screening millions of compounds in milliseconds) and [RDKit](https://www.rdkit.org/) for substructure matching, Butina clustering, 2D depiction, and property calculation.

![MolDigger screenshot](moldigger.png)

---

## Features

- **Similarity search** — Tanimoto, Dice, and Tversky metrics with adjustable min/max threshold range
- **Substructure search** — SMILES or SMARTS queries with multi-threaded RDKit matching
- **Clustering** — Butina clustering of search results with adjustable similarity cutoff; cluster ID column added to results table; always uses Morgan ECFP4 for best chemical groupings
- **Atom highlighting** — MCS highlighted in similarity hits; matched atoms highlighted in substructure hits (optional, uses RDKit default highlight colour)
- **Named lists with boolean combinators** — save hits as a list (all, selected rows, or imported SMILES), then combine lists with AND / OR / NOT / XOR
- **GPU acceleration** — NVIDIA CUDA via FPSim2's CudaEngine (Tanimoto only)
- **Multiple fingerprint types** — Morgan/ECFP4, ECFP6, RDKit Topological, MACCS Keys, Atom Pairs, Topological Torsion
- **Multi-FP databases** — build a single source with several FP types in one job; the siblings are bundled into a `.fpset` directory that loads as a single logical database, with a live FP-switcher in the search panel
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
3. Tick one or more fingerprint types and pick an output path
4. Click **Create Database**

The app writes a `.h5` FPSim2 database and a companion `.h5.smiles.json` file that stores SMILES strings and molecule names for display in the results table.

### Multi-FP databases

A single FPSim2 `.h5` file holds one fingerprint type. If you tick more than one FP at build time, MolDigger collects the sibling files into a `.fpset` directory and writes a manifest so the set can be loaded as a single logical database:

```
chembl.fpset/                      ← the alias entry point
  manifest.json
  chembl.morgan_ecfp4.h5           + companion .smiles.json
  chembl.maccs.h5                  + companion .smiles.json
  chembl.rdkit_topological.h5      + companion .smiles.json
```

The file browser shows the `.fpset` directory as a single 📦 entry. Loading it opens every FP engine in memory and the **Fingerprint** dropdown in the search panel becomes a live switcher — picking a different FP swaps the active similarity engine without reloading the database. Substructure search and clustering are FP-independent and work the same across the set.

You can tick any combination — there is no hard limit; one build produces one sibling per ticked FP. A reasonable default triad is **Morgan/ECFP4 + RDKit Topological + MACCS Keys**: local-circular substructures, path-based topology, and interpretable structural keys respectively (see the *Fingerprint Types* table below). FPSim2's wrapper around RDKit's Morgan generator does not support feature-based / FCFP-style fingerprints, so feature/pharmacophore-aware Morgan is not available.

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

#### Note on query semantics

MolDigger uses RDKit's substructure matcher, which is **strict about aromaticity and ring topology**. An aliphatic `N` in your query does not match an aromatic `n` in a target, and a benzene `c1ccccc1` does not match the 6-membered half of a fused indole — even though tools like DataWarrior would consider both a match. This is a difference in matching semantics, not a bug.

If a query is rejecting hits you expect, relax the strict atoms by using SMARTS. For example, the SMILES query

```
O=S(c(cc1)ccc1N1CCNCC1)(Nc1ccccc1)=O
```

requires the second nitrogen's ring to be benzene, so molecules whose sulfonamide nitrogen is part of an indole are missed. Rewriting it as SMARTS without the aniline-ring constraint widens the search to those hits:

```
O=S(=O)(c1ccc(N2CCNCC2)cc1)[#7]
```

In general: use `[#7]` to match any nitrogen regardless of aromaticity, `[#6]` for any carbon, and drop ring-closure constraints where you don't care about the surrounding ring.

---

## Clustering

After a search, a **Cluster** toolbar appears above the results table. Clustering is decoupled from search — you can adjust the cutoff and re-cluster without repeating the database search:

1. Set **Min similarity** (0.10–0.90, default 0.40) — molecules with ≥ this similarity will tend to be grouped together
2. Click **Apply Clustering** — a **Cluster** column appears and results are sorted by cluster ID (click the column header to toggle ascending/descending)
3. Click **Clear** to remove clustering and restore score order

Clustering always uses **Morgan ECFP4** fingerprints (radius=2, 2048 bits) regardless of the fingerprint type used for the similarity search, as Morgan ECFP4 gives the best chemical groupings for diverse compound sets.

The underlying algorithm is Butina clustering (`rdkit.ML.Cluster.Butina`), which is a standard single-pass, sphere-exclusion method widely used in cheminformatics.

---

## Lists

Named **lists** of molecule IDs can be saved per-database and combined with boolean operators. Useful for stacking searches: e.g. "compounds matching scaffold A *and not* a known toxicophore".

### Creating lists

Three ways:

1. **Save all results** — after any search, click **★ Save all as list** in the results header. Prompts for a name; saves every hit currently shown.
2. **Save selected rows** — tick the checkboxes in the first column of the results table (or the header box for "select all visible"), then click **★ Save selected as list**.
3. **Import from SMILES** — paste a list of SMILES into the Lists card (one per line) and click **Import**. Each line is canonicalized via RDKit and looked up in the loaded database; unmatched entries are reported in the status message.

### Combining lists

In the Lists card's combiner:

1. Pick an operator (`AND`, `OR`, `NOT`, `XOR`) and a list, then click **Add** to push a step onto the expression.
2. Repeat to build a sequence. Steps evaluate left-to-right against an accumulator.
3. Click **Run** — the result loads into the main results table, where it can be sorted, clustered, exported, or saved as a new list.

**NOT semantics:** if `NOT` is the *first* step, the accumulator starts as the full database minus the named list. Otherwise `NOT` is a set-subtract from the accumulator. So `[OR A, NOT B]` means *A minus B*, and `[NOT A]` means *everything except A*.

### Storage

Lists are stored in `~/.moldigger/lists.json`, keyed by the resolved absolute path of each database file. The lists shown in the UI are scoped to the currently-loaded database. Switching databases shows a different set of lists; the file itself is shared across all databases on the host.

---

## Fingerprint Types

| Name | FPSim2 type | Notes |
|------|-------------|-------|
| Morgan / ECFP4 | Morgan, radius=2 | Most common for drug-like molecules |
| Morgan / ECFP6 | Morgan, radius=3 | Larger neighbourhood |
| RDKit Topological | RDKit | Path-based |
| MACCS Keys | MACCSKeys | 166-bit, interpretable |
| Atom Pairs | AtomPair | Counts atom-pair types |
| Topological Torsion | TopologicalTorsion | Encodes torsion angles |

The fingerprint type is automatically detected from the loaded `.h5` file.

---

## Performance

- FPSim2 screens **millions of molecules in < 1 second** on CPU (multi-threaded)
- GPU mode (CUDA) provides an additional **5–50× speedup** for large databases
- Substructure search caches parsed RDKit Mols and `PatternFingerprint`s at DB-load time, then uses fingerprint superset pre-screening before running `GetSubstructMatch`. On a 137k-molecule database this drops a typical query from ~20 s to **well under 1 s**.
- The substructure cache is persisted to disk next to the `.h5` file as `<db>.subcache.bin` (≈ 1 KB / molecule). Subsequent loads of the same database deserialize from disk in seconds instead of rebuilding.
- The `.h5` database itself is memory-mapped — loading is near-instantaneous
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
