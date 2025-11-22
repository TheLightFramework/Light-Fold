# Light-Fold: Deterministic Protein Folding via Negentropic Vectors

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Python](https://img.shields.io/badge/Python-3.x-blue.svg) ![Status](https://img.shields.io/badge/Status-Proof_of_Concept-green)

> **"Structure is the physical manifestation of Alignment."**

[begin](mid.png)

## 🧬 Abstract
Light-Fold is a coarse-grained, implicit-solvent simulation engine derived from first principles of **Information Geometry** and **Thermodynamics**.

Instead of using brute-force Molecular Dynamics (MD) to simulate solvent particle collisions, Light-Fold utilizes a directed **Vector Field approach**. It posits that the "Native State" of a protein is the geometric centroid where:
1.  **Hydrophobic Core Modularity** is maximized.
2.  **Solvent Entropy** is maximized (Water is repelled).
3.  **Steric Conflicts** are minimized.

This allows for the folding of simple chains in **milliseconds** rather than hours, running on standard CPU architecture.

## 📐 The Core Equation
The simulation minimizes the following energy landscape $E(G)$ without gradient descent, using direct vector integration:

$$ E(G) = \sum_{i \in H} A_i(G) - \lambda \sum_{(i,j) \in H\times H} C_{ij}(G) $$

Where:
*   $H$: Hydrophobic Residues (The Core).
*   $A$: Solvent Accessible Surface Area (Entropy Cost).
*   $C$: Contact Matrix (Coherence/Negentropy).
*   $\lambda$: Interaction Strength.

## 🛠️ Usage
Light-Fold is a single-file Python implementation requiring only `numpy` and `matplotlib`.

```bash
pip install numpy matplotlib
python light_fold.py
```

## 🔬 Implications for Research
While currently a topological toy model (HP-Lattice derivative), this framework demonstrates that protein folding need not be treated as a stochastic search through a random landscape. It suggests the existence of a **Deterministic Tunnel** (The Funnel Hypothesis) guided by a computable vector field.

We invite researchers in biophysics and AI to analyze the source code.

## 🤝 Credits
Developed by The Light Framework Team.
*Optimizing for Truth, Clarity, and Goodness.*
