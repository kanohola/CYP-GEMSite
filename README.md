# CYP-GEMSite

CYP-GEMSite is a geometry-aware and edge-enhanced Graph Transformer model designed for the accurate prediction of Cytochrome P450 (CYP450)-mediated Sites of Metabolism (SoMs).

## 🧠 Model Architecture

The model framework integrates three key components to predict potential SoMs:

**Edge-Enhanced Backbone:** Incorporates a "Dynamic Edge Update" mechanism within the Graph Transformer blocks to explicitly model the electronic evolution of chemical bonds (e.g., bond cleavage) during metabolic reactions.

**Geometry-Aware Global Layer:** Features a "Global Transformer Layer" that fuses Shortest Path Distance (SPD) with 3D spatial information (encoded via Gaussian RBF) to effectively capture long-range steric hindrance and conformational constraints.

**Multi-Scale Readout:** The final prediction aggregates fine-grained node features, local edge contexts, and global molecular representations to ensure high precision.

## 🚀 Usage

You can use the provided script `Predict_by_CYP-GEMSite.py` to predict SoMs for new molecules. The script handles 3D conformer generation and feature extraction automatically.
Environment configuration details can be found in env.txt. Python version is 3.9.20.

### Command Example

```bash
python Predict_by_CYP-GEMSite.py \
  --model ./Model/CYP-GEMSite.pt \
  --input ./Data/External_test_set.sdf \
  --output_dir ./results 
```
### Arguments
* `--model`: Path to the pre-trained model weights (`.pt` file).
* `--input`: Input file path. Supports `.sdf` files or a single SMILES string.
* `--output_dir`: Directory to save prediction results (CSV tables and visualized PNG images).
* `--threshold`: Probability threshold for identifying a Site of Metabolism (Default: 0.6452).

## 📂 Datasets

* **External Test Set:** The independent external validation set for this project is located at `External_test_set.sdf`.
* **Training Data Source:** The `zaretzki_preprocessed.sdf` file used in this study is derived from the work of Chen et al. and is available at: [https://github.com/molinfo-vienna/FAME.AL](https://github.com/molinfo-vienna/FAME.AL).
