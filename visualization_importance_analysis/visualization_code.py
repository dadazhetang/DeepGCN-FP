from joblib import dump, load
import joblib
import torch
import csv
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from models import GCNModelWithEdgeAFPreadout

from matplotlib import cm
from dataset import get_node_dim, get_edge_dim, smiles2graph, feature_to_dgl_graph
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR
print(PROJECT_ROOT)
result_path = os.path.join(PROJECT_ROOT,'visualization_pics')
os.makedirs(result_path, exist_ok=True)

model_path = PROJECT_ROOT/'best_model_fold_0.joblib'
best_model = load(model_path)
data = PROJECT_ROOT/"data0_mh.joblib"
label = PROJECT_ROOT/"label0_mh.joblib"
all_data = joblib.load(data)
all_labels = joblib.load(label)
GCN_fp = all_data[:, -200:]
predictions = best_model.predict_proba(GCN_fp, raw=False)[:, 1]
smiles_path = PROJECT_ROOT/"MminH_train.txt"


with open(smiles_path, "rt", encoding="utf-8", errors='ignore') as csvfile:
    reader = csv.reader((line.replace('\0', '') for line in csvfile), delimiter='\t')
    output = [row for row in reader]
    output = output[1:]
    all_mols = []
    for i in range(len(output)):
        all_mols.append(["", output[i][0], predictions[i]])

for i in all_mols:
    mol = Chem.MolFromSmiles(i[1])
    if mol is None:
        raise ValueError(f"Invalid SMILES: {i[1]}")
    i.append(mol)

all_standard = PROJECT_ROOT/"standard_dataset.csv"
with open(all_standard, "rt", encoding="utf-8", errors='ignore') as csvfile:
    reader = csv.reader((line.replace('\0', '') for line in csvfile), delimiter=",")
    output_standard = [row for row in reader]

for i in all_mols:
    matched = False
    for k in output_standard:
        if i[1] == k[0]:
            i.append(k[1])
            matched = True
            break
    if not matched:
        i.append("Unknown")

def get_node_and_edge_mask_indices(graph, exclude_node_idx):
    edge_mask = np.any(graph["edge_index"] == exclude_node_idx, axis=0)
    return np.where(edge_mask)[0]

model = GCNModelWithEdgeAFPreadout(
    node_in_dim=get_node_dim(),
    edge_in_dim=get_edge_dim(),
    hidden_feats=[200] * 16,
    dropout=0.1
)
model.load_state_dict(torch.load(PROJECT_ROOT/"MH.pth"))
model.eval()
print("yes2")
all_new_fp = []
for num, mol_info in enumerate(all_mols):
    mol = mol_info[3]
    num_atoms = mol.GetNumAtoms()
    base_graph = smiles2graph(mol_info[1], exclude_node=None, exclude_edge=None)
    if "num_nodes" not in base_graph:
        base_graph["num_nodes"] = base_graph["node_feat"].shape[0]
    required_keys = ["node_feat", "edge_feat", "edge_index", "num_nodes"]
    missing = [k for k in required_keys if k not in base_graph]
    if missing:
        raise KeyError(f"Base graph missing keys: {missing} for molecule {num}")
    masked_representations = []
    for atom_idx in range(num_atoms):
        graph_copy = {
            "node_feat": base_graph["node_feat"].copy(),
            "edge_feat": base_graph["edge_feat"].copy(),
            "edge_index": base_graph["edge_index"].copy(),
            "num_nodes": base_graph["num_nodes"]
        }
        graph_copy["node_feat"][atom_idx, :] = 0
        edge_indices = get_node_and_edge_mask_indices(graph_copy, atom_idx)
        graph_copy["edge_feat"][edge_indices, :] = 0
        g = feature_to_dgl_graph(graph_copy)
        _, readout, _ = model(g, mol_info[0])
        masked_representations.append(readout)
    all_new_fp.append(masked_representations)
    print(f"Processed molecule {num + 1}/{len(all_mols)}")

mol_weights = []
for i, mol_info in enumerate(all_mols):
    pred_original = mol_info[2]
    weights = []
    for masked_rep in all_new_fp[i]:
        pred_masked = best_model.predict_proba(masked_rep.detach().numpy(), raw=False)[:, 1][0]
        weights.append(pred_original - pred_masked)
    mol_weights.append(weights)

def get_normalized_weights(weights):
    normalized = []
    for w in weights:
        abs_max = max(abs(x) for x in w) if any(w) else 1.0
        normalized.append([x / abs_max for x in w])
    return normalized

mol_weights = get_normalized_weights(mol_weights)
top_k = 5
filtered_weights = []
for w in mol_weights:
    sorted_indices = sorted(range(len(w)), key=lambda i: abs(w[i]), reverse=True)
    top_indices = set(sorted_indices[:top_k])
    filtered = [w[i] if i in top_indices else 0.0 for i in range(len(w))]
    filtered_weights.append(filtered)

with open(PROJECT_ROOT/"mol_weights.json", "w", encoding="utf-8") as f:
    json.dump(mol_weights, f, ensure_ascii=False, indent=2)
with open(PROJECT_ROOT/"filtered_weights.json", "w", encoding="utf-8") as f:
    json.dump(filtered_weights, f, ensure_ascii=False, indent=2)

def generate_similarity_maps(mols, weights, fp_name):
    mycm = cm.PiYG
    for i, mol in enumerate(mols):
        if mol is None:
            continue
        fig = Draw.MolToMPL(mol, coordScale=1.5, size=(250, 250))
        x, y, z = Draw.calcAtomGaussians(mol, 0.02, step=0.01, weights=weights[i])
        maxscale = max(abs(np.min(z)), abs(np.max(z))) or 1.0
        fig.axes[0].imshow(
            z, cmap=mycm, interpolation='bilinear', origin='lower',
            extent=(0, 1, 0, 1), vmin=-maxscale, vmax=maxscale
        )
        fig.axes[0].contour(x, y, z, 10, colors='k', alpha=0.5)
        title = f"Molecule: {all_mols[i][4]}  Label: {int(all_labels[i])}"
        plt.title(title)
        new_temp_name = os.path.join(PROJECT_ROOT,"visualization_pics",f"mol{i + 1}_{fp_name}.svg")

        fig.savefig(new_temp_name, bbox_inches='tight',facecolor='white',
                   transparent=False)
        plt.close(fig)

print("Generating similarity maps...")
mols_list = [item[3] for item in all_mols]
generate_similarity_maps(mols_list, filtered_weights, 'rf')
print("Done.")

