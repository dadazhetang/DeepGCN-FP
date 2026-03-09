from rdkit import Chem
from typing import Dict, List, Tuple
import csv
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR

FUNCTIONAL_GROUP_SMARTS = {
    "Alcohol": "[CX4][OX2H]",
    "Phenol": "[cX3][OX2H]",
    "Ether": "[CX4][OX2][CX4]",
    "Aldehyde": "[CX3H1](=[OX1])[#6]",
    "Ketone": "[#6][CX3](=[OX1])[#6]",
    "Carboxylic_Acid": "[CX3](=[OX1])[OX2H]",
    "Ester": "[CX3](=[OX1])[OX2][#6]",
    "Amide": "[N,n][CX3]=[OX1]",

    "Amine_Primary": "[NX3H2][#6]",
    "Amine_Secondary": "[NX3H1]([#6])[#6]",
    "Amine_Tertiary": "[NX3]([#6])([#6])[#6]",
    "Nitrile": "[CX2]#[NX1]",
    "Nitro": "[NX3+](=[OX1])[OX1-]",
    "Nitroso": "[NX2]=[OX1]",

    "Thiol": "[CX4][SX2H]",
    "Sulfide": "[CX4][SX2][CX4]",
    "Disulfide": "[SX2][SX2]",

    "Alkene": "[CX3]=[CX3]",
    "Alkyne": "[CX2]#[CX2]",
    "Conjugated_Diene": "[CX3]=[CX3][CX3]=[CX3]",

    "Halogen": "[F,Cl,Br,I]",
    "Phenyl": "c1ccccc1",
}

def extract_functional_groups(smiles: str, smarts_dict: Dict[str, str] = None) -> List[Dict]:
    if smarts_dict is None:
        smarts_dict = FUNCTIONAL_GROUP_SMARTS

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string：{smiles}")

    Chem.SanitizeMol(mol)

    results = []

    for fg_name, smarts_pattern in smarts_dict.items():
        query_mol = Chem.MolFromSmarts(smarts_pattern)
        if query_mol is None:
            print(f"Warning: Invalid SMARTS pattern '{smarts_pattern}' corresponding functional group '{fg_name}'")
            continue
        matches = mol.GetSubstructMatches(query_mol)
        for match in matches:
            results.append({
                "name": fg_name,
                "atoms": match,
                "smarts": smarts_pattern
            })
    return results

if __name__ == "__main__":
    smiles_path = PROJECT_ROOT/"MminH_train.txt"
    with open(smiles_path, "rt", encoding="utf-8", errors='ignore') as csvfile:
        reader = csv.reader((line.replace('\0', '') for line in csvfile), delimiter='\t')
        output = [row for row in reader]
        output = output[1:]
        all_mols = []
        for i in range(len(output)):
            all_mols.append(["", output[i][0],output[i][1]])

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
    all_temp = []
    all_we_need = []
    for mm in all_mols:
        test_smiles = mm[1]
        print(f"Analyzing molecule：{test_smiles}")
        try:
            fg_list = extract_functional_groups(test_smiles)
            if len(fg_list) ==0:
                all_temp.append(mm)
            print(f"\nFound {len(fg_list)} functional group matches：")
            print("-" * 40)
            mol = Chem.MolFromSmiles(test_smiles)
            for i, fg in enumerate(fg_list):
                atom_indices = fg['atoms']
                atom_symbols = [mol.GetAtomWithIdx(idx).GetSymbol() for idx in atom_indices]
                print(f"{i + 1}. {fg['name']}")
                print(f"   SMARTS: {fg['smarts']}")
                print(f"   RDKit atom index：{atom_indices}")
                print(f"   Corresponding atom symbol：{atom_symbols}")
                print("-" * 40)

            highlight_atoms = set()
            for fg in fg_list:
                highlight_atoms.update(fg['atoms'])
            all_we_need.append([fg_list,mm[4]])

        except Exception as e:
            print(f"An error occurred：{e}")
    with open(PROJECT_ROOT/"Functional group results.json", "w", encoding="utf-8") as f:
        json.dump(all_we_need, f, ensure_ascii=False, indent=2)
