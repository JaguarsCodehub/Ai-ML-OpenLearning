import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_graph(smiles):
    """Convert SMILES to molecular graph features"""
    mol = Chem.MolFromSmiles(smiles)
    
    # Get adjacency matrix
    adj = Chem.GetAdjacencyMatrix(mol)
    
    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            atom.GetIsAromatic(),
        ]
        atom_features.append(features)
        
    return {
        'adjacency': torch.tensor(adj, dtype=torch.float),
        'node_features': torch.tensor(atom_features, dtype=torch.float)
    }