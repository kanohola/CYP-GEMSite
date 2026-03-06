import os, re, pickle
import networkx as nx
from tqdm import tqdm
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdchem import HybridizationType, ChiralType
from torch_geometric.data import Dataset, Data

SUBTYPES = ['1A2', '2A6', '2B6', '2C8', '2C9', '2C19', '2D6', '2E1', '3A4']

def _one_hot_index(idx: int, num_classes: int) -> torch.Tensor:
    v = torch.zeros(num_classes, dtype=torch.float32)
    if 0 <= idx < num_classes: v[idx] = 1.0
    return v

class V7_Multitask_Dataset(Dataset):
    def __init__(self, root, filename, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.filename = filename
        self.max_spd = 64
        super().__init__(root, transform, pre_transform, pre_filter)
        
        if os.path.exists(self.processed_paths[0]):
            print(f"Loading processed data from {self.processed_paths[0]}...")
            with open(self.processed_paths[0], 'rb') as f:
                self._data_list = pickle.load(f)
        else:
            self._data_list = []

    @property
    def raw_file_names(self): 
        return self.filename

    @property
    def processed_file_names(self): 
        return 'data.pkl' 

    def download(self): 
        pass

    def process(self):
        print(f"Start generating multi-task data (Tasks: {SUBTYPES})...")
        data_list = []
        
        if not os.path.exists(self.raw_paths[0]):
            raise FileNotFoundError(f"Original file not found: {self.raw_paths[0]}")
            
        mols = Chem.SDMolSupplier(self.raw_paths[0], removeHs=True)
        print(f"The original SDF contains the number of molecules: {len(mols)}")

        valid_count = 0
        for i, mol in enumerate(tqdm(mols, desc="Processing")):
            if mol is None: continue
            
            
            try:
                x = self._get_node_features(mol)
                edge_attr = self._get_edge_features(mol)
                edge_index = self._get_adjacency_info(mol)
                
                y = self._get_multitask_labels(mol) 

                pos = self._get_optimized_3d_coords(mol, i)
                spd = self._get_spd_matrix(mol, edge_index)

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                            y=y, pos=pos, spd=spd, 
                            smiles=Chem.MolToSmiles(mol))
                data_list.append(data)
                valid_count += 1
            except Exception as e:
                print(f"Skipping molecule {i} due to error: {e}")
                continue

        with open(self.processed_paths[0], 'wb') as f:
            pickle.dump(data_list, f)
        print(f"Finish! A total of {valid_count} effective molecules were processed. Data saved to {self.processed_paths[0]}")
        
        self._data_list = data_list

    def len(self):
        return len(self._data_list)

    def get(self, idx):
        return self._data_list[idx]

    def _get_multitask_labels(self, mol):
        num_atoms = mol.GetNumAtoms()
        num_tasks = len(SUBTYPES)
        y = torch.zeros((num_atoms, num_tasks), dtype=torch.float32) 
        
        
        if not mol.HasProp('SOM_Enzyme_Map'):
            return y
            
        map_str = mol.GetProp('SOM_Enzyme_Map')
        
        
        sites = map_str.split('|')
        
        for site in sites:
            site = site.strip()
            if ':' not in site: continue
            
            try:
                idx_str, enzymes_str = site.split(':')
                atom_idx = int(idx_str.strip()) 
                
                if not (0 <= atom_idx < num_atoms):
                    continue
                
                
                enzymes = [e.strip().upper() for e in enzymes_str.split(',') if e.strip()]
                
                for enz_name in enzymes:
                    for task_i, subtype in enumerate(SUBTYPES):
                        if subtype in enz_name:
                            y[atom_idx, task_i] = 1.0
                            
            except Exception as e:
                pass
                
        return y


    def _get_optimized_3d_coords(self, mol, mol_idx):
        num_atoms = mol.GetNumAtoms()
        try:
            mol_h = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42 
            status = AllChem.EmbedMolecule(mol_h, params)
            if status == -1: return torch.zeros((num_atoms, 3), dtype=torch.float32)
            AllChem.MMFFOptimizeMolecule(mol_h)
            mol_opt = Chem.RemoveHs(mol_h)
            return torch.tensor(mol_opt.GetConformer().GetPositions(), dtype=torch.float32)
        except: return torch.zeros((num_atoms, 3), dtype=torch.float32)

    def _get_spd_matrix(self, mol, edge_index):
        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0 or edge_index.size(1) == 0: return torch.zeros((0,), dtype=torch.long)
        G = nx.Graph()
        G.add_nodes_from(range(num_atoms))
        for i, j in edge_index.t().tolist(): G.add_edge(i, j)
        spd = nx.floyd_warshall_numpy(G, nodelist=range(num_atoms))
        spd[np.isinf(spd)] = self.max_spd + 1
        src, dst = edge_index[0].numpy(), edge_index[1].numpy()
        return torch.tensor(np.clip(spd[src, dst], 0, self.max_spd), dtype=torch.long)

    def _get_node_features(self, mol):
        try:
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            mol_wt = Descriptors.MolWt(mol)
            num_rings = Descriptors.RingCount(mol)
        except: 
            logp, tpsa, mol_wt, num_rings = 0.0, 0.0, 0.0, 0.0

        global_feats = torch.tensor([
            np.clip(logp / 10.0, -2.0, 2.0),
            np.clip(tpsa / 150.0, 0.0, 2.0),
            np.clip(mol_wt / 500.0, 0.0, 2.0),
            np.clip(num_rings / 5.0, 0.0, 2.0)
        ], dtype=torch.float32)
        
        atom_features = []
        for i, atom in enumerate(mol.GetAtoms()):
            symbol_list = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']; atom_type_oh = _one_hot_index(symbol_list.index(atom.GetSymbol()) if atom.GetSymbol() in symbol_list else 9, 10)
            hybrid_list = [HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3]; hybrid_oh = _one_hot_index(hybrid_list.index(atom.GetHybridization()) if atom.GetHybridization() in hybrid_list else 3, 4)
            degree_idx = atom.GetDegree() if atom.GetDegree() <= 4 else 5; degree_oh = _one_hot_index(degree_idx, 6)
            h_count_idx = atom.GetTotalNumHs() if atom.GetTotalNumHs() <= 3 else 4; h_count_oh = _one_hot_index(h_count_idx, 5)
            aromatic_feat = torch.tensor([1.0] if atom.GetIsAromatic() else [0.0], dtype=torch.float32)
            ring_count_idx = mol.GetRingInfo().NumAtomRings(atom.GetIdx()); ring_count_idx = ring_count_idx if ring_count_idx <= 2 else 3; ring_count_oh = _one_hot_index(ring_count_idx, 4)
            implicit_valence = atom.GetImplicitValence(); implicit_valence_idx = implicit_valence if implicit_valence <= 4 else 5; implicit_valence_oh = _one_hot_index(implicit_valence_idx, 6)
            
            gasteiger_charge = atom.GetDoubleProp('_GasteigerCharge') if atom.HasProp('_GasteigerCharge') else 0.0
            if not np.isfinite(gasteiger_charge): gasteiger_charge = 0.0 
            
            gasteiger_charge_feat = torch.tensor([gasteiger_charge], dtype=torch.float32)
            formal_charge_feat = torch.tensor([atom.GetFormalCharge()], dtype=torch.float32)
            radical_electrons_feat = torch.tensor([atom.GetNumRadicalElectrons()], dtype=torch.float32)
            
            chiral_tag = atom.GetChiralTag()
            chiral_map = {ChiralType.CHI_TETRAHEDRAL_CCW: 0, ChiralType.CHI_TETRAHEDRAL_CW: 1, ChiralType.CHI_OTHER: 2}
            chiral_oh = _one_hot_index(chiral_map.get(chiral_tag, 3), 4)

            final_atom_feature = torch.cat([
                atom_type_oh, hybrid_oh, degree_oh, h_count_oh, aromatic_feat, 
                ring_count_oh, implicit_valence_oh, gasteiger_charge_feat, 
                formal_charge_feat, radical_electrons_feat, chiral_oh, global_feats
            ], dim=0) 
            atom_features.append(final_atom_feature)
        
        if len(atom_features) == 0: return torch.zeros((0, 47), dtype=torch.float32)
        return torch.stack(atom_features, dim=0)

    def _get_edge_features(self, mol):
        feats=[]
        def _one_hot_bool(flag: bool) -> torch.Tensor:
            return torch.tensor([1., 0.], dtype=torch.float32) if flag else torch.tensor([0., 1.], dtype=torch.float32)
        hybrid_map = {HybridizationType.SP: 0, HybridizationType.SP2: 1, HybridizationType.SP3: 2}
        combo_map = {(2, 2): 0, (1, 2): 1, (1, 1): 2, (0, 2): 3, (0, 1): 4}

        for b in mol.GetBonds():
            i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx(); ai,aj=mol.GetAtomWithIdx(i),mol.GetAtomWithIdx(j); bt=b.GetBondType()
            if bt==Chem.BondType.SINGLE:bond_type_oh=_one_hot_index(0,4)
            elif bt==Chem.BondType.DOUBLE:bond_type_oh=_one_hot_index(1,4)
            elif bt==Chem.BondType.TRIPLE:bond_type_oh=_one_hot_index(2,4)
            elif bt==Chem.BondType.AROMATIC or b.GetIsAromatic():bond_type_oh=_one_hot_index(3,4)
            else:bond_type_oh=_one_hot_index(0,4)
            
            charge_i = ai.GetDoubleProp('_GasteigerCharge') if ai.HasProp('_GasteigerCharge') else 0.0
            charge_j = aj.GetDoubleProp('_GasteigerCharge') if aj.HasProp('_GasteigerCharge') else 0.0
            bond_polarity_feat = torch.tensor([abs(charge_i - charge_j)], dtype=torch.float32)
            
            h_i = ai.GetHybridization(); h_j = aj.GetHybridization()
            key = tuple(sorted((hybrid_map.get(h_i, 99), hybrid_map.get(h_j, 99))))
            hybrid_combo_oh = _one_hot_index(combo_map.get(key, 5), 6)
            
            is_in_ring_oh = _one_hot_bool(b.IsInRing())
            is_conjugated_oh = _one_hot_bool(b.GetIsConjugated())

            e=torch.cat([bond_type_oh, bond_polarity_feat, hybrid_combo_oh, is_in_ring_oh, is_conjugated_oh], dim=0) 
            feats.append(e); feats.append(e) 
            
        if len(feats)==0: return torch.zeros((0,15),dtype=torch.float32)
        return torch.stack(feats,dim=0)

    def _get_adjacency_info(self, mol):
        edges=[]
        for bond in mol.GetBonds():
            i,j=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
            edges+=[[i,j],[j,i]]
        if not edges: return torch.empty((2,0),dtype=torch.long)
        return torch.tensor(edges,dtype=torch.long).t().contiguous()


if __name__ == '__main__':

    SDF_FILE = '' 
    
    if os.path.exists(SDF_FILE):
        
        dataset = V7_Multitask_Dataset(root='.', filename=SDF_FILE)
        
        print("\n=== Data Check ===")
        print(f"Total sample size: {len(dataset)}")

        if len(dataset) > 0:
            data = dataset[0]
            print(f"SMILES: {data.smiles}")
            print(f"Node feature x shape: {data.x.shape}")
            print(f"Label y shape: {data.y.shape} (expected: [NumAtoms, 9])")
            
            nonzero_indices = torch.nonzero(data.y)
            if nonzero_indices.shape[0] > 0:
                print("SoM tag detected (AtomIdx, TaskIdx):")
                for row in nonzero_indices:
                    atom_idx, task_idx = row.tolist()
                    enzyme_name = SUBTYPES[task_idx]
                    print(f"  - Atom {atom_idx} -> CYP{enzyme_name}")
            else:
                print("Sample 0 did not detect any metabolic sites (possibly a negative sample or a mismatch in analysis)")
    else:
        print(f"Error: File not found {SDF_FILE}")
