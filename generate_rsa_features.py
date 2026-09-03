import os
import pickle
import freesasa
import numpy as np
from tqdm import tqdm

MAX_ASA = {
    'ALA': 129.0, 'ARG': 274.0, 'ASN': 195.0, 'ASP': 193.0, 'CYS': 167.0,
    'GLN': 225.0, 'GLU': 223.0, 'GLY': 104.0, 'HIS': 224.0, 'ILE': 197.0,
    'LEU': 201.0, 'LYS': 236.0, 'MET': 224.0, 'PHE': 240.0, 'PRO': 159.0,
    'SER': 155.0, 'THR': 172.0, 'TRP': 285.0, 'TYR': 263.0, 'VAL': 160.0
}

def main():
    dataset_path = "./Dataset/"
    output_dir = "./Feature/rsa/"
    pdb_dir = "./PDB/"
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = ["Train_335.pkl", "Test_60.pkl", "Test_315-28.pkl", "UBtest_31-6.pkl"]
    protein_sequences = {}
    
    for ds_name in datasets:
        ds_path = os.path.join(dataset_path, ds_name)
        if not os.path.exists(ds_path):
            continue
        with open(ds_path, "rb") as f:
            data = pickle.load(f)
        for pid, val in data.items():
            if pid == '2j3rA':
                continue
            protein_sequences[pid] = val[0]
            
    print(f"Loaded {len(protein_sequences)} unique protein IDs.")
    
    success_count = 0
    for pid, seq in tqdm(protein_sequences.items(), desc="Generating continuous RSA features"):
        out_path = os.path.join(output_dir, f"{pid}.npy")
        if os.path.exists(out_path):
            success_count += 1
            continue
            
        pdb_path = os.path.join(pdb_dir, f"{pid}.pdb")
        seq_len = len(seq)
        
        if not os.path.exists(pdb_path):
            print(f"Warning: {pdb_path} not found. Defaulting to 0.5.")
            rsa_arr = np.full((seq_len,), 0.5, dtype=np.float32)
            np.save(out_path, rsa_arr)
            continue
            
        try:
            structure = freesasa.Structure(pdb_path)
            result = freesasa.calc(structure)
            res_area = result.residueAreas()
            
            rsa_vals = []
            for chain_id in res_area:
                for res_num in res_area[chain_id]:
                    res = res_area[chain_id][res_num]
                    res_name = res.residueType
                    total_area = res.total
                    max_area = MAX_ASA.get(res_name, 200.0)
                    rsa = min(1.0, max(0.0, total_area / max_area))
                    rsa_vals.append(rsa)
            
            rsa_arr = np.array(rsa_vals, dtype=np.float32)
            
            # Align length if discrepancy occurs
            if len(rsa_arr) != seq_len:
                if len(rsa_arr) > seq_len:
                    rsa_arr = rsa_arr[:seq_len]
                else:
                    padding = np.full((seq_len - len(rsa_arr),), 0.5, dtype=np.float32)
                    rsa_arr = np.concatenate([rsa_arr, padding])
                    
            np.save(out_path, rsa_arr)
            success_count += 1
        except Exception as e:
            print(f"Error processing {pid}: {e}. Fallback to 0.5.")
            rsa_arr = np.full((seq_len,), 0.5, dtype=np.float32)
            np.save(out_path, rsa_arr)
            
    print(f"Successfully generated continuous RSA features for {success_count}/{len(protein_sequences)} proteins.")

if __name__ == "__main__":
    main()
