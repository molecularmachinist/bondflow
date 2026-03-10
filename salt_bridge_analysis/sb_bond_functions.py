#Improved functions

import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis import contacts
import pandas as pd
import os
from tqdm import tqdm
from itertools import combinations, product
from joblib import Parallel, delayed
from itertools import combinations
from statsmodels.stats.diagnostic import lilliefors



def salt_bridge_contact_map(ref, trj):

    u = mda.Universe(ref, trj)

    # Collect all positive and negative residues 
    #positive_residues = u.select_atoms("(resname LYS ARG) and (name NZ NH*)")
    #negative_residues = u.select_atoms("(resname ASP GLU) and (name OE* OD*)")
    #positive_residues = u.select_atoms("((resname ARG and name CZ) or (resname LYS and name CE))")
    #negative_residues = u.select_atoms("((resname ASP and name CG) or (resname GLU and name CD))")
    positive_residues = u.select_atoms("(resname ARG and name CZ) or (resname LYS and name CE)")
    negative_residues = u.select_atoms("((resname ASP and name CG) or (resname GLU and name CD))")

    num_pos_residues = len(positive_residues)
    num_neg_residues = len(negative_residues)
    num_combinations = num_pos_residues * num_neg_residues
    num_frames = len(u.trajectory)

    # Pre-allocate arrays
    binary_contact_map = np.zeros((num_combinations, num_frames), dtype=np.int8)
    contact_map_names = np.empty(num_combinations, dtype='U15')
    distance_map = np.zeros((num_combinations, num_frames))

    # Generate contact map names
    n = 0
    for pos_res in positive_residues:
        for neg_res in negative_residues:
            contact_map_names[n] = f"{pos_res.resname}{pos_res.resid}_{neg_res.resname}{neg_res.resid}"
            n += 1

    # Compute distances and contact map
    for ts in u.trajectory:
        pos_positions = positive_residues.positions
        neg_positions = negative_residues.positions

        # Calculate distance matrix between positive and negative residues
        dist_matrix = contacts.distance_array(pos_positions, neg_positions)
        
        # Flatten distance matrix and store in distance map
        distance_map[:, ts.frame] = dist_matrix.flatten()

        # Update binary contact map based on distance threshold
        binary_contact_map[:, ts.frame] = (dist_matrix.flatten() <= 5).astype(np.int8)

    return binary_contact_map, distance_map, contact_map_names



def moving_average(data, n):
    if n < 1:
        raise ValueError("Window size n must be at least 1")
    if n > len(data):
        raise ValueError("Window size n must not be larger than the data length")
    
    # Compute the cumulative sum
    cumsum = np.cumsum(np.insert(data, 0, 0)) 
    cumsum = np.delete(cumsum, 0)
    mvalue = (cumsum[n:] - cumsum[:-n]) / n

    # Pad the start and end to match the size of the input array
    start_padding = [np.mean(data[:i+1]) for i in range(n//2)]
    end_padding = [np.mean(data[-(i+1):]) for i in range(n//2, 0, -1)]
 
    mvalue = np.concatenate([start_padding, mvalue, end_padding])

    return mvalue



def slow_moving_average(data, n):
    mvalue = []

    for k in range(0, n//2):
        data_in_window = data[:k+n//2+1]
        mvalue.append(sum(data_in_window)/len(data_in_window))

    for k in range(n//2, len(data)-n//2):
        data_in_window = data[k-n//2:k+n//2+1]
        mvalue.append(sum(data_in_window)/len(data_in_window))

    for k in range(len(data)-n//2, len(data)):
        data_in_window = data[k-n//2:]
        mvalue.append(sum(data_in_window)/len(data_in_window))

    return np.array(mvalue)



def moving_average_contact_map_maker(binary_contact_map,ma_window):
    
    ma_binary_contact_map = np.zeros(shape=(binary_contact_map.shape))

    
    for n in range(len(binary_contact_map[:,0])):
        index_i = n
            
        contacts = binary_contact_map[index_i,:]

        contacts = moving_average(contacts,ma_window)

        ma_binary_contact_map[index_i] = contacts

    return ma_binary_contact_map



def create_sorted_distance_dataframe(contact_map_names, contact_map, distance_map):


    # Calculate standard deviation of the binary contact map
    std_contact_map = np.std(contact_map, axis=1)

    # Get indices of non-zero standard deviations
    indices = np.flatnonzero(std_contact_map)
    stds = std_contact_map[indices]

    # Extract names and standard deviations
    sb_names = contact_map_names[indices].astype(str)

    # Create DataFrame and sort by standard deviation
    data = {'Salt bridge': sb_names, 'Standard deviation': stds, 'Index': indices}
    df_std = pd.DataFrame(data)
    df_std_sorted = df_std.sort_values(by='Standard deviation', ascending=False)

    # Extract the sorted indices
    sorted_indices = df_std_sorted['Index'].values

    # Extract and transpose the relevant distances
    dist = distance_map[sorted_indices].T

    # Create the final DataFrame with distances
    distance_data_frame = pd.DataFrame(dist, columns=df_std_sorted['Salt bridge'])

    return distance_data_frame, df_std_sorted['Salt bridge'].tolist(), df_std_sorted



def create_contact_map_from_dataframe(df):

    contact_map = df.to_numpy()
    cols_list = df.columns.tolist()

    return contact_map, cols_list

# UUTTA!

# ---------- Utility Functions ---------- #

def compute_batch_distances(coords, batch_pairs):
    """Compute pairwise distances for a batch of residue pairs across all frames."""
    i_indices = np.array([i for i, _ in batch_pairs])
    j_indices = np.array([j for _, j in batch_pairs])
    diff = coords[:, i_indices, :] - coords[:, j_indices, :]
    return np.linalg.norm(diff, axis=-1)  # (n_frames, n_pairs)


def filter_by_contact(batch_distances, batch_pairs, contact_threshold=8.0):
    """Keep residue pairs with min distance <= threshold."""
    min_distances = np.nanmin(batch_distances, axis=0)
    return [pair for pair, keep in zip(batch_pairs, min_distances <= contact_threshold) if keep]


def filter_by_variance(batch_distances, batch_pairs, variance_percentile=75):
    """Keep residue pairs above the given variance percentile."""
    min_d = np.nanmin(batch_distances, axis=0)
    max_d = np.nanmax(batch_distances, axis=0)
    
    normalized = (batch_distances - min_d) / (max_d - min_d + 1e-8)
    variances = np.nanvar(normalized, axis=0)
    
    threshold = np.percentile(variances, variance_percentile)
    return [pair for pair, keep in zip(batch_pairs, variances >= threshold) if keep]


def parallel_batch_filtering(coords, pairs, filter_func, batch_size, desc, n_jobs=1, **kwargs):
    """Generic parallel batching for any filter function."""
    n_batches = (len(pairs) + batch_size - 1) // batch_size

    def process_batch(batch_idx):
        start, end = batch_idx * batch_size, min((batch_idx + 1) * batch_size, len(pairs))
        batch_pairs = pairs[start:end]
        batch_distances = compute_batch_distances(coords, batch_pairs)
        return filter_func(batch_distances, batch_pairs, **kwargs)

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_batch)(batch) for batch in tqdm(range(n_batches), desc=desc)
    )
    return [pair for sublist in results for pair in sublist]


def pairs_to_dataframe(pairs, n_mainchain_residues, protein_residues):
    """
    Convert index pairs to a pandas DataFrame with residue names, numbers, and chain types.
    """
    data = []
    
    for i, j in pairs:
        type_i = 'MC' if i < n_mainchain_residues else 'SC'
        type_j = 'MC' if j < n_mainchain_residues else 'SC'
        
        # Map to residue index within main chain block
        resid_i = protein_residues[i % n_mainchain_residues]
        resid_j = protein_residues[j % n_mainchain_residues]
        
        res_i_str = f"{resid_i.resname}{resid_i.resnum}_{resid_i.resindex}"  # Include resindex
        res_j_str = f"{resid_j.resname}{resid_j.resnum}_{resid_j.resindex}"  # Include resindex

        # Skip pairs where both residues are the same
        if res_i_str == res_j_str:
            continue
        
        pair_name = f"{res_i_str}({type_i})_{res_j_str}({type_j})"
        
        data.append([res_i_str, type_i, res_j_str, type_j, pair_name, i, j])
    
    df = pd.DataFrame(data, columns=['residue_i', 'type_i', 'residue_j', 'type_j', 'pair_name', 'idx_i', 'idx_j'])
    return df




# ---------- Main Processing ---------- #

def compute_com_contacts(trajs, ref, batch_size=100, stride=1,
                         mc_contact_threshold=8.0, sc_contact_threshold=8.0, 
                         mc_sc_contact_threshold=8.0, variance_percentile=75, n_jobs=1):
    """
    Compute COM residue contact pairs from multiple trajectories.
    Faster version using vectorized MDAnalysis operations and parallel trajectory processing.
    
    Returns:
        current_mainchain_pairs: filtered residue pairs
        com_mainchain_coords: concatenated COM coordinates
    """

    # -------- Helper to process one trajectory -------- #
    def process_single_traj(traj):
        model = mda.Universe(ref, traj)
        protein_residues = model.select_atoms("protein").residues
        n_residues = len(protein_residues)
        n_frames = len(model.trajectory[::stride])

        com_mainchain_coords = np.empty((n_frames, n_residues, 3))
        com_sidechain_coords = np.full((n_frames, n_residues, 3), np.nan)  # fill with NaN

        # Selections
        mainchain = model.select_atoms(
            "protein and (name N or name CA or name C or name O) and not type H"
        )
        sidechain = model.select_atoms(
            "protein and not (name N or name CA or name C or name O) and not type H"
        )

        # Residue ID mapping (to fill COMs in correct order)
        residue_nums = protein_residues.resnums
        sidechain_resnums = sidechain.residues.resnums
        resnum_to_index = {resnum: i for i, resnum in enumerate(residue_nums)}

        for idx, ts in enumerate(tqdm(model.trajectory[::stride], desc=os.path.basename(traj))):
            com_mainchain_coords[idx] = mainchain.center_of_mass(compound='residues')

            # COMs for sidechains that exist
            sidechain_coms = sidechain.center_of_mass(compound='residues')
            for sc_resnum, com in zip(sidechain_resnums, sidechain_coms):
                j = resnum_to_index.get(sc_resnum)
                if j is not None:
                    com_sidechain_coords[idx, j] = com

        return com_mainchain_coords, com_sidechain_coords


    # -------- Process trajectories in parallel -------- #
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_traj)(traj) for traj in trajs
    )

    # Combine results across trajectories
    com_mainchain_coords = np.concatenate([r[0] for r in results], axis=0)
    com_sidechain_coords = np.concatenate([r[1] for r in results], axis=0)

    # Combine: shape = (n_frames_total, 2*n_residues, 3)
    combined_coords = np.concatenate([com_mainchain_coords, com_sidechain_coords], axis=1)

    n_mainchain_residues = com_mainchain_coords.shape[1]
    n_sidechain_residues = com_sidechain_coords.shape[1]
    n_coords = combined_coords.shape[1]
    
    # --- Generate pairs ---

    all_pairs = list(combinations(range(n_coords), 2))
    #print(f"Total contact pairs: {len(all_pairs)}")
    mc_start = 0
    mc_end = n_mainchain_residues

    sc_start = n_mainchain_residues
    sc_end = n_coords

    # Index ranges
    mc_indices = range(mc_start, mc_end)
    sc_indices = range(sc_start, sc_end)

    # --- 1. Mainchain - Mainchain ---
    mc_mc_pairs = list(combinations(mc_indices, 2))
    print(f"Mainchain-Mainchain pairs: {len(mc_mc_pairs)}")

    # --- 2. Sidechain - Sidechain ---
    sc_sc_pairs = list(combinations(sc_indices, 2))
    print(f"Sidechain-Sidechain pairs: {len(sc_sc_pairs)}")

    # --- 3. Mainchain - Sidechain ---
    mc_sc_pairs = [(i, j) for i in mc_indices for j in sc_indices]
    print(f"Mainchain-Sidechain pairs: {len(mc_sc_pairs)}")

    # -------- Stage 1: Contact Filter -------- #
    mc_current_pairs = parallel_batch_filtering(
        coords=combined_coords,
        pairs=mc_mc_pairs,
        filter_func=filter_by_contact,
        batch_size=batch_size,
        desc="Stage 1: Contact Filter",
        n_jobs=n_jobs,
        contact_threshold=mc_contact_threshold,
    )
    print(f"MC-MC pairs after contact filter: {len(mc_current_pairs):,}")

    sc_current_pairs = parallel_batch_filtering(
        coords=combined_coords,
        pairs=sc_sc_pairs,
        filter_func=filter_by_contact,
        batch_size=batch_size,
        desc="Stage 1: Contact Filter",
        n_jobs=n_jobs,
        contact_threshold=sc_contact_threshold,
    )
    print(f"SC-SC pairs after contact filter: {len(sc_current_pairs):,}")

    mc_sc_current_pairs = parallel_batch_filtering(
        coords=combined_coords,
        pairs=mc_sc_pairs,
        filter_func=filter_by_contact,
        batch_size=batch_size,
        desc="Stage 1: Contact Filter",
        n_jobs=n_jobs,
        contact_threshold=mc_sc_contact_threshold,
    )
    print(f"MC-SC pairs after contact filter: {len(mc_sc_current_pairs):,}")

    current_pairs = (
    mc_current_pairs
    + sc_current_pairs
    + mc_sc_current_pairs
)

    print(f"Total pairs after Stage 1 (all classes): {len(current_pairs):,}")

    # -------- Stage 2: Variance Filter -------- #
    current_pairs = parallel_batch_filtering(
        coords=combined_coords,
        pairs=current_pairs,
        filter_func=filter_by_variance,
        batch_size=batch_size,
        desc="Stage 2: Variance Filter",
        n_jobs=n_jobs,
        variance_percentile=variance_percentile,
    )
    print(f"Pairs after variance filter: {len(current_pairs):,}")

    return all_pairs, current_pairs, combined_coords, n_mainchain_residues


def compute_residue_pair_distances(coords, pairs, n_mainchain_residues, protein_residues):
    """
    Compute Z-score normalized distances between residue pairs across trajectory frames.
    Only normalized distances are included in the output.

    Returns a DataFrame with:
    - pair_name
    - bond_type
    - idx_i, idx_j
    - frame_0 ... frame_N (normalized)
    - mean_distance, std_distance
    """
    n_frames, n_residues, _ = coords.shape

    # Build pair metadata
    df_pairs = pairs_to_dataframe(pairs, n_mainchain_residues, protein_residues)

    # Define bond type
    df_pairs["bond_type"] = df_pairs["type_i"] + "-" + df_pairs["type_j"]

    # --- Compute distances ---
    all_distances = np.empty((len(df_pairs), n_frames))
    for k, row in df_pairs.iterrows():
        i, j = int(row["idx_i"]), int(row["idx_j"])
        diff = coords[:, i, :] - coords[:, j, :]
        all_distances[k] = np.linalg.norm(diff, axis=1)

    # Compute stats
    mean_dist = all_distances.mean(axis=1)
    std_dist = all_distances.std(axis=1)
    std_dist[std_dist == 0] = np.nan

    # Z-score normalization per pair
    #norm_distances = (all_distances - mean_dist[:, None]) / std_dist[:, None]
    norm_distances = all_distances
    # Build normalized DataFrame
    frame_cols = [f"frame_{i}" for i in range(n_frames)]
    df_norm = pd.DataFrame(norm_distances, columns=frame_cols)

    # Combine metadata + normalized distances (only)
    df_full = pd.concat([df_pairs.reset_index(drop=True)[["pair_name", "bond_type", "idx_i", "idx_j"]], df_norm], axis=1)

    # Add summary stats
    df_full["mean_distance"] = mean_dist
    df_full["std_distance"] = std_dist

    return df_full




# -------- Helper functions for contact type processing -------- #

def remove_duplicate_pairs(current_pairs, protein_residues, n_mainchain_residues):
    """
    Keep the first encounter of each unique residue-residue pair (by name + number + index),
    regardless of mainchain/sidechain composition.
    Excludes self-pairs (e.g., VAL495–VAL495).
    """

    seen = set()
    unique_pairs = []

    for i, j in current_pairs:
        res_i = protein_residues[i % n_mainchain_residues]
        res_j = protein_residues[j % n_mainchain_residues]

        # Build residue identifiers (e.g., VAL495_0)
        res_i_name = f"{res_i.resname}{res_i.resnum}_{res_i.resindex}"
        res_j_name = f"{res_j.resname}{res_j.resnum}_{res_j.resindex}"

        # Skip self-pairs (same residue name and id)
        if res_i_name == res_j_name:
            continue

        # Canonical key ignores order (A_B == B_A)
        key = "_".join(sorted([res_i_name, res_j_name]))

        if key not in seen:
            unique_pairs.append((i, j))
            seen.add(key)

    return unique_pairs



def map_combined_to_original(unique_pairs, n_mainchain_residues):
    """
    Map combined COM indices back to original residue indices.

    Args:
        unique_pairs (list[tuple[int, int]]): Pairs of indices into combined_coords
        n_mainchain_residues (int): Number of residues in the protein

    Returns:
        list[tuple[int, int]]: Pairs of original residue indices
    """
    original_pairs = []
    for i, j in unique_pairs:
        orig_i = i % n_mainchain_residues
        orig_j = j % n_mainchain_residues
        original_pairs.append((orig_i, orig_j))
    return original_pairs

# -------- Contact type processing -------- #

def salt_bridge_contact_map_filtered(ref, traj_list, current_pairs, distance_threshold=6.0):
    """
    Compute salt bridge contact maps for filtered residue pairs over multiple trajectories.

    Args:
        ref (str): topology file
        traj_list (list[str]): list of trajectory file paths
        filtered_pairs (list[tuple]): residue index pairs from earlier filtering
        distance_threshold (float): contact cutoff in Å
        
    Returns:
        binary_contact_map: (n_pairs, total_frames) binary contact map
        distance_map: (n_pairs, total_frames) salt bridge distances
        contact_map_names: numpy array (n_pairs,)
    """
    # Load first trajectory to identify salt bridge pairs
    u_test = mda.Universe(ref, traj_list[0])

    protein_residues = u_test.select_atoms("protein").residues
    n_mainchain_residues = len(protein_residues)

    unique_pairs = remove_duplicate_pairs(current_pairs, protein_residues, n_mainchain_residues)
    filtered_pairs = map_combined_to_original(unique_pairs, n_mainchain_residues)
    # Atom selections for salt bridge atoms
    pos_sel = "(resname ARG and name CZ) or (resname LYS and name CE)"
    neg_sel = "(resname ASP and name CG) or (resname GLU and name CD)"

    positive_residues = u_test.select_atoms(pos_sel).residues
    negative_residues = u_test.select_atoms(neg_sel).residues

    # Filter the given pairs to only include those where one is positive and one is negative
    sb_pairs = []
    sb_names_tmp = []

    for i, j in filtered_pairs:
        res_i = u_test.residues[i]
        res_j = u_test.residues[j]

        if ((res_i in positive_residues and res_j in negative_residues) or
            (res_j in positive_residues and res_i in negative_residues)):

            sb_pairs.append((i, j))

            name_i = f"{res_i.resname}{res_i.resid}"
            name_j = f"{res_j.resname}{res_j.resid}"
            sb_names_tmp.append(f"{name_i}_{name_j}")

    if not sb_pairs:
        print("No filtered pairs correspond to possible salt bridges.")
        return None, None, None

    # Preallocate numpy array for names
    n_pairs = len(sb_pairs)
    max_len = max(len(name) for name in sb_names_tmp)
    contact_map_names = np.empty(n_pairs, dtype=f'U{max_len}')

    for idx, name in enumerate(sb_names_tmp):
        contact_map_names[idx] = name

    # We'll accumulate results for all trajectories
    binary_contact_maps = []
    distance_maps = []

    for traj in traj_list:
        u = mda.Universe(ref, traj)

        # Recreate the selections for this trajectory universe
        traj_pair_selections = []
        for i, j in sb_pairs:
            res_i = u.residues[i]
            res_j = u.residues[j]
            atom_i = res_i.atoms.select_atoms(pos_sel if res_i.resname in ["ARG", "LYS"] else neg_sel)
            atom_j = res_j.atoms.select_atoms(pos_sel if res_j.resname in ["ARG", "LYS"] else neg_sel)
            traj_pair_selections.append((atom_i, atom_j))

        n_frames = len(u.trajectory)
        bin_map = np.zeros((n_pairs, n_frames), dtype=np.int8)
        dist_map = np.zeros((n_pairs, n_frames))

        # Loop over trajectory frames
        for ts in u.trajectory:
            for idx, (atom_i, atom_j) in enumerate(traj_pair_selections):
                dist_matrix = contacts.distance_array(atom_i.positions, atom_j.positions)
                min_dist = np.min(dist_matrix)
                dist_map[idx, ts.frame] = min_dist
                bin_map[idx, ts.frame] = int(min_dist <= distance_threshold)

        binary_contact_maps.append(bin_map)
        distance_maps.append(dist_map)

    # Concatenate results across all trajectories
    binary_contact_map = np.concatenate(binary_contact_maps, axis=1)
    distance_map = np.concatenate(distance_maps, axis=1)

    return binary_contact_map, distance_map, contact_map_names

