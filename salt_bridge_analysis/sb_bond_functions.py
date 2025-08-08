#Improved functions

import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis import contacts
import pandas as pd



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

import os
import numpy as np
import MDAnalysis as mda
from tqdm import tqdm
from itertools import combinations
from joblib import Parallel, delayed


# ---------- Utility Functions ---------- #

def compute_batch_distances_vectorized(coords, batch_pairs):
    """Compute pairwise distances for a batch of residue pairs across all frames."""
    i_indices = np.array([i for i, _ in batch_pairs])
    j_indices = np.array([j for _, j in batch_pairs])
    diff = coords[:, i_indices, :] - coords[:, j_indices, :]
    return np.linalg.norm(diff, axis=-1)  # (n_frames, n_pairs)


def filter_by_contact(batch_distances, batch_pairs, contact_threshold=8.0):
    """Keep residue pairs with min distance <= threshold."""
    min_distances = np.min(batch_distances, axis=0)
    return [pair for pair, keep in zip(batch_pairs, min_distances <= contact_threshold) if keep]


def filter_by_variance(batch_distances, batch_pairs, variance_percentile=75):
    """Keep residue pairs above the given variance percentile."""
    min_d = np.min(batch_distances, axis=0)
    max_d = np.max(batch_distances, axis=0)
    
    normalized = (batch_distances - min_d) / (max_d - min_d + 1e-8)
    variances = np.var(normalized, axis=0)
    
    threshold = np.percentile(variances, variance_percentile)
    return [pair for pair, keep in zip(batch_pairs, variances >= threshold) if keep]


def parallel_batch_filtering(coords, pairs, filter_func, batch_size, desc, n_jobs=-1, **kwargs):
    """Generic parallel batching for any filter function."""
    n_batches = (len(pairs) + batch_size - 1) // batch_size

    def process_batch(batch_idx):
        start, end = batch_idx * batch_size, min((batch_idx + 1) * batch_size, len(pairs))
        batch_pairs = pairs[start:end]
        batch_distances = compute_batch_distances_vectorized(coords, batch_pairs)
        return filter_func(batch_distances, batch_pairs, **kwargs)

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_batch)(batch) for batch in tqdm(range(n_batches), desc=desc)
    )
    return [pair for sublist in results for pair in sublist]


# ---------- Main Processing ---------- #

def compute_com_contacts(trajs, ref, batch_size=100, stride=1, 
                         contact_threshold=8.0, variance_percentile=75, n_jobs=-1):
    """
    Compute COM residue contact pairs from multiple trajectories.
    
    Returns:
        current_pairs: filtered residue pairs
        com_coords: concatenated COM coordinates
    """
    com_coords_list = []

    for traj in trajs:
        model = mda.Universe(ref, traj)
        
        heavy = model.select_atoms(
            "protein and not type H and resindex 2321 to 3004 3047 to 3174 1554 to 1900 654 to 1403"
        )
        n_frames = len(model.trajectory[::stride])
        n_heavy_residues = heavy.residues.n_residues
        
        com_coords = np.zeros((n_frames, n_heavy_residues, 3))
        
        for idx, frame in enumerate(model.trajectory[::stride]):
            com_coords[idx] = heavy.residues.center_of_mass(compound='residues')
            print(f"Frame {idx}/{n_frames} from {os.path.basename(traj)}", end="\r")

        com_coords_list.append(com_coords)

    com_coords = np.concatenate(com_coords_list, axis=0)

    # Generate all residue pairs
    n_residues = com_coords.shape[1]
    all_pairs = list(combinations(range(n_residues), 2))
    print(f"Total possible residue pairs: {len(all_pairs)}")

    # Stage 1: Contact Filter
    current_pairs = parallel_batch_filtering(
        coords=com_coords, pairs=all_pairs,
        filter_func=filter_by_contact,
        batch_size=batch_size, desc="Stage 1: Contact Filter",
        n_jobs=n_jobs, contact_threshold=contact_threshold
    )
    print(f"Pairs after contact filter: {len(current_pairs)}")

    # Stage 2: Variance Filter
    current_pairs = parallel_batch_filtering(
        coords=com_coords, pairs=current_pairs,
        filter_func=filter_by_variance,
        batch_size=batch_size, desc="Stage 2: Variance Filter",
        n_jobs=n_jobs, variance_percentile=variance_percentile
    )
    print(f"Pairs after variance filter: {len(current_pairs)}")

    return current_pairs, com_coords





