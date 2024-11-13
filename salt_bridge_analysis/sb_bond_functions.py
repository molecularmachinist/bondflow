import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import contacts
import pandas as pd 


def salt_bridge_contact_map(ref, trj):

    u = mda.Universe(ref, trj)

    #Contact map

    #Collect all negative and positive residues

    positive_residues = u.select_atoms("(resname ARG) and (name CZ)") + u.select_atoms("(resname LYS) and (name CE)")
    negative_residues = u.select_atoms("(resname ASP) and (name CG)") + u.select_atoms("(resname GLU) and (name CD)")


    i = len(positive_residues)*len(negative_residues)
    j = len(u.trajectory) #time

    binary_contact_map = np.zeros((i,j)) # i, j = residue combination, frame
    contact_map_names =  np.chararray((i),itemsize=15)
    distance_map = np.zeros((i,j)) # i, j = residue combination distance, frame

    for ts in u.trajectory:
        n = 0
        for pos_res in positive_residues:
            
            for neg_res in negative_residues: 
                n += 1
                if ts.frame == 0:
                    
                    contact_map_names[n-1] =  str(pos_res.resname) + str(int(pos_res.resid)) + '_' + str(neg_res.resname) + str(int(neg_res.resid))
                
                dist = contacts.distance_array(pos_res.position, neg_res.position)
                
                distance_map[n-1, int(ts.frame)] = dist
                
                if dist <= 6.0:
                    
                    binary_contact_map[n-1, int(ts.frame)] = 1

    return binary_contact_map, distance_map, contact_map_names # Returns a contact map with only 1 and 0, a distance map where there are the distances of the chosen residues, and a map of residue pairing names


def create_sorted_distance_dataframe(contact_map_names, binary_contact_map, distance_map):
    
    std_contact_map = np.std(binary_contact_map, axis=1)

    indices = np.flatnonzero(std_contact_map)

    j = indices
    sb_names = []
    stds = []
    for n in range(len(j)):
        index_i = j[n]
        
        sb_name = contact_map_names[index_i]
        sb_name = sb_name.decode("UTF-8")
       
        sb_names.append(sb_name)
        std = std_contact_map[index_i]
        stds.append(std)

    data = {'Salt bridge': sb_names,
            'Standard deviation': stds,
        'Index': indices}
    
    #Create dataframe
    df_std = pd.DataFrame(data)

    #Sort the standard deviations highest to lowest
    df_std_sorted=df_std.sort_values(by=['Standard deviation'], ascending=False)

    dist = []
    sb_names = []

    for ind in df_std_sorted.index:
        index, name = df_std_sorted['Index'][ind], df_std_sorted['Salt bridge'][ind]
        
        time_dependence = distance_map[index,:]
        dist.append(list(time_dependence))
        sb_names.append(name)
        
    dist = np.array(dist)
    dist = np.transpose(dist)

    distance_data_frame = pd.DataFrame(dist, columns= sb_names)
    
    return distance_data_frame, sb_names, df_std_sorted





def moving_average(data, n):
    mvalue = []

    # Käydään läpi datan alkiot, joita ennen olevat alkiot ovat kaikki keskiarvoistusikkunassa:
    for k in range(0, n//2):
        data_in_window = data[:k+n//2+1]
        mvalue.append(sum(data_in_window)/len(data_in_window))

    for k in range(n//2, len(data)-n//2):
        data_in_window = data[k-n//2:k+n//2+1]
        mvalue.append(sum(data_in_window)/len(data_in_window))

    # Lopuksi datan alkiot, joiden jälkeen olevat alkiot ovat kaikki keskiarvoistusikkunassa:
    for k in range(len(data)-n//2, len(data)):
        data_in_window = data[k-n//2:]
        mvalue.append(sum(data_in_window)/len(data_in_window))

    return np.array(mvalue)



def moving_average_contact_map_maker(contact_map_names, binary_contact_map,ma_window):
    
    ma_binary_contact_map = np.zeros(shape=(binary_contact_map.shape))

    j = len(contact_map_names)
    for n in range(j):
        index_i = n
            
        contacts = binary_contact_map[index_i,:]

        contacts = moving_average(contacts,ma_window)

        ma_binary_contact_map[index_i] = contacts

    return ma_binary_contact_map

        

def create_contact_map_from_dataframe(df):

    contact_map = df.to_numpy()
    cols_list = df.columns.tolist()

    return contact_map, cols_list
    
      
  