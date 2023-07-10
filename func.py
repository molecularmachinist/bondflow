import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import contacts
import pandas as pd 

def salt_bridge_contact_map(ref, trj):

    u = mda.Universe(ref, trj)

    #Contact map

    #Collect all negative and positive residues

    positive_residues = u.select_atoms("(resname LYS ARG) and (name NH* NZ)")
    negative_residues = u.select_atoms("(resname ASP GLU) and (name OE* OD*)")


    i = len(positive_residues)*len(negative_residues)
    j = len(u.trajectory) #time

    contact_map = np.zeros((i,j))
    contact_map_names =  np.chararray((i),itemsize=25)

    for ts in u.trajectory:
        n = 0
        for pos_res in positive_residues:
        
            for neg_res in negative_residues: 
                n += 1
                if ts.frame == 0:
                    
                    contact_map_names[n-1] = str(pos_res.resname) + str(int(pos_res.resid)) + '_' + '_' + str(neg_res.resname) + str(int(neg_res.resid))
                
                dist = contacts.distance_array(pos_res.position, neg_res.position)
                if dist <= 6.0:
                    
                    contact_map[n-1, int(ts.frame)] = 1

    return contact_map, contact_map_names


def average_contact(map):


    average_contact_map = np.mean(map, axis=1)


    average_contact_map = np.where(average_contact_map < 0.6, average_contact_map, 0) 
    average_contact_map = np.where(average_contact_map > 0.4, average_contact_map, 0) 
    indices = np.flatnonzero(average_contact_map)

    return indices

def std_contact(map):

    std_contact_map = np.std(map, axis=1)


    std_contact_map = np.where(std_contact_map > 0.49, std_contact_map, 0) 

    indices = np.flatnonzero(std_contact_map)

    return indices

def sb_time_dependence(contact_map, contact_map_names, indices):
#Time dependens for the salt bridges that break/form

    data = []
    sb_names = []

    for n in range(len(indices)):
        index_i = indices[n]

        sb_name = contact_map_names[index_i]
        sb_name = sb_name.decode("UTF-8")
        
        if sb_name not in sb_names:
            sb_names.append(sb_name)
            
            time_dependence = contact_map[index_i,:]
            data.append(list(time_dependence))

    data = np.array(data)

    data = np.transpose(data)


    df = pd.DataFrame(data, columns= sb_names)

    return df
