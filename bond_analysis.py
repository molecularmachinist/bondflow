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

    contact_map = np.zeros((i,j)) # i, j = residue combination, frame
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
                
                if dist <= 4.0:
                    
                    contact_map[n-1, int(ts.frame)] = 1

    return contact_map, distance_map, contact_map_names # Returns a contact map with only 1 and 0, a distance map where there are the distances of the chosen residues, and a map of residue pairing names


def average_contact_indices(map):


    average_contact_map = np.mean(map, axis=1)


    average_contact_map = np.where(average_contact_map < 0.6, average_contact_map, 0) 
    average_contact_map = np.where(average_contact_map > 0.4, average_contact_map, 0) 
    indices = np.flatnonzero(average_contact_map)

    return indices

def std_contact_indices(map):

    std_contact_map = np.std(map, axis=1)


    std_contact_map = np.where(std_contact_map > 0.4, std_contact_map, 0) 

    indices = np.flatnonzero(std_contact_map)
    
    return indices

def make_dataframe(contact_map, contact_map_names, indices):
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

def find_indices_moving_average_distance(distance_map,contact_map_names,j,window,min_change):
    sb_names = []
    indices = []
    print(len(j))
    for n in range(len(j)):
        index_i = j[n]
        print(n)
        sb_name = contact_map_names[index_i]
        sb_name = sb_name.decode("UTF-8")
        
        if sb_name not in sb_names:
            sb_names.append(sb_name)
            
            dist = distance_map[index_i,:]
            dist = moving_average(dist,window)
            maxx = np.max(dist)
            minn = np.min(dist)
            if maxx > 4.0 and minn < 4.0:
            
                distance_change = maxx - minn
                if distance_change >= min_change:
                    indices.append(index_i)

    return indices


def distance_change(distance_map,contact_map_names,change):
    sb_names = []
    indices = []
    for n in range(len(j)):
        index_i = j[n]
        
        sb_name = contact_map_names[index_i]
        sb_name = sb_name.decode("UTF-8")
        
        if sb_name not in sb_names:
            sb_names.append(sb_name)
            
            dist = distance_map[index_i,:]
            maxx = np.max(dist)
            minn = np.min(dist)
            if maxx > 4.0 and minn < 4.0:
            
                distance_change = maxx - minn
                if distance_change >= change:
                    indices.append(index_i)

    return indices

#Hydrogen bond analysis

def hydrogen_bond_contact_map(ref,trj):

    u = mda.Universe(ref,trj)

    hydrogens = u.select_atoms("protein and (name HN)")

    acceptors_sel = u.select_atoms("protein and (name O OE* OD*)")



    i = len(hydrogens)*len(acceptors_sel)
    j = len(u.trajectory) #time

    contact_map = np.zeros((i,j))
    contact_map_names =  np.chararray((i),itemsize=25)

    distance_map = np.zeros((i,j)) # i, j = distance, frame

    for ts in u.trajectory:
        n = 0
    for hydrogen in hydrogens:
        
        for acceptor in acceptors_sel: 
        
            n += 1
            if ts.frame == 0:
                
                contact_map_names[n-1] = str(hydrogen.name) + '_' + str(hydrogen.resname) + str(int(hydrogen.resid)) + '_' + str(acceptor.name) + '_' + str(acceptor.resname) + str(int(acceptor.resid))
               
            dist = contacts.distance_array(hydrogen.position, acceptor.position)

            distance_map[n-1, int(ts.frame)] = dist
            
            if dist < 2.5:
                
                donor_index = hydrogen.index - 1
                donor = u.atoms[donor_index]
                
                vector_donor_to_hydrogen = hydrogen.position - donor.position
                vector_acceptor_to_hydrogen = hydrogen.position - acceptor.position
                
                angle = np.degrees(np.arccos(np.dot(vector_donor_to_hydrogen, vector_acceptor_to_hydrogen) /
                                             (np.linalg.norm(vector_donor_to_hydrogen) *
                                              np.linalg.norm(vector_acceptor_to_hydrogen))))
                
                # Angle is now in degrees and represents the angle between the donor, hydrogen, and acceptor atoms
  
                if angle <= 180 and angle > 160:
                 
                  
                    contact_map[n-1, int(ts.frame)] = 1

    return contact_map, distance_map, contact_map_names

def h_find_indices_moving_average_distance(distance_map,contact_map_names,j,window,min_change):
    sb_names = []
    indices = []
    for n in range(len(j)):
        index_i = j[n]
        
        sb_name = contact_map_names[index_i]
        sb_name = sb_name.decode("UTF-8")
        
        if sb_name not in sb_names:
            sb_names.append(sb_name)
            
            dist = distance_map[index_i,:]
            dist = moving_average(dist,window)
            maxx = np.max(dist)
            minn = np.min(dist)
            if max > 2.5 and min < 2.5:
            
                distance_change = maxx - minn
                if distance_change >= min_change:
                    indices.append(index_i)

    return indices


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

def moving_average_contact_map(map):

    moving_contact_map = map
    i = len(map)

    for n in range(i):
        
        moving_contact_map[n,:] = moving_average(map[n,:],5)

    return moving_contact_map

def std_indices(map):
    std_contact_map = np.std(map, axis=1)

    std_contact_map = np.where(std_contact_map >= 0.35, std_contact_map, 0) 

    j = np.flatnonzero(std_contact_map)

    return j
