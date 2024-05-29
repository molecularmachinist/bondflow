import bond_analysis as ba
import numpy as np


reference_struc = 'centered_active.gro' 
trajectory = 'reduced_frames.xtc' 

print('Making contact maps, this will take a while...')
contact_map, distance_map, contact_map_names = ba.salt_bridge_contact_map(reference_struc, trajectory)
print('Contact map made') 

with open('contact_map.npy', 'wb') as f:

    np.save(f, contact_map)

with open('distance_map.npy', 'wb') as f:

    np.save(f, distance_map)

with open('contact_map_names.npy', 'wb') as f:

    np.save(f, contact_map_names)




