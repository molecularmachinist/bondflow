import bond_analysis as ba
import numpy as np

moving_average_window = 500
minimun_distance_change = 9

with open('contact_map.npy', 'rb') as f:

    contact_map = np.load(f)

with open('distance_map.npy', 'rb') as f:

    distance_map = np.load(f)

with open('contact_map_names.npy', 'rb') as f:

    contact_map_names = np.load(f)
 

std_indices = ba.std_contact_indices(contact_map)

indices = ba.find_indices_moving_average_distance(distance_map, contact_map_names, std_indices, moving_average_window, minimun_distance_change)

print(len(indices), 'salt bridges has been found ')



sb_df = ba.make_dataframe(distance_map, contact_map_names, indices)

distance_df = ba.make_dataframe(contact_map, contact_map_names, indices)

sb_df.to_csv('sb_data.csv', index=False)

distance_df.to_csv('distance_data.csv', index=False)