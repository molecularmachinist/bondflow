import sb_bond_functions as ba
import numpy as np

reference_struc = 'centered_active.gro' 
trajectory = 'test.xtc' 
ma_window = 25

print('Making contact maps, this will take a while...')
binary_contact_map, distance_map, contact_map_names = ba.salt_bridge_contact_map(reference_struc, trajectory)
print('Contact map made') 

ma_binary_contact_map = ba.moving_average_contact_map_maker(contact_map_names, binary_contact_map,ma_window)

distance_df, sb_names_list = ba.create_sorted_distance_dataframe(contact_map_names, ma_binary_contact_map, distance_map)

distance_df.to_csv('ma_distance_data.csv', index=False)

ma_contact_map, ma_contact_map_names = ba.create_contact_map_from_dataframe(distance_df)


with open('ma_contact_map.npy', 'wb') as f:

    np.save(f, ma_contact_map)


with open('ma_contact_map_names.npy', 'wb') as f:

    np.save(f, ma_contact_map_names)
