import find_bond_functions
import sys

reference_struc = sys.argv[1]
trajectory = sys.argv[2]

contact_map, contact_map_names = find_bond_functions.salt_bridge_contact_map(reference_struc, trajectory)
                            
                            
i = find_bond_functions.std_contact(contact_map)

df = find_bond_functions.sb_time_dependence(contact_map, contact_map_names, i)

df.to_csv('sb_data.csv', index=False)
df.to_excel('sb_data.xlsx', index=False)