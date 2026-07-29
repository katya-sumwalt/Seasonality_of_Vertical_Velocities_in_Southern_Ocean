
import xarray as xr 
import numpy as np 

summer_dataset = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_summer_200m_not_removed_3.nc')
winter_dataset = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_winter_200m_not_removed_3.nc')
fall_dataset = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_fall_200m_not_removed_4.nc') 
spring_dataset = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_spring_200m_not_removed.nc') #5 is most clean 
ds_east = xr.open_dataset('/Users/kat/Desktop/LS/siegelman-lab-ksumwalt/seal-vertical-velocities/Data/east/seal_data_with_vertical_velocity_east_without_200removed.nc')
# ds_east_2 = xr.open_dataset('/Users/kat/Desktop/LS/seal_data_with_vertical_velocity_east.nc')
# ds_east_3 = xr.open_dataset('/Users/kat/Desktop/LS/seal_data_with_vertical_velocity.nc')
# print(np.unique(ds_east_2.name))
# print(np.unique(ds_east_3.name))

print(ds_east)

#######################
#######################
#######################
# print('number of profiles')
# count_of_profiles = (np.size(summer_dataset.distance)+ np.size(winter_dataset.distance) + np.size(spring_dataset.distance)+ np.size(fall_dataset.distance))
# print(f'{count_of_profiles}')
# print(np.size(ds_east.distance))
########################
########################
########################
# print('start and end dates')
# print(np.min(ds_east.time[0].values))
# print(np.max(ds_east.time.values))
########################
########################
########################
# print('median dive depth') 
# dive_depths = (-1 * ds_east.depth) * ds_east.maskprof
# print(np.nanmedian(dive_depths)) 
########################
########################
########################
# print('median and max dive depth for each seal')
# list_of_seal_names = np.unique(ds_east.name)

# def med_max_dive_depth(ds,seal): 
#     subset = ds.where(ds.name == seal, drop = True)
#     array_of_depths = (-1 * subset.depth) * subset.maskprof
#     med_depth = np.nanmedian(array_of_depths)
#     max_depth = np.nanmax(array_of_depths)
#     number_of_dives = np.argmax(subset.distance.values)
#     return med_depth,max_depth,number_of_dives

# for seal in list_of_seal_names:
#     med_d,max_d,number_of_dives = med_max_dive_depth(ds_east,seal)
#     print(f' for seal {seal} \n max depth: {max_d} \n med depth = {med_d} \n number of dives = {number_of_dives}')
    

########################
########################
######################## 
# print('maximum and minimum lat lon')
# print(f'max:lat {np.max(ds_east.lat).values}')
# print(f'max:lon {np.max(ds_east.lon).values}')
# print(f'min:lat {np.min(ds_east.lat).values}')
# print(f'min:lon {np.min(ds_east.lon).values}')



#######################
#######################
#######################  
# print('median depth values for the data that gave me an issue') 
# ct131_048BAT2_15 = ds_east.where(ds_east.name == 'ct158-198-19', drop = True)
# # print()
# ct131_048BAT2_15_bx =  (-1 * ct131_048BAT2_15.depth)  * ct131_048BAT2_15.maskprof
# print(ct131_048BAT2_15_bx)
# ct131_048BAT2_15_bx_depth = np.nanmedian(ct131_048BAT2_15_bx) 
# print(ct131_048BAT2_15_bx_depth)

########################
########################
########################  
# print('total median dive depth') 
# # ct131_048BAT2_15 = ds_east.where(ds_east.name == 'ct158-198-19', drop = True)
# # print()
# ds_east_blah =  (-1 * ds_east.depth)  * ds_east.maskprof
# ds_east_blah_depth = np.nanmedian(ds_east_blah) 
# print(ds_east_blah_depth)

