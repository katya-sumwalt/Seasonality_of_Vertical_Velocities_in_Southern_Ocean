import cleanup_seal_data as csd 
import compute_w as cw 
import xarray as xr
import numpy as np
import matplotlib.pyplot as  plt

raw_data = xr.open_dataset('/Users/kat/Desktop/LS_organized/data/updated_along_track_strain_raw_seal_data.nc')
masked_dataset = csd.apply_bxmask_to_raw_dataset(raw_data)
extended_masked_dataset = csd.extend_dataset_to_depth(masked_dataset)
sliced_extended_masked_dataset = csd.slice_dataset_at_ends(extended_masked_dataset)

unique_names = np.unique(sliced_extended_masked_dataset.name)

select_seal = unique_names[0]
seal_dataset = sliced_extended_masked_dataset.where(sliced_extended_masked_dataset.name == select_seal, drop= True)
distance = np.arange(seal_dataset.sizes['time']) * 1000.00  ## gives distance in meters
ds_dist = seal_dataset.rename({'time': 'distance'}).assign_coords(distance=distance) 
selected_1000km_chunk = ds_dist.isel(distance = slice(0,1000))
w = cw.computew(selected_1000km_chunk)

plt.pcolor(ds_dist.distance,ds_dist.depth,w)
