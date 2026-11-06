from computing_vertical_velocities import cleanup_seal_data as csd
from computing_vertical_velocities import compute_w as cw
import config
import xarray as xr
import numpy as np
import matplotlib.pyplot as  plt

if __name__ == '__main__':
    raw_data = xr.open_dataset(config.RAW_SEAL_FILE)
    masked_dataset = csd.apply_bxmask_to_raw_dataset(raw_data)
    extended_masked_dataset = csd.extend_dataset_to_depth(masked_dataset)
    sliced_extended_masked_dataset = csd.slice_dataset_at_ends(extended_masked_dataset)

    unique_names = np.unique(sliced_extended_masked_dataset.name)

    select_seal = unique_names[0]
    seal_dataset = sliced_extended_masked_dataset.where(sliced_extended_masked_dataset.name == select_seal, drop= True)
    distance = np.arange(seal_dataset.sizes['time']) * 1000.00  ## gives distance in meters
    ds_dist = seal_dataset.rename({'time': 'distance'}).assign_coords(distance=distance) 
    selected_1000km_chunk = ds_dist.isel(distance = slice(0,1000))
    w = cw.compute_w(selected_1000km_chunk)

    plt.pcolor(selected_1000km_chunk.distance,selected_1000km_chunk.depth,w)
    plt.show()
