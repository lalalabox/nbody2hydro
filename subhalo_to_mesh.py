import numpy as np
import pyarrow.feather as feather
from nbodykit.source.catalog import ArrayCatalog
from nbodykit import setup_logging

setup_logging("debug")

'''load data'''
subhalo_df = feather.read_feather("./tmp_save/TNG-50-hydro-z0-subhalo-catalog.df.feather")

'''make catalog for nbodykit
Position[Mpc/h]: (x,y,z) We need to /1000 so that [kpc/h] -> [Mpc/h]
'''
BoxSize = 35  # [Mpc/h]
Nmesh = 64 * 10  # 10 mini-box
cat = ArrayCatalog({
    'Position': subhalo_df.loc[:,['SubhaloPos_x','SubhaloPos_y','SubhaloPos_z']].values/1000
})
subhalo_df.drop(columns=['SubhaloLen','SubhaloGrNr','SubhaloPos_x','SubhaloPos_y','SubhaloPos_z'],inplace=True)
keys = subhalo_df.columns.to_list()
for key in keys:
    cat[key] = subhalo_df.loc[:,key].values
    pass

'''paint to mesh'''
def _goto_mesh(key):
    if key == 'Position':
        value = 'Value'
    else:
        value = key
    mesh = cat.to_mesh(BoxSize=BoxSize, Nmesh=Nmesh, resampler='nearest', value=value)
    meshed = mesh.paint(mode='real')
    np.save(f"./tmp_save/TNG-50-hydro-z0-subhalo-{key}.npy",meshed.value)
    return

tmp = [_goto_mesh(key) for key in keys]
# '''multiprocessing'''
# #************************************************************************************
# from multiprocessing import Pool
# from time import time

# pool = Pool(processes=len(keys))
# time_start = time()

# tmp_data = list(pool.map(
#     _goto_mesh,
#     keys + ['Position']
# ))

# pool.close()
# pool.join()
# print(f"Done in {time() - time_start} second")
#************************************************************************************
