import numpy as np
import pyarrow.feather as feather
from nbodykit.source.catalog import ArrayCatalog
from nbodykit import setup_logging

setup_logging("debug")

'''load data'''
dm_df = feather.read_feather("./tmp_save/TNG-50-hydro-z0-dm-snap.df.feather")

'''make catalog for nbodykit
Position[Mpc/h]: (x,y,z) We need to /1000 so that [kpc/h] -> [Mpc/h]
'''
h = 0.6774
BoxSize = 35  # [Mpc/h]
Nmesh = 64 * 10  # 10 mini-box
cat = ArrayCatalog({
    'Position': dm_df.loc[:,['x','y','z']].values/1000
})

'''paint on mesh!'''
# convert to mesh object
mesh = cat.to_mesh(BoxSize=BoxSize, Nmesh=Nmesh, value='Value',
                   resampler='cic')
# start paint value on mesh!!!
meshed_data = mesh.paint(mode='real')

'''save result'''
np.save("./tmp_save/TNG-50-hydro-z0-dm-mesh-pos.npy",meshed_data.value)
