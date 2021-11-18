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
BoxSize = 35  # [Mpc/h]
Nmesh = 64 * 10  # 10 mini-box
cat = ArrayCatalog({
    'Position': dm_df.loc[:,['x','y','z']].values/1000,
    'vx': dm_df.loc[:,['vx']].values[:,0],
    'vy': dm_df.loc[:,['vy']].values[:,0],
    'vz': dm_df.loc[:,['vz']].values[:,0]
})

'''paint on mesh!'''
# convert to mesh object
mesh = cat.to_mesh(BoxSize=BoxSize, Nmesh=Nmesh,
                   resampler='nearest')
meshx = cat.to_mesh(BoxSize=BoxSize, Nmesh=Nmesh, value='vx', 
                    resampler='nearest')
meshy = cat.to_mesh(BoxSize=BoxSize, Nmesh=Nmesh, value='vy', 
                    resampler='nearest')
meshz = cat.to_mesh(BoxSize=BoxSize, Nmesh=Nmesh, value='vz', 
                    resampler='nearest')
# start paint value on mesh!!!
meshed = mesh.paint(mode='real')
meshed_x = meshx.paint(mode='real')
meshed_y = meshy.paint(mode='real')
meshed_z = meshz.paint(mode='real')

'''save result'''
np.save("./tmp_save/TNG-50-hydro-z0-dm-mesh-v-num.npy",meshed.value)
np.save("./tmp_save/TNG-50-hydro-z0-dm-mesh-vx.npy",meshed_x.value)
np.save("./tmp_save/TNG-50-hydro-z0-dm-mesh-vy.npy",meshed_y.value)
np.save("./tmp_save/TNG-50-hydro-z0-dm-mesh-vz.npy",meshed_z.value)