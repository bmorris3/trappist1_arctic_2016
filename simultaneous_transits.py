
from toolkit.transit_model import (transit_model, params_b, params_g, params_h,
                                   params_f, params_e, params_d, params_c)
from toolkit import PhotometryResults
import matplotlib.pyplot as plt

from glob import glob

names = list('bcdefgh')
transit_params = [params_b, params_c, params_d, params_e, params_f, params_g,
                  params_h]

paths = glob('outputs/trappist*.npz')
print(paths, )
fig, ax = plt.subplots(len(paths), 1, figsize=(6, 14))
for i, path in enumerate(paths):
    times = PhotometryResults.load(path).times
    for params, name in zip(transit_params, names):
        print(name)
        ax[i].plot(times, transit_model(times, params), label=name)
    ax[i].legend()
plt.show()