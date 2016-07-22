import astropy.units as u
import numpy as np
from astropy.time import Time
from matplotlib import pyplot as plt

from toolkit import regression_coeffs, regression_model, transit_model_b

times = np.load('outputs/times.npy')
xcentroids = np.load('outputs/xcentroids.npy')
ycentroids = np.load('outputs/ycentroids.npy')
fluxes = np.load('outputs/fluxes.npy')
errors = np.load('outputs/errors.npy')
airmass = np.load('outputs/airmass.npy')
airpress = np.load('outputs/airpress.npy')
humidity = np.load('outputs/humidity.npy')
telfocus = np.load('outputs/telfocus.npy')
medians = np.load('outputs/medians.npy')

mid_transit_time = Time(2457580.872658, format='jd')
b_duration = 36.12 * u.min + 10*u.min
apertureradii = np.arange(7, 18) #np.arange(7, 14)
# aperture_index = 5

nstars = xcentroids.shape[1]
stds = []
models = []
oot_masks = []
for aperture_index in range(len(apertureradii)):
    target_fluxes = fluxes[:, 0, aperture_index]
    target_errors = errors[:, 0, aperture_index]

    regressors = np.hstack([fluxes[:, 1:, aperture_index],
                            xcentroids[:, 0, np.newaxis],
                            ycentroids[:, 0, np.newaxis],
                            airmass[:, np.newaxis],
                            airpress[:, np.newaxis],
                            humidity[:, np.newaxis],
                            #telfocus[:, np.newaxis],
                            medians[:, np.newaxis]
                            ])

    labels = (nstars*['fluxes'] + ['xcentroids'] + ['ycentroids'] +
              ['airmass', 'airpress', 'humidity', 'telfocus', 'medians'])

    out_of_transit = ((Time(times, format='jd') > mid_transit_time + b_duration/2) |
                      (Time(times, format='jd') < mid_transit_time - b_duration/2))

    n_iterations = 10

    for i in range(n_iterations):
        c = regression_coeffs(regressors[out_of_transit],
                              target_fluxes[out_of_transit],
                              target_errors[out_of_transit])

        # for c_i, l_i in zip(c, labels):
        #     print(c_i, l_i)

        m = regression_model(c, regressors)

        light_curve = target_fluxes/m

        median = np.median(light_curve[out_of_transit])
        std = np.std(light_curve[out_of_transit])
        outliers = np.abs(light_curve - median) > 4*std
        #np.ones_like(light_curve).astype(bool)
        out_of_transit &= np.logical_not(outliers)

    stds.append(light_curve[out_of_transit].std())
    models.append(m)
    oot_masks.append(out_of_transit)

best_lc_index = np.argmin(stds)
out_of_transit = oot_masks[best_lc_index]

light_curve = target_fluxes/models[best_lc_index]
light_curve_errors = target_errors/models[best_lc_index]

oot_median = np.median(light_curve[out_of_transit])
light_curve = light_curve / oot_median
light_curve_errors = light_curve_errors / oot_median

np.save('outputs/bestlc.npy', light_curve)
np.save('outputs/bestlc_errors.npy', light_curve_errors)

plt.figure()
plt.plot(apertureradii, stds)
plt.xlabel('aperture radius')
plt.ylabel('stddev')
plt.show()

plt.figure()
plt.plot(times, light_curve, '.')
plt.plot(times, transit_model_b(times))
#plt.plot(times[out_of_transit], light_curve[out_of_transit], 'r.')
plt.show()

