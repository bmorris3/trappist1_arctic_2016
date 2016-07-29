import astropy.units as u
import numpy as np
from astropy.time import Time
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from astropy.stats import mad_std

from toolkit import regression_coeffs, regression_model, PhotometryResults, params_b

pr = PhotometryResults.load('outputs/trappist1b.npz')
transit_parameters = params_b

expected_mid_transit_jd = (np.max(np.abs(pr.times - transit_parameters.t0) //
                                  transit_parameters.per) *
                           transit_parameters.per + transit_parameters.t0)
mid_transit_time = Time(expected_mid_transit_jd, format='jd')

transit_duration = transit_parameters.duration + 5*u.min
# aperture_index = 5

aperture_index = 5

final_lc_mad = np.ones(len(pr.aperture_radii))

final_lc = None

for aperture_index in range(len(pr.aperture_radii)):
    target_fluxes = pr.fluxes[:, 0, aperture_index]
    target_errors = pr.errors[:, 0, aperture_index]

    inliers = np.ones_like(pr.fluxes[:, 0, aperture_index]).astype(bool)

    for i in range(pr.fluxes.shape[1]):
        flux_i = pr.fluxes[:, i, aperture_index]

        linear_flux_trend = np.polyval(np.polyfit(pr.times, flux_i, 2), pr.times)
        new_inliers = np.abs(flux_i - linear_flux_trend) < 2.*mad_std(flux_i)
        inliers &= new_inliers

        # plt.plot(pr.times, flux_i - linear_flux_trend)
        # plt.plot(pr.times[np.logical_not(new_inliers)], (flux_i - linear_flux_trend)[np.logical_not(new_inliers)], 'ro')
        # plt.show()

    out_of_transit = ((Time(pr.times, format='jd') > mid_transit_time + transit_duration/2) |
                      (Time(pr.times, format='jd') < mid_transit_time - transit_duration/2))

    validation_duration = 4/5 * transit_duration

    validation_mask = ((Time(pr.times, format='jd') < mid_transit_time +
                        1.5 * transit_duration + validation_duration / 2) &
                       (Time(pr.times, format='jd') > mid_transit_time +
                        1.5 * transit_duration - validation_duration / 2))

    oot = out_of_transit & inliers
    oot_no_validation = (out_of_transit & inliers & np.logical_not(validation_mask))

    ones = np.ones((len(pr.times), 1))
    regressors = np.hstack([pr.fluxes[:, 1:, aperture_index],
                            pr.xcentroids[:, 0, np.newaxis],
                            pr.ycentroids[:, 0, np.newaxis],
                            pr.airmass[:, np.newaxis],
                            pr.airpressure[:, np.newaxis],
                            pr.humidity[:, np.newaxis],
                            pr.background_median[:, np.newaxis]
                            ])

    n_components = np.arange(2, regressors.shape[1])

    def train_pca_linreg_model(out_of_transit_mask, oot_no_validation_mask, n_comp):

        # OOT chunk first:
        pca = PCA(n_components=n_comp)
        reduced_regressors = pca.fit_transform(regressors[out_of_transit_mask],
                                               target_fluxes[out_of_transit_mask])

        prepended_regressors_oot = np.hstack([ones[out_of_transit_mask],
                                              reduced_regressors])
        c_oot = regression_coeffs(prepended_regressors_oot,
                                  target_fluxes[out_of_transit_mask],
                                  target_errors[out_of_transit_mask])

        lc_training = (target_fluxes[out_of_transit_mask] -
                       regression_model(c_oot, prepended_regressors_oot))

        median_oot = np.median(target_fluxes[out_of_transit_mask])
        std_lc_training = np.std((lc_training + median_oot) / median_oot)

        # Now on validation chunk:
        reduced_regressors_no_validation = pca.fit_transform(regressors[oot_no_validation_mask],
                                                             target_fluxes[oot_no_validation_mask])

        prepended_regressors_no_validation = np.hstack([ones[oot_no_validation_mask],
                                                        reduced_regressors_no_validation])
        c_no_validation = regression_coeffs(prepended_regressors_no_validation,
                                            target_fluxes[oot_no_validation_mask],
                                            target_errors[oot_no_validation_mask])

        lc_validation = (target_fluxes[out_of_transit_mask] -
                         regression_model(c_no_validation, prepended_regressors_oot))

        std_lc_validation = np.std((lc_validation + median_oot) / median_oot)

        #return lc_training, lc_validation
        return lc_training, lc_validation, std_lc_training, std_lc_validation


    stds_validation = np.zeros_like(n_components, dtype=float)
    stds_training = np.zeros_like(n_components, dtype=float)

    for i, n_comp in enumerate(n_components):

        lc_training, lc_validation, std_lc_training, std_lc_validation = train_pca_linreg_model(oot, oot_no_validation, n_comp)

        validation_mask = np.logical_not(oot_no_validation) & out_of_transit

        stds_validation[i] = std_lc_validation
        stds_training[i] = std_lc_training

        # plt.title(n_comp)
        # plt.plot(times[oot], lc_training, 'b.')
        # plt.plot(times[oot], lc_validation, 'r.')
        # # plt.plot(times[inliers], target_fluxes[inliers], '.')
        # # plt.plot(times[not_masked], model)
        # plt.show()

    best_n_components = n_components[np.argmin(stds_validation)]
    # plt.plot(n_components, stds_validation, label='validation')
    # plt.plot(n_components, stds_training, label='training')
    # plt.axvline(best_n_components, color='r', ls='--')
    # plt.title(aperture_index)
    # plt.legend()
    # plt.show()


    # Now apply PCA to generate light curve with best number of components
    pca = PCA(n_components=best_n_components)
    reduced_regressors = pca.fit_transform(regressors[oot], target_fluxes[oot])

    all_regressors = pca.transform(regressors)
    prepended_all_regressors = np.hstack([ones, all_regressors])

    prepended_regressors_oot = np.hstack([ones[oot], reduced_regressors])
    c_oot = regression_coeffs(prepended_regressors_oot,
                              target_fluxes[oot],
                              target_errors[oot])

    best_lc = ((target_fluxes - regression_model(c_oot, prepended_all_regressors)) /
               np.median(target_fluxes)) + 1

    final_lc_mad[aperture_index] = mad_std(best_lc[out_of_transit])

    if final_lc_mad[aperture_index] == np.min(final_lc_mad):
        final_lc = best_lc.copy()


plt.plot(pr.aperture_radii, final_lc_mad)
plt.axvline(pr.aperture_radii[np.argmin(final_lc_mad)], ls='--', color='r')
plt.xlabel('Aperture radii')
plt.ylabel('mad(out-of-transit light curve)')

plt.figure()
plt.plot(pr.times, final_lc, 'k.')
plt.xlabel('Time [JD]')
plt.ylabel('Flux')
plt.show()
# plt.figure()
# plt.plot(pr.times, best_lc, 'k.')
# plt.show()

#    print(np.log10(lam), np.std(model))
#     print(weights.shape)
#     mads[i] = mad_std(target_fluxes[not_masked] - model)

# plt.loglog(lambdas, mads)
# plt.show()
# nstars = xcentroids.shape[1]
# stds = []
# models = []
# oot_masks = []
# for aperture_index in range(len(apertureradii)):
#     target_fluxes = fluxes[:, 0, aperture_index]
#     target_errors = errors[:, 0, aperture_index]
#
#     regressors = np.hstack([fluxes[:, 1:, aperture_index],
#                             xcentroids[:, 0, np.newaxis],
#                             ycentroids[:, 0, np.newaxis],
#                             airmass[:, np.newaxis],
#                             airpress[:, np.newaxis],
#                             humidity[:, np.newaxis],
#                             #telfocus[:, np.newaxis],
#                             medians[:, np.newaxis]
#                             ])
#
#     labels = (nstars*['fluxes'] + ['xcentroids'] + ['ycentroids'] +
#               ['airmass', 'airpress', 'humidity', 'telfocus', 'medians'])
#
#     out_of_transit = ((Time(times, format='jd') > mid_transit_time + b_duration/2) |
#                       (Time(times, format='jd') < mid_transit_time - b_duration/2))
#
#     n_iterations = 10
#
#     for i in range(n_iterations):
#         c = regression_coeffs(regressors[out_of_transit],
#                               target_fluxes[out_of_transit],
#                               target_errors[out_of_transit])
#
#         # for c_i, l_i in zip(c, labels):
#         #     print(c_i, l_i)
#
#         m = regression_model(c, regressors)
#
#         light_curve = target_fluxes/m
#
#         median = np.median(light_curve[out_of_transit])
#         std = np.std(light_curve[out_of_transit])
#         outliers = np.abs(light_curve - median) > 4*std
#         #np.ones_like(light_curve).astype(bool)
#         out_of_transit &= np.logical_not(outliers)
#
#     stds.append(light_curve[out_of_transit].std())
#     models.append(m)
#     oot_masks.append(out_of_transit)
#
# best_lc_index = np.argmin(stds)
# out_of_transit = oot_masks[best_lc_index]
#
# light_curve = target_fluxes/models[best_lc_index]
# light_curve_errors = target_errors/models[best_lc_index]
#
# oot_median = np.median(light_curve[out_of_transit])
# light_curve = light_curve / oot_median
# light_curve_errors = light_curve_errors / oot_median
#
# np.save('outputs/bestlc.npy', light_curve)
# np.save('outputs/bestlc_errors.npy', light_curve_errors)
# #np.save('outputs/bestlc_times.npy', times[out_of_transit])
#
# plt.figure()
# plt.plot(apertureradii, stds)
# plt.xlabel('aperture radius')
# plt.ylabel('stddev')
#
# plt.figure()
# plt.plot(times, light_curve, '.')
# plt.plot(times, transit_model_b(times))
# #plt.plot(times[out_of_transit], light_curve[out_of_transit], 'r.')
#
#
# from interpacf import interpolated_acf, dominant_period
# from toolkit.transit_model import params_b, transit_model_b_depth_t0
#
# initp = np.array([params_b.rp**2, params_b.t0])
# init_transit_model = transit_model_b_depth_t0(times, initp[0], initp[1])
#
# # Compute acf, find autoregressive period
# median = np.median(light_curve)
# inliers = np.abs(median - light_curve) < 3*np.std(light_curve)
#
# lags, acf = interpolated_acf(times[inliers],
#                              light_curve[inliers] - init_transit_model[inliers])
# residual_period = dominant_period(lags, acf, fwhm=20,
#                                   min=0.005, max=0.02, plot=True)
# print(residual_period)
# plt.show()
