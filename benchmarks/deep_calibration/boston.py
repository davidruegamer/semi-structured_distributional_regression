import os

import sys

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

import numpy
import scipy.stats
import matplotlib.pyplot
import sklearn.datasets
import sklearn.model_selection
import utils
from GP_Beta_cal import GP_Beta
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings("ignore")

(x_raw, y_raw) = sklearn.datasets.load_boston(return_X_y=True)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_raw, y_raw.ravel(),
                                                                            test_size=0.25, shuffle=True, 
                                                                            random_state = 1337)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

base_model = utils.get_mdl(x_train, y_train, 'gp') # change 'olr' to 'deep' / 'br' / 'gp' for other base models

lls_base = []
lls_iso = []
lls_gp = []
mses_base = []
mses_iso = []
mses_gp = []

for i in range(20):

    print("Iteration " + str(i))

    blockPrint()
    
    mu_cal, sigma_cal = utils.get_prediction(x_train, base_model)

    n_t_test = 1024

    t_list_test = numpy.linspace(numpy.min(mu_cal) - 16.0 * numpy.max(sigma_cal),
                                 numpy.max(mu_cal) + 16.0 * numpy.max(sigma_cal),
                                 n_t_test).reshape(1, -1)

    n_u = 16 # number of induced points
    GP_Beta_mdl = GP_Beta()
    failed = True
    while failed:
        try:
            GP_Beta_mdl.fit(y_train, mu_cal, sigma_cal, n_u=n_u)
            failed = False
        except:
            failed = True

    iso_q, iso_q_hat = utils.get_iso_cal_table(y_train, mu_cal, sigma_cal)
    iso_mdl = IsotonicRegression(out_of_bounds='clip')
    iso_mdl.fit(iso_q, iso_q_hat)

    mu_base, sigma_base = utils.get_prediction(x_test, base_model)
    y_base = mu_base.ravel()
    q_base, s_base = utils.get_norm_q(mu_base.ravel(), sigma_base.ravel(), t_list_test.ravel())

    s_gp, q_gp = GP_Beta_mdl.predict(t_list_test, mu_base, sigma_base)
    y_gp = utils.get_y_hat(t_list_test.ravel(), s_gp)

    q_iso = iso_mdl.predict(q_base.ravel()).reshape(numpy.shape(q_base))
    s_iso = numpy.diff(q_iso, axis=1) / \
            (t_list_test[0, 1:] - t_list_test[0, :-1]).ravel().reshape(1, -1).repeat(len(y_test), axis=0)
    y_iso = utils.get_y_hat(t_list_test.ravel(), s_iso)

    ll_base = - scipy.stats.norm.logpdf(y_test.reshape(-1, 1),
                                        loc=mu_base.reshape(-1, 1),
                                        scale=sigma_base.reshape(-1, 1)).ravel()
    ll_iso = utils.get_log_loss(y_test, t_list_test.ravel(), s_iso)
    ll_gp = utils.get_log_loss(y_test, t_list_test.ravel(), s_gp)
    lls_base.append(ll_base)
    lls_iso.append(ll_iso)
    lls_gp.append(ll_gp)
    print([numpy.mean(ll_base), numpy.mean(ll_iso), numpy.mean(ll_gp)])

    se_base = utils.get_se(y_base, y_test)
    se_iso = utils.get_se(y_iso, y_test)
    se_gp = utils.get_se(y_gp, y_test)
    mses_base.append(se_base)
    mses_iso.append(se_iso)
    mses_gp.append(se_gp)
    print([numpy.mean(se_base), numpy.mean(se_iso), numpy.mean(se_gp)])
    
    res = [numpy.mean(lls_base), numpy.mean(lls_iso), numpy.mean(lls_gp),
           numpy.std(lls_base), numpy.std(lls_iso), numpy.std(lls_gp), 
           numpy.mean(mses_base), numpy.mean(mses_iso), numpy.mean(mses_gp), 
           numpy.std(mses_base), numpy.std(mses_iso), numpy.std(mses_gp)]
    
    print(res)

    numpy.savetxt('test_boston.out', res, delimiter=',')
