import json
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

direction_std = 0.5 * np.pi


def getConditionalBivariateNorm(mu_X, mu_Y, sigma_XX, sigma_XY, sigma_YY, X):
    # Y|X が多変量正規分布N(mu, Sigma)に従う場合のmu, Sigmaを求める
    sigma_XX_i = LA.inv(sigma_XX)
    mu = mu_Y + np.dot(np.dot(sigma_XY.T, sigma_XX_i), (X - mu_X))
    Sigma = sigma_YY - np.dot(np.dot(sigma_XY.T, sigma_XX_i), sigma_XY)
    return mu, Sigma


def getEclipseParameters(mu, sigma, alpha=0.95):
    # mu: 平均 ([alt][x|y])
    # sigma: 分散共分散行列
    # alpha: 楕円内に点が入る確率

    sigma_i = LA.inv(sigma)
    eigvals, M = LA.eig(sigma_i)
    lmda2 = -2*np.log(1.00000000 - alpha)
    scale = np.sqrt(lmda2/eigvals)
    return scale, M, mu


def getEclipsePlot(a, b, Mrot=None, angle_rad=0, x0=0, y0=0, n_plots=100):
    theta = np.linspace(0, 2 * np.pi, n_plots)

    v0 = np.array([x0, y0])
    v = np.array([a * np.cos(theta), b * np.sin(theta)])

    if Mrot is None:
        Mrot = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
            ])

    return np.dot(Mrot, v) + v0[:, None]


def getAzimuthWindByPlot(U, V, azimuth_rad):
    # 基準高度の楕円上の各風の方位計算[-pi, +pi]
    wind_dir = np.arctan2(-U, -V)
    idx = np.argmin(abs(wind_dir - azimuth_rad))
    return np.array([U[idx], V[idx]])


def getProbEclipse(mu, sigma, alpha=0.95, n_plots=100):
    # mu: 平均 ([alt][x|y])
    # sigma: 分散共分散行列
    # alpha: 楕円内に点が入る確率
    # n_plots: プロット数

    scale, M, _ = getEclipseParameters(mu, sigma, alpha)
    plots = getEclipsePlot(
        scale[0],
        scale[1],
        M,
        x0=mu[0],
        y0=mu[1],
        n_plots=n_plots)
    return plots[0], plots[1]


def getStatWindVector(wind_statistics, wind_std):
    alt_index_std = wind_statistics['altitude_idx_std']
    alt_axis = wind_statistics['alt_axis']
    # alt_std = alt_axis[alt_index_std]
    n_alt = len(alt_axis)

    mu4 = np.array(wind_statistics['mu4'])
    sigma4 = np.array(wind_statistics['sigma4'])

    # ----------------------------
    # Probabillity Eclipse
    # ----------------------------
    stat_wind_u = []
    stat_wind_v = []
    for h in range(n_alt):
        if h == alt_index_std:
            stat_wind_u.append(wind_std[0])
            stat_wind_v.append(wind_std[1])
            print(
                'Altitude: ', alt_axis[h], ' Wind: ',
                stat_wind_u[h], ', ', stat_wind_v[h])
            continue
        # u,vが決まった時のdu,dvの条件付き正規分布の平均と共分散行列
        mu, sigma = getConditionalBivariateNorm(
            mu_X=mu4[h][2:],
            mu_Y=mu4[h][:2],
            sigma_XX=sigma4[h][2:, 2:],
            sigma_XY=sigma4[h][2:, :2],
            sigma_YY=sigma4[h][:2, :2],
            X=wind_std)

        dU, dV = getProbEclipse(
            mu=mu,
            sigma=sigma,
            alpha=0.99)
        w = getAzimuthWindByPlot(
                            dU + wind_std[0],
                            dV + wind_std[1],
                            direction_std)
        stat_wind_u.append(w[0])
        stat_wind_v.append(w[1])
        print(
            'Altitude: ', alt_axis[h], ' ',
            stat_wind_u[h], ', ', stat_wind_v[h])

    stat_wind = np.array([stat_wind_u, stat_wind_v])
    return stat_wind
'''


def getStatWindVector(wind_statistics, wind_direction):
    alt_index_std = wind_statistics['altitude_idx_std']
    alt_axis = wind_statistics['alt_axis']
    # alt_std = alt_axis[alt_index_std]
    n_alt = len(alt_axis)

    mu4 = np.array(wind_statistics['mu4'])
    sigma4 = np.array(wind_statistics['sigma4'])

    # ----------------------------
    # 基準風の導出
    # ----------------------------
    mu_stdalt = mu4[alt_index_std][2:]
    sigma_stdalt = sigma4[alt_index_std][2:, 2:]
    u_std, v_std = getProbEclipse(mu_stdalt, sigma_stdalt, alpha=0.95, n_plots=500)
    wind_std = getAzimuthWindByPlot(u_std, v_std, np.deg2rad(wind_direction))

    # ----------------------------
    # Probabillity Eclipse
    # ----------------------------
    stat_wind_u = []
    stat_wind_v = []
    for h in range(n_alt):
        if h == alt_index_std:
            stat_wind_u.append(wind_std[0])
            stat_wind_v.append(wind_std[1])
            print(
                'Altitude: ', alt_axis[h], ' Wind: ',
                stat_wind_u[h], ', ', stat_wind_v[h])
            continue
        # u,vが決まった時のdu,dvの条件付き正規分布の平均と共分散行列
        mu, sigma = getConditionalBivariateNorm(
            mu_X=mu4[h][2:],
            mu_Y=mu4[h][:2],
            sigma_XX=sigma4[h][2:, 2:],
            sigma_XY=sigma4[h][2:, :2],
            sigma_YY=sigma4[h][:2, :2],
            X=wind_std)

        dU, dV = getProbEclipse(
            mu=mu,
            sigma=sigma,
            alpha=0.99,
            n_plots=500)
        w = getAzimuthWindByPlot(
                            dU + wind_std[0],
                            dV + wind_std[1],
                            np.deg2rad(wind_direction))
        stat_wind_u.append(w[0])
        stat_wind_v.append(w[1])
        print(
            'Altitude: ', alt_axis[h], ' ',
            stat_wind_u[h], ', ', stat_wind_v[h])

    stat_wind = np.array([stat_wind_u, stat_wind_v])
    return stat_wind
