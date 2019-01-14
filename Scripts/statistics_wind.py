import numpy as np
import numpy.linalg as LA
import sympy
# direction_std = 0.5 * np.pi


def getConditionalBivariateNorm(mu_X, mu_Y, sigma_XX, sigma_XY, sigma_YY, X):
    # Y|X が多変量正規分布N(mu, Sigma)に従う場合のmu, Sigmaを求める
    sigma_XX_i = LA.inv(sigma_XX)
    mu = mu_Y + np.dot(np.dot(sigma_XY.T, sigma_XX_i), (X - mu_X))
    Sigma = sigma_YY - np.dot(np.dot(sigma_XY.T, sigma_XX_i), sigma_XY)
    return mu, Sigma


def getEllipseParameters(mu, sigma, alpha=0.95):
    # mu: 平均 ([alt][x|y])
    # sigma: 分散共分散行列
    # alpha: 楕円内に点が入る確率

    sigma_i = LA.inv(sigma)
    eigvals, M = LA.eig(sigma_i)
    lmda2 = -2*np.log(1.00000000 - alpha)
    scale = np.sqrt(lmda2/eigvals)
    return scale, M, mu


def getEllipsePlot(a, b, Mrot=None, angle_rad=0, x0=0, y0=0, n_plots=100):
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


def getAzimuthWindByEllipse(center_pos, scale, rotateMatrix, wind_direction):
    # 媒介変数表示による楕円 U(t), V(t)と V = alpha * U (alphaは飛行方位の傾き)の関係から
    # 方程式を立ててtについて解き、U,V を求める。
    t = sympy.Symbol('t')
    x = scale[0] * sympy.cos(t)
    y = scale[1] * sympy.sin(t)
    U = rotateMatrix[0, 0] * x + rotateMatrix[0, 1] * y + center_pos[0]
    V = rotateMatrix[1, 0] * x + rotateMatrix[1, 1] * y + center_pos[1]
    alpha = sympy.cot(wind_direction)
    print('alpha: ', alpha)
    expr = V - alpha * U
    print(expr)
    # expr3 = V - alpha * U
    t_solutions_tmp = sympy.solve(expr)
    t_solutions = [float(t) for t in t_solutions_tmp]
    print('solutions: ', t_solutions)

    n_solutions = len(t_solutions)
    wind_tmp = np.zeros((n_solutions, 2))
    for i, t_solution in enumerate(t_solutions):
        wind_tmp[i] = [
                        float(U.subs(t, t_solution)),
                        float(V.subs(t, t_solution))]
        print('U(t=', t_solution, '): ', U.subs(t, t_solution))
        print('V(t=', t_solution, '): ', V.subs(t, t_solution))

    print('wind_tmp: ', wind_tmp)
    az_wind_directions = np.arctan2(-wind_tmp.T[0], -wind_tmp.T[1])
    direction_diff = np.round(az_wind_directions - wind_direction, 8)
    mask = direction_diff == 0
    print('mask: ', mask)
    if not mask.any():
        idx = np.argmax(LA.norm(wind_tmp.T, axis=0))
        azimuth_wind = wind_tmp[idx]
    else:
        idx = np.argmax(LA.norm(wind_tmp[mask].T, axis=0))
        azimuth_wind = wind_tmp[idx]
    # print('norm: ', LA.norm(wind_tmp.T, axis=0))
    # max_index = np.argmax(LA.norm(azimuth_wind.T, axis=0))
    # max_index = np.argmax(LA.norm(wind_tmp.T, axis=0))
    # wind = np.max()
    print('azimuth_wind: ', azimuth_wind)
    return azimuth_wind


def getProbEllipse(mu, sigma, alpha=0.95, n_plots=100):
    # mu: 平均 ([alt][x|y])
    # sigma: 分散共分散行列
    # alpha: 楕円内に点が入る確率
    # n_plots: プロット数

    scale, M, _ = getEllipseParameters(mu, sigma, alpha)
    plots = getEllipsePlot(
        scale[0],
        scale[1],
        M,
        x0=mu[0],
        y0=mu[1],
        n_plots=n_plots)
    return plots[0], plots[1]


'''
def getStatWindVector(wind_statistics, wind_std):
    alt_index_std = wind_statistics['altitude_idx_std']
    alt_axis = wind_statistics['alt_axis']
    # alt_std = alt_axis[alt_index_std]
    n_alt = len(alt_axis)

    mu4 = np.array(wind_statistics['mu4'])
    sigma4 = np.array(wind_statistics['sigma4'])

    # ----------------------------
    # Probabillity Ellipse
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

        dU, dV = getProbEllipse(
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


def getStatWindVector(statistics_parameters, wind_direction_deg):
    alt_index_std = statistics_parameters['altitude_idx_std']
    alt_axis = statistics_parameters['alt_axis']
    # alt_std = alt_axis[alt_index_std]
    n_alt = len(alt_axis)

    mu4 = np.array(statistics_parameters['mu4'])
    sigma4 = np.array(statistics_parameters['sigma4'])

    # ----------------------------
    # 基準風の導出
    # ----------------------------
    mu_stdalt = mu4[alt_index_std][2:]
    sigma_stdalt = sigma4[alt_index_std][2:, 2:]
    # u_std, v_std = getProbEllipse(mu_stdalt, sigma_stdalt, alpha=0.95, n_plots=500)
    # wind_std = getAzimuthWindByPlot(u_std, v_std, np.deg2rad(wind_direction))
    scale, M, _ = getEllipseParameters(mu_stdalt, sigma_stdalt, alpha=0.95)
    wind_std = getAzimuthWindByEllipse(
                        mu_stdalt,
                        scale,
                        M,
                        np.deg2rad(wind_direction_deg)
                        )

    # ----------------------------
    # Probabillity Ellipse
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

        scale, M, _ = getEllipseParameters(mu, sigma, alpha=0.99)
        w = getAzimuthWindByEllipse(
                            mu,
                            scale,
                            M,
                            np.deg2rad(wind_direction_deg)
                            )

        stat_wind_u.append(w[0])
        stat_wind_v.append(w[1])
        print(
            'Altitude: ', alt_axis[h], ' ',
            stat_wind_u[h], ', ', stat_wind_v[h])

    stat_wind = np.array([stat_wind_u, stat_wind_v])
    return stat_wind
