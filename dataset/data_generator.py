import random
import numpy as np
from scipy.stats import levy_stable
from fbm import FBM

""" Generate non-stationary stochastic time series """

def generate_nonstationary_fou(total_length=10000,
                               min_seg=10, max_seg=50,
                               theta_range=(0.0, 0.0),
                               mu_range=(0, 0),
                               sigma_range=(0.3, 1.0),
                               H_range=(0.6, 0.9),
                               dt=1.0, seed=2025,
                               max_trials=1000):

    random.seed(seed)
    np.random.seed(seed)

    series = []
    t = 0
    x_last = 0.0
    used_params = set()

    def round_param(p, decimals=4):
        return tuple(round(x, decimals) for x in p)

    while t < total_length:
        seg_len = random.randint(min_seg, max_seg)
        if t + seg_len > total_length:
            seg_len = total_length - t

        # try new parameter combination
        for _ in range(max_trials):
            theta = random.uniform(*theta_range)
            mu    = random.uniform(*mu_range)
            sigma = random.uniform(*sigma_range)
            H     = random.uniform(*H_range)

            params_rounded = round_param((theta, mu, sigma, H))
            if params_rounded not in used_params:
                used_params.add(params_rounded)
                break
        else:
            raise RuntimeError("Exceeded maximum attempts to find a new parameter combination.")

        f = FBM(n=seg_len, hurst=H, length=seg_len*dt, method='daviesharte')
        B = f.fbm()

        X_seg = np.zeros(seg_len, dtype=np.float32)
        X_seg[0] = x_last
        for i in range(1, seg_len):
            dB = B[i] - B[i-1]
            X_seg[i] = X_seg[i-1] + theta * (mu - X_seg[i-1]) * dt + sigma * dB

        series.append(X_seg)
        x_last = float(X_seg[-1])
        t += seg_len

    return np.concatenate(series)


def generate_nonstationary_fou_levy(
        total_length=10000,
        min_seg=10, max_seg=50,
        theta_range=(0.0, 0.0),
        mu_range=(0, 0),
        sigma_range=(0.3, 1.0),
        H_range=(0.0, 0.0),
        alpha_range=(1.2, 1.8),       
        beta_range=(-1.0, 1.0),      
        dt=1.0, seed=2025,
        max_trials=1000):
    
    random.seed(seed)
    np.random.seed(seed)

    series = []
    t = 0
    x_last = 0.0
    used_params = set()

    def round_param(p, decimals=4):
        return tuple(round(x, decimals) for x in p)

    while t < total_length:
        seg_len = random.randint(min_seg, max_seg)
        if t + seg_len > total_length:
            seg_len = total_length - t

        for _ in range(max_trials):
            theta = random.uniform(*theta_range)
            mu    = random.uniform(*mu_range)
            sigma = random.uniform(*sigma_range)
            H     = random.uniform(*H_range)
            alpha_i = random.uniform(*alpha_range)
            beta_i  = random.uniform(*beta_range)

            params = (theta, mu, sigma, H, alpha_i, beta_i)
            pr = round_param(params)
            if pr not in used_params:
                used_params.add(pr)
                break
        else:
            raise RuntimeError("Exceeded max_trials for new params.")

        dL = levy_stable.rvs(
            alpha_i, beta_i,
            scale=sigma * (dt**(1/alpha_i)),
            size=seg_len
        ).astype(np.float32)

        X_seg = np.zeros(seg_len, dtype=np.float32)
        X_seg[0] = x_last
        for i in range(1, seg_len):
            X_seg[i] = (
                X_seg[i-1]
                + theta * (mu - X_seg[i-1]) * dt
                + dL[i]
            )

        series.append(X_seg)
        x_last = float(X_seg[-1])
        t += seg_len

    return np.concatenate(series)


def generate_nonstationary_fou_mixed(
        total_length=10000,
        min_seg=10, max_seg=50,
        theta_range=(0.0, 0.0),
        mu_range=(0, 0),
        sigma_bm_range=(0.3, 1.0),    
        sigma_levy_range=(0.3, 1.0),  
        H_bm_range=(0.6, 0.9),        
        H_levy_range=(0.6, 0.9),      
        alpha_range=(1.2, 1.8),       
        beta_range=(-1.0, 1.0),       
        dt=1.0, seed=2025,
        max_trials=1000):

    random.seed(seed)
    np.random.seed(seed)

    series = []
    t = 0
    x_last = 0.0
    used_params = set()

    def round_param(p, d=4):
        return tuple(round(x, d) for x in p)

    while t < total_length:
        seg_len = random.randint(min_seg, max_seg)
        if t + seg_len > total_length:
            seg_len = total_length - t

        for _ in range(max_trials):
            theta     = random.uniform(*theta_range)
            mu        = random.uniform(*mu_range)
            sigma_bm  = random.uniform(*sigma_bm_range)
            sigma_lv  = random.uniform(*sigma_levy_range)
            H_bm      = random.uniform(*H_bm_range)
            H_levy    = random.uniform(*H_levy_range)
            alpha_i   = random.uniform(*alpha_range)
            beta_i    = random.uniform(*beta_range)

            params = (theta, mu, sigma_bm, sigma_lv,
                      H_bm, H_levy, alpha_i, beta_i)
            pr = round_param(params)
            if pr not in used_params:
                used_params.add(pr)
                break
        else:
            raise RuntimeError("Exceeded max_trials for new params.")

        fbm = FBM(n=seg_len, hurst=H_bm, length=seg_len*dt, method='daviesharte')
        B = fbm.fbm().astype(np.float32)
        dB = np.diff(B, prepend=0)

        dL = levy_stable.rvs(
            alpha_i, beta_i,
            scale=sigma_lv * (dt**H_levy),
            size=seg_len
        ).astype(np.float32)

        X_seg = np.zeros(seg_len, dtype=np.float32)
        X_seg[0] = x_last
        for i in range(1, seg_len):
            X_seg[i] = (
                X_seg[i-1]
                + theta * (mu - X_seg[i-1]) * dt
                + sigma_bm * dB[i]
                + dL[i]
            )

        series.append(X_seg)
        x_last = float(X_seg[-1])
        t += seg_len


    return np.concatenate(series)
