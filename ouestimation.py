# ouestimation.py
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# ---------- CONFIG ----------
TICKERS = ["PEP", "KO"]
START = "2018-01-01"
END = None  # None -> up to today
DT = 1/252.0  # daily sampling ~ 252 trading days/year
WINDOW = None  
# --- Prediction Config ---
PREDICTION_DAYS = 60       
N_PATHS_HISTORICAL = 20    
N_PATHS_FUTURE = 1000 

def download_prices(tickers, start, end):
    """
    Download prices and return a DataFrame of adjusted-close prices (tickers as columns).
    Handles these yfinance shapes:
      - MultiIndex columns with top level 'Adj Close' (multiple tickers, auto_adjust=False)
      - Single-level columns (tickers) when auto_adjust=True
      - Single ticker Series -> converted to DataFrame
    """
    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)
    # If yfinance returned MultiIndex (e.g. ('Adj Close', 'PEP'), ('Adj Close','KO'), ...)
    if isinstance(data.columns, pd.MultiIndex):
        # Prefer 'Adj Close' if present
        if 'Adj Close' in data.columns.levels[0]:
            df = data['Adj Close']
        # Fallback to 'Close' if 'Adj Close' not present
        elif 'Close' in data.columns.levels[0]:
            df = data['Close']
        else:
            # Last resort: collapse to single-level by taking first level's data (not ideal)
            # but usually 'Close' or 'Adj Close' will exist.
            first_level = data.columns.levels[0][0]
            df = data[first_level]
    else:
        # Single-level columns: usually already adjusted/close and columns are tickers
        df = data

    # If a single Series (single ticker), convert to DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame()

    # Drop rows with any NA (you can change behavior if you prefer forward-fill)
    df = df.dropna(how='any')
    return df

def compute_log_spread(df, t1, t2):
    s = np.log(df[t1]) - np.log(df[t2])
    s = s.dropna()
    return s

def estimate_ou_params_from_series(x, dt=DT):
    """
    Estimate OU parameters (kappa, theta, sigma) from evenly spaced series x
    using the exact AR(1) solution.
    
    Returns: dict of parameters
    """
    x = np.asarray(x, dtype=float)
    x_prev = x[:-1]
    x_next = x[1:]
    
    # Linear regression x_next = a*x_prev + b
    A = np.vstack([x_prev, np.ones_like(x_prev)]).T
    sol, *_ = np.linalg.lstsq(A, x_next, rcond=None)
    a, b = sol  # slope and intercept
    
    # Recover parameters
    if a <= 0 or a >= 1:
        # Handle non-stationary cases
        kappa = np.nan
        theta = np.nan
    else:
        # kappa = -ln(a) / dt
        kappa = -np.log(a) / dt
        # theta = b / (1 - a)
        theta = b / (1 - a)
    
    # residuals and sigma
    resid = x_next - (a * x_prev + b)
    var_eps = np.var(resid, ddof=1) # Note: ddof=1 for sample variance
    
    # variance relation: var_eps = (sigma^2 / (2*kappa)) * (1 - a**2)
    # => sigma = sqrt( var_eps * 2*kappa / (1 - a**2) )
    if not np.isnan(kappa) and (1 - a**2) > 0:
        sigma_sq = max(var_eps * 2 * kappa / (1 - a**2), 0.0)
        sigma = np.sqrt(sigma_sq)
    else:
        sigma = np.nan
        
    return {"kappa": kappa, "theta": theta, "sigma": sigma, "a_slope": a, "b_intercept": b, "resid_var": var_eps}

def euler_maruyama_ou_paths(x0, kappa, theta, sigma, dt, n_steps, n_paths=10, random_seed=None):
    """
    Simulate multiple Euler–Maruyama OU paths.
    Returns: array shape (n_steps+1, n_paths)
    """
    rng = np.random.default_rng(random_seed)
    sqrt_dt = np.sqrt(dt)
    xs = np.empty((n_steps + 1, n_paths))
    xs[0, :] = x0

    for i in range(n_steps):
        dW = rng.standard_normal(n_paths)
        # EM step: dS = kappa(theta - S)dt + sigma*dW
        xs[i + 1, :] = xs[i, :] + kappa * (theta - xs[i, :]) * dt + sigma * sqrt_dt * dW

    return xs


def main():
    df = download_prices(TICKERS, START, END)
    print("Downloaded data shape:", df.shape)

    # compute log spread (PEP - KO)
    spread = compute_log_spread(df, TICKERS[0], TICKERS[1])
    print("First 5 spread values:\n", spread.head())

    # ADF test for stationarity (optional)
    adf_res = adfuller(spread)
    print(f"ADF statistic: {adf_res[0]:.4f}, p-value: {adf_res[1]:.4f}")

    # Estimate OU params
    est = estimate_ou_params_from_series(spread.values, dt=DT)
    print("Estimated params (spread):")
    for k, v in est.items():
        print(f"  {k}: {v:.6f}")

    # <-- START: PLOT 1 - HISTORICAL SIMULATION -->
    print("\nGenerating historical simulation plot...")
    
    n_steps_hist = len(spread) - 1
    hist_paths = euler_maruyama_ou_paths(
        spread.values[0], # Start from the beginning
        est["kappa"], est["theta"], est["sigma"], # Use notation-matched params
        DT, n_steps_hist, n_paths=N_PATHS_HISTORICAL, random_seed=42
    )

    # Compute mean and std bands for historical sim
    hist_sim_mean = hist_paths.mean(axis=1)
    hist_sim_std = hist_paths.std(axis=1)
    hist_upper_band = hist_sim_mean + 1.0 * hist_sim_std # 1-sigma band
    hist_lower_band = hist_sim_mean - 1.0 * hist_sim_std

    # Plot Historical Simulation
    plt.figure(figsize=(12, 5))
    plt.plot(spread.index, spread.values, label="Observed", color="blue", linewidth=1.5)
    # Plot all paths (semi-transparent)
    plt.plot(spread.index, hist_paths, color="orange", alpha=0.2)
    # Plot mean and 1σ band
    plt.plot(spread.index, hist_sim_mean, color="red", label="Mean simulation", linewidth=2)
    plt.fill_between(spread.index, hist_lower_band, hist_upper_band, color="red", alpha=0.15, label="±1σ band")
    plt.title(f"OU Historical Simulation: Log-Spread ({TICKERS[0]} - {TICKERS[1]})")
    plt.legend()
    plt.tight_layout()
    plt.show()
    # <-- END: PLOT 1 -->


    # --- Start: Future Prediction Logic ---
    print("\nGenerating future prediction cone plot...")
    
    # Get the last known value as the starting point for simulation
    x0_future = spread.values[-1]
    last_date = spread.index[-1]

    # Simulate future paths
    future_paths = euler_maruyama_ou_paths(
        x0_future, # Start from the last known value
        est["kappa"], est["theta"], est["sigma"], # Use notation-matched params
        DT,
        PREDICTION_DAYS,  # Number of steps
        n_paths=N_PATHS_FUTURE,
        random_seed=42
    )

    # Create a future date range for the x-axis
    future_dates = pd.date_range(
        start=last_date,
        periods=PREDICTION_DAYS + 1, # +1 to include the starting point
        freq='B' # BusinessDay
    )

    # Calculate the mean
    future_mean = future_paths.mean(axis=1)
    
    # <-- START: MODIFICATION FOR 1-SIGMA and 2-SIGMA CONES -->
    future_std = future_paths.std(axis=1)
    
    # Calculate the upper and lower bounds for 1-sigma cone
    upper_cone_1std = future_mean + 1.0 * future_std
    lower_cone_1std = future_mean - 1.0 * future_std
    
    # Calculate the upper and lower bounds for 2-sigma cone
    upper_cone_2std = future_mean + 2.0 * future_std
    lower_cone_2std = future_mean - 2.0 * future_std

    # --- Start: PLOT 2 - FUTURE PREDICTION CONE ---
    plt.figure(figsize=(14, 7)) # Wider figure to see future
    
    # 1. Plot historical observed spread
    plt.plot(spread.index, spread.values, label="Observed Spread", color="blue", linewidth=1.5)

    # 2. Plot the predicted mean path
    plt.plot(future_dates, future_mean, color="red", label="Predicted Mean Path", linestyle="--", linewidth=2)

    # <-- START: MODIFICATION TO PLOT BOTH CONES -->
    # 3. Plot the 2-sigma cone of uncertainty (wider, more transparent)
    #    We plot this one FIRST so the 1-sigma cone can be drawn on top.
    plt.fill_between(
        future_dates,
        lower_cone_2std,
        upper_cone_2std,
        color="red",
        alpha=0.15, # Lighter
        label="±2σ Prediction Cone (~95%)"
    )

    # 4. Plot the 1-sigma cone of uncertainty (narrower, less transparent)
    plt.fill_between(
        future_dates,
        lower_cone_1std,
        upper_cone_1std,
        color="red",
        alpha=0.3, # Darker
        label="±1σ Prediction Cone (~68%)"
    )
    # <-- END: MODIFICATION -->

    # 5. Plot the estimated long-term mean (theta) as a horizontal line
    plt.axhline(
        est["theta"],
        color="green",
        linestyle=":",
        linewidth=2,
        label=f"Estimated Mean (θ={est['theta']:.4f})"
    )
    
    plt.title(f"OU Prediction Cone: Log-Spread ({TICKERS[0]} - {TICKERS[1]})")
    plt.ylabel("Log Spread")
    plt.xlabel("Date")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    # --- End: PLOT 2 ---


if __name__ == "__main__":
    main()