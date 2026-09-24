#import libraries
import pandas as pd
import numpy as np
import baseflow #baseflow package:


def process_baseflow_station(
    nameStation,
    df_obs,
    sim_file,
    station_locations_unique
):
    """
    Extract baseflow and BFI for all separation methods for one station.

    The best method is selected from the OBSERVED streamflow
    using the highest KGE. The same method is then used for
    the observed-simulated baseflow comparison.
    """

    # ---------------------------------------------
    # Read simulated discharge
    # ---------------------------------------------
    df_sim = pd.read_csv(
        sim_file,
        parse_dates=["time"],
        index_col="time"
    )

    # ---------------------------------------------
    # Coordinates
    # ---------------------------------------------
    lat = station_locations_unique.loc[nameStation, "lat"]
    lon = station_locations_unique.loc[nameStation, "lon"]

    # ---------------------------------------------
    # OBSERVED:
    # run ALL available baseflow methods
    # ---------------------------------------------
    bf_obs_all, bfi_obs, kge_obs = baseflow.separation(
        df_obs,
        return_bfi=True,
        return_kge=True
    )

    # Convert one-row DataFrames to Series
    bfi_obs_s = bfi_obs.iloc[0].astype(float)
    kge_obs_s = kge_obs.iloc[0].astype(float)

    # ---------------------------------------------
    # Select best method from OBSERVED data
    # ---------------------------------------------
    best_method = kge_obs_s.idxmax()
    best_kge = kge_obs_s.loc[best_method]

    # ---------------------------------------------
    # SIMULATED:
    # also run ALL methods so that we retain
    # method-specific simulated BFI
    # ---------------------------------------------
    bf_sim_all, bfi_sim, kge_sim = baseflow.separation(
        df_sim,
        return_bfi=True,
        return_kge=True
    )

    bfi_sim_s = bfi_sim.iloc[0].astype(float)

    # ---------------------------------------------
    # Ensure selected method exists for simulation
    # ---------------------------------------------
    if best_method not in bf_sim_all:
        raise KeyError(
            f"{best_method} not available for simulated "
            f"data at {nameStation}"
        )

    # ---------------------------------------------
    # Extract baseflow series using SAME method
    # for observed and simulated data
    # ---------------------------------------------
    qbf_obs = bf_obs_all[best_method]
    qbf_sim = bf_sim_all[best_method]

    # normalize Series/DataFrames
    if isinstance(df_obs, pd.DataFrame):
        q_obs = df_obs.iloc[:, 0].rename("Q")
    else:
        q_obs = df_obs.rename("Q")

    if isinstance(qbf_obs, pd.DataFrame):
        qbf_obs = qbf_obs.iloc[:, 0]

    if isinstance(qbf_sim, pd.DataFrame):
        qbf_sim = qbf_sim.iloc[:, 0]

    qbf_obs = qbf_obs.rename("Q_bf_obs")
    qbf_sim = qbf_sim.rename("Q_bf_sim")

    df_station = pd.concat(
        [q_obs, qbf_obs, qbf_sim],
        axis=1
    )

    return {
        "station": nameStation,
        "timeseries": df_station,

        # every BFI
        "bfi_obs": bfi_obs_s,
        "bfi_sim": bfi_sim_s,

        # every observed KGE
        "kge_obs": kge_obs_s,

        # station-specific selection
        "best_method": best_method,
        "best_kge": best_kge,

        # coordinates
        "lon": lon,
        "lat": lat
    }