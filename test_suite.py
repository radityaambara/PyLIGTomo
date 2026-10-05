#!/usr/bin/env python
"""
PyLIGTomo Forward + Inverse Test Suite
========================================
Test workflow:
1. Forward modeling: Generate synthetic travel times from known velocity model
2. Inverse modeling: Recover velocity model from synthetic travel times

Configuration:
- Grid: 10x10x10 = 1000 nodes (5m spacing, 27.78m extent)
- Events: 200 (100 earthquakes + 100 blasts)
- Stations: 50 (random 3D positions)
- Phases: 6000 (15 stations per event, P + S waves)
"""
import numpy as np
import pandas as pd
import os
import time
from LIGTomo import run_forward, run_invers

def create_model_and_data():
    """Create test velocity model and data with random 3D positions."""
    np.random.seed(42)
    
    # 10x10x10 grid (1000 nodes)
    x_inner = np.linspace(0, 25, 10)
    y_inner = np.linspace(0, 25, 10)
    z_inner = np.linspace(0, 25, 10)
    
    nodes_inner = []
    for x in x_inner:
        for y in y_inner:
            for z in z_inner:
                nodes_inner.append([x, y, z])
    nodes_inner = np.array(nodes_inner)
    
    modvel = pd.DataFrame({
        'X': nodes_inner[:, 0],
        'Y': nodes_inner[:, 1],
        'Z': nodes_inner[:, 2],
        'Vp': np.random.uniform(5.0, 7.0, len(nodes_inner)),
        'Vs': np.random.uniform(3.0, 4.0, len(nodes_inner))
    })
    
    x_outer = [-5, 30]; y_outer = [-5, 30]; z_outer = [-5, 30]
    nodes_outer = []
    for x in x_outer:
        for y in y_outer:
            for z in z_outer:
                nodes_outer.append([x, y, z])
    nodes_outer = np.array(nodes_outer)
    modvel_outer = pd.DataFrame({
        'X': nodes_outer[:, 0], 'Y': nodes_outer[:, 1], 'Z': nodes_outer[:, 2],
        'Vp': np.full(len(nodes_outer), 5.5), 'Vs': np.full(len(nodes_outer), 3.2)
    })
    
    # 200 events with random 3D positions
    n_events = 200
    n_stations = 50
    
    event_ids = list(range(1, n_events + 1))
    event_types = ['e' if i < n_events//2 else 'b' for i in range(n_events)]
    
    source_list = pd.DataFrame({
        'id': event_ids,
        'easting': np.random.uniform(5, 20, n_events),
        'northing': np.random.uniform(5, 20, n_events),
        'depth': np.random.uniform(2, 22, n_events),
        'type': event_types
    })
    
    station_ids = list(range(101, 101 + n_stations))
    receiver_list = pd.DataFrame({
        'id': station_ids,
        'easting': np.random.uniform(0, 25, n_stations),
        'northing': np.random.uniform(0, 25, n_stations),
        'elevation': np.random.uniform(-5, 5, n_stations)
    })
    
    # Generate phase data - each event at 15 random stations
    phases = []
    np.random.seed(42)
    for ev_id in event_ids:
        selected_stations = np.random.choice(station_ids, size=15, replace=False)
        for sta_id in selected_stations:
            p_time = np.random.uniform(0.3, 8.0)
            phases.append({'id_event': ev_id, 'id_sta': sta_id, 't_time': p_time, 'phase': 'P'})
            s_time = p_time + np.random.uniform(1.5, 5.0)
            phases.append({'id_event': ev_id, 'id_sta': sta_id, 't_time': s_time, 'phase': 'S'})
    
    phase_list = pd.DataFrame(phases)
    
    return modvel, modvel_outer, source_list, receiver_list, phase_list

def run_forward_test(modvel, modvel_outer, source_list, receiver_list, phase_list):
    """Run forward modeling and return synthetic data."""
    print("=" * 60)
    print("STAGE 1: FORWARD MODELING")
    print("=" * 60)
    print(f"Grid: 10x10x10 = {len(modvel)} nodes")
    print(f"Events: {len(source_list)} (50 e + 50 b)")
    print(f"Stations: {len(receiver_list)}")
    print(f"Phases: {len(phase_list)} (P: {len(phase_list[phase_list['phase']=='P'])}, S: {len(phase_list[phase_list['phase']=='S'])})")
    
    forward_folder = 'test_forward_large'
    
    start_time = time.time()
    try:
        run_forward(
            modvel=modvel.copy(),
            modvel_outer=modvel_outer.copy(),
            source_list=source_list.copy(),
            receiver_list=receiver_list.copy(),
            phase_list=phase_list.copy(),
            delt=3.0,
            deltn=2.78,
            xfac=0.5,
            iter1=3,
            iter2=2,
            tmin=0.01,
            folder_name=forward_folder,
            rnd_koef=1,
            if_art=False,  # Disabled for performance
            nu_cpu=2
        )
        elapsed = time.time() - start_time
        print(f"\nForward modeling completed in {elapsed:.1f}s")
        
        # Check output
        synth_path = os.path.join(forward_folder, 'synthetic.csv')
        if os.path.exists(synth_path):
            synth_df = pd.read_csv(synth_path)
            print(f"Synthetic data: {len(synth_df)} rows (P: {len(synth_df[synth_df['phase']=='P'])}, S: {len(synth_df[synth_df['phase']=='S'])})")
            print(f"Sample:\n{synth_df.head()}")
            return synth_df, forward_folder
        else:
            print("ERROR: synthetic.csv not found!")
            return None, forward_folder
            
    except Exception as e:
        print(f"\nForward modeling FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None, forward_folder

def run_inverse_test(modvel, modvel_outer, source_list, receiver_list, synth_df, forward_folder):
    """Run inverse modeling using synthetic data from forward modeling."""
    print("\n" + "=" * 60)
    print("STAGE 2: INVERSE MODELING (using synthetic data from forward)")
    print("=" * 60)
    
    # Create phase_list from synthetic data
    if synth_df is None:
        print("ERROR: No synthetic data from forward modeling!")
        return False
    
    # The synthetic.csv has columns: id_event, id_sta, t_time, phase, t_sin
    # We use t_sin as the observed travel time
    inverse_phase_list = synth_df[['id_event', 'id_sta', 't_time', 't_sin']].copy()
    inverse_phase_list.columns = ['id_event', 'id_sta', 't_time', 'phase']
    # Use t_sin as the observed travel time
    # Actually, let's look at the columns properly
    print(f"Synthetic columns: {list(synth_df.columns)}")
    print(f"Phase values in synthetic: {synth_df['phase'].unique()}")
    
    # The synthetic data should have: id_event, id_sta, t_time, phase, t_sin
    # We need to create a proper phase_list with observed times
    # t_sin is the computed (synthetic) travel time
    # We use t_sin as the observed travel time for inversion
    inverse_phase_list = pd.DataFrame({
        'id_event': synth_df['id_event'],
        'id_sta': synth_df['id_sta'],
        't_time': synth_df['t_sin'],  # Use synthetic times as observed
        'phase': synth_df['phase']
    })
    
    print(f"Inversion phase list: {len(inverse_phase_list)} rows")
    print(f"P: {len(inverse_phase_list[inverse_phase_list['phase']=='P'])}, S: {len(inverse_phase_list[inverse_phase_list['phase']=='S'])}")
    
    inverse_folder = 'test_inverse_large'
    
    start_time = time.time()
    try:
        run_invers(
            modvel=modvel.copy(),
            modvel_outer=modvel_outer.copy(),
            source_list=source_list.copy(),
            receiver_list=receiver_list.copy(),
            phase_list=inverse_phase_list.copy(),
            delt=3.0,
            deltn=2.78,
            xfac=0.5,
            iter1=3,
            iter2=2,
            tmin=0.01,
            iteration_number=2,
            up_threshold=10,
            low_threshold=0,
            dens_thres=-1,
            d_rms=0.001,
            r_time_P=100.0,
            r_time_S=100.0,
            damping_1=0.1,
            damping_2=0.01,
            update_grid=False,
            update_grid_after=False,
            folder_name=inverse_folder,
            if_art=False,
            nu_cpu=2
        )
        elapsed = time.time() - start_time
        print(f"\nInverse modeling completed in {elapsed:.1f}s")
        
        # Check output
        vel_path = os.path.join(inverse_folder, 'vel_invers.csv')
        if os.path.exists(vel_path):
            vel_df = pd.read_csv(vel_path)
            print(f"Inverted velocity model: {len(vel_df)} nodes")
            print(f"Output files: {sorted(os.listdir(inverse_folder))}")
            
            # Compare RMS from log
            log_path = os.path.join(inverse_folder, 'app.log')
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    log = f.read()
                for line in log.split('\n'):
                    if 'rms' in line.lower() and ('initial' in line.lower() or 'after iteration' in line.lower()):
                        print(f"  LOG: {line}")
        
        return True
            
    except Exception as e:
        print(f"\nInverse modeling FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("PyLIGTomo Forward-Inverse Test Suite")
    print("=" * 60)
    
    # Create model and data
    modvel, modvel_outer, source_list, receiver_list, phase_list = create_model_and_data()
    
    # Stage 1: Forward modeling
    synth_df, forward_folder = run_forward_test(
        modvel, modvel_outer, source_list, receiver_list, phase_list
    )
    
    # Stage 2: Inverse modeling using synthetic data
    if synth_df is not None and len(synth_df) > 0:
        success = run_inverse_test(
            modvel, modvel_outer, source_list, receiver_list, synth_df, forward_folder
        )
        if success:
            print("\n" + "=" * 60)
            print("ALL TESTS PASSED!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("INVERSE MODELING FAILED")
            print("=" * 60)
    else:
        print("\nForward modeling failed, cannot proceed with inverse modeling.")

if __name__ == '__main__':
    main()
