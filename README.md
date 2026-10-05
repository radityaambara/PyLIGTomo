# PyLIGTomo

**Adaptive Irregular Grid Local Earthquake Tomography using Voronoi-based Grids**

PyLIGTomo is a Python package for performing local earthquake tomography using adaptive irregular grids based on Voronoi tessellation. It supports 3D P-wave and S-wave velocity structure inversion with flexible node placement and anisotropic velocity parametrization.

## Features

- Voronoi-based adaptive grid for efficient 3D velocity model parametrization
- Joint inversion of P-wave (Vp) and S-wave (Vs) travel times
- Support for double-difference (DD) and absolute travel time inversions
- Automatic node removal based on ray hit count (RHC) and density tensor criteria
- Anisotropic velocity parametrization with configurable damping
- Parallel processing support using multiple CPU cores
- VTK output for 3D visualization of velocity models and ray paths
- Synthetic data generation (forward modeling) for testing and validation

## Installation

```bash
pip install https://github.com/radityaambara/PyLIGTomo
```

Or install from source:

```bash
git clone https://github.com/radityaambara/PyLIGTomo.git
cd PyLIGTomo
pip install -e .
```

## Requirements

- Python >= 3.8
- numpy
- scipy
- matplotlib
- pyevtk
- pandas

## Quick Start

### Forward Modeling (Synthetic Data Generation)

```python
import numpy as np
import pandas as pd
from LIGTomo import run_forward

# Define velocity model
modvel = pd.DataFrame({
    'X': [0, 5, 10, 15, 20],
    'Y': [0, 5, 10, 15, 20],
    'Z': [0, 0, 0, 0, 0],
    'Vp': [5.5, 5.8, 6.0, 5.9, 5.6],
    'Vs': [3.2, 3.4, 3.5, 3.4, 3.2]
})

# Define events and stations
source_list = pd.DataFrame({
    'id': [1],
    'easting': [10.0],
    'northing': [10.0],
    'depth': [5.0],
    'type': ['e']
})

receiver_list = pd.DataFrame({
    'id': [101],
    'easting': [5.0],
    'northing': [5.0],
    'elevation': [0.0]
})

phase_list = pd.DataFrame({
    'id_event': [1],
    'id_sta': [101],
    't_time': [1.0],
    'phase': ['P']
})

# Run forward modeling
run_forward(
    modvel=modvel,
    modvel_outer=modvel,
    source_list=source_list,
    receiver_list=receiver_list,
    phase_list=phase_list,
    delt=2.0,
    deltn=1.0,
    xfac=0.5,
    iter1=2,
    iter2=2,
    tmin=0.01,
    folder_name='test_forward',
    if_art=False,
    nu_cpu=2
)
```

### Inverse Modeling (Velocity Structure Recovery)

```python
from LIGTomo import run_invers

run_invers(
    modvel=modvel,
    modvel_outer=modvel,
    source_list=source_list,
    receiver_list=receiver_list,
    phase_list=phase_list,
    delt=2.0,
    deltn=1.0,
    xfac=0.5,
    iter1=2,
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
    folder_name='test_invers',
    if_art=False,
    nu_cpu=2
)
```

### Complete Test Suite

A complete test script (`test_suite.py`) is provided that runs both forward modeling and inverse modeling in a two-stage workflow:

1. **Stage 1**: Generate synthetic travel times from a known velocity model
2. **Stage 2**: Invert the synthetic data to recover the velocity structure

```bash
python test_suite.py
```

## Input Data Format

### Velocity Model (`modvel`)
A pandas DataFrame with columns:
- `X`: Node easting coordinate (meters)
- `Y`: Node northing coordinate (meters)
- `Z`: Node depth (meters)
- `Vp`: P-wave velocity (km/s)
- `Vs`: S-wave velocity (km/s)

### Source List (`source_list`)
A pandas DataFrame with columns:
- `id`: Event ID (integer)
- `easting`: Event easting coordinate (meters)
- `northing`: Event northing coordinate (meters)
- `depth`: Event depth (meters)
- `type`: Event type (`'e'` for earthquake, `'b'` for blast)

### Receiver List (`receiver_list`)
A pandas DataFrame with columns:
- `id`: Station ID (integer)
- `easting`: Station easting coordinate (meters)
- `northing`: Station northing coordinate (meters)
- `elevation`: Station elevation (meters)

### Phase List (`phase_list`)
A pandas DataFrame with columns:
- `id_event`: Event ID
- `id_sta`: Station ID
- `t_time`: Travel time (seconds)
- `phase`: Phase type (`'P'` or `'S'`)

## Parameters

### Common Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `delt` | Grid spacing in X/Y direction | - |
| `deltn` | Grid spacing in Z direction | - |
| `xfac` | Node expansion factor | 0.5 |
| `iter1` | Smoothing iterations for RHC | 3 |
| `iter2` | Smoothing iterations for density | 2 |
| `tmin` | Minimum travel time threshold | 0.01 |
| `if_art` | Enable analytical ray tracing (slower but accurate) | False |
| `nu_cpu` | Number of CPU cores | 1 |

### Inversion-Specific Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `iteration_number` | Number of inversion iterations | 2 |
| `up_threshold` | RHC threshold for adding nodes | 10 |
| `low_threshold` | RHC threshold for removing nodes | 0 |
| `dens_thres` | Density tensor threshold for node removal | -1 |
| `d_rms` | RMS damping parameter | 0.001 |
| `damping_1` | Damping for hypocenter coordinates | 0.1 |
| `damping_2` | Damping for velocity perturbations | 0.01 |
| `update_grid` | Update grid nodes after inversion | False |

## Output Files

Both `run_forward` and `run_invers` generate the following output files in the specified `folder_name` directory:

- `synthetic.csv` / `vel_invers.csv`: Inverted/synthetic velocity model
- `source_invers.csv`: Inverted event locations
- `statistical.png`: RMS evolution across iterations
- `RHC hist P.png`, `RHC hist S.png`: Ray hit count histograms
- `_ray_initial.vtk`, `_ray_final.vtk`: Ray path output (VTK format)
- `app.log`: Detailed inversion log
- `compress.zip`: Compressed VTK outputs


