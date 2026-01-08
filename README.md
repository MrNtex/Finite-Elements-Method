# Thermal Simulation Engine
> [!IMPORTANT]
> This documentation and configuration guide apply primarily to the **3D Simulation Module**. Directly in the `3D/` directory.
>
> The **2D implementation** is a simplified version with limited configurability and hardcoded parameters. 

## Project Overview
This project implements a numerical solver for transient heat transfer analysis. It is designed to simulate thermal distribution across various mesh topologies (Structured, Unstructured, MixGrid) under defined boundary conditions.

## Configuartion
The repository is organized as follows:

* **`simpulations/`** - Simulation `toml` files. Place your simulation parameters (material data, boundary conditions) here. Contains pre-configured scenarios.
* **`config.py`** - Global parameters for solver and mesh generator.
* **`example/`** - Generated simulation results and logs.
* **`output/`** - Generated simulation results and logs.

## Features
* Support for arbitrary material properties (density, thermal conductivity, specific heat).
* Handling of different mesh types, including complex hybrid meshes.
* Configurable time-stepping and simulation duration.

## Usage

### 1. Configuration
Create a `.toml` config file in the `simulations/` directory. You can use the templates provided in the same folder.

### 2. Running a Simulation
To run the simulation with a specific configuration:

```bash
python main.py simulation/my_simulation.toml
```
*Note: If no simulation files are provided, program uses all the files inside the simulations directory*
