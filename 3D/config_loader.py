import tomllib
import os
from dataclasses import dataclass
from typing import Any

from mesh_generator.mesh_generator import (
    PastePattern, 
    MaterialConfig, 
    MaterialProperties, 
    LayerConfig,
    GeometryParameters
)

@dataclass
class SimulationSettings:
    sim_time: float
    step_time: float
    initial_temp: float
    ambient_temp: float
    water_temp: float
    alpha: float

@dataclass
class FullConfiguration:
    simulation: SimulationSettings
    geometry: GeometryParameters
    materials: MaterialConfig
    layers: LayerConfig
    power: float
    paste_pattern: PastePattern

class ConfigLoader:
    @staticmethod
    def load_from_file(filepath: str) -> FullConfiguration:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(filepath, "rb") as f:
            data = tomllib.load(f)

        sim_data = data.get("simulation", {})
        env_data = data.get("environment", {})
        
        simulation_settings = SimulationSettings(
            sim_time=float(sim_data.get("time", 50.0)),
            step_time=float(sim_data.get("step_time", 1.0)),
            initial_temp=float(sim_data.get("initial_temp", 25.0)),
            ambient_temp=float(env_data.get("ambient_temp", 25.0)),
            water_temp=float(env_data.get("water_temp", 30.0)),
            alpha=float(env_data.get("alpha", 50000.0))
        )

        geo_data = data.get("geometry", {})
        mesh_data = data.get("mesh", {})
        die_data = data.get("die", {})
        die_width = die_data.get("width") or geo_data.get("die_width", 0.015)
        die_depth = die_data.get("depth") or geo_data.get("die_depth", 0.012)
        
        geometry = GeometryParameters(
            width=float(geo_data.get("width", 0.04)),
            depth=float(geo_data.get("depth", 0.04)),
            height=float(geo_data.get("height", 0.03)),
            nx=int(mesh_data.get("nx", 25)),
            ny=int(mesh_data.get("ny", 25)),
            nz=int(mesh_data.get("nz", 30)),
            die_width=float(die_width),
            die_depth=float(die_depth)
        )

        power = float(die_data.get("power", 95.0))
        lay_data = data.get("layers", {})
        layers = LayerConfig(
            silicon=float(lay_data.get("silicon", 20.0)),
            ihs=float(lay_data.get("ihs", 25.0)),
            paste=float(lay_data.get("paste", 5.0))
        )

        paste_data = data.get("paste", {})
        pattern_str = paste_data.get("pattern", "full").upper()
        
        try:
            paste_pattern = PastePattern[pattern_str]
        except KeyError:
            print(f"Warning: Unknown paste pattern '{pattern_str}'. Defaulting to FULL.")
            paste_pattern = PastePattern.FULL

        mat_data = data.get("materials", {})
        
        def get_mat(name: str) -> MaterialProperties:
            m = mat_data.get(name, {})
            return MaterialProperties(
                k=float(m.get("k", 1.0)),
                rho=float(m.get("rho", 1000.0)),
                cp=float(m.get("c", 1000.0))
            )

        materials = MaterialConfig(
            silicon=get_mat("silicon"),
            ihs=get_mat("ihs"),
            paste=get_mat("paste"),
            heatsink=get_mat("heatsink"),
            air=get_mat("air"),
            substrate=get_mat("substrate")
        )

        return FullConfiguration(
            simulation=simulation_settings,
            geometry=geometry,
            materials=materials,
            layers=layers,
            power=power,
            paste_pattern=paste_pattern
        )