from dataclasses import dataclass
from fem_types import GlobalData

@dataclass
class MaterialConstants:
    K_SILICON: float = 150.0
    K_IHS: float     = 380.0
    K_PASTE: float   = 8.5
    K_AIR: float     = 0.026
    K_HEATSINK: float= 200.0

def get_global_data() -> GlobalData:
    return GlobalData(
        SimulationTime=100.0,
        SimulationStepTime=0.1,
        Conductivity=0.0,
        Alfa=25.0,
        Tot=25.0,
        InitialTemp=100.0,
        Density=2330.0,
        SpecificHeat=700.0
    )