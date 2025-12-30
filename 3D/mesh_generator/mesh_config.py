from dataclasses import dataclass
from fem_types import GlobalData

@dataclass
class MaterialConstants:
    # Thermal Conductivity in W/mK
    K_SILICON: float = 150.0
    K_IHS: float     = 380.0
    K_PASTE: float   = 80.0 # Effective (bounded by density of the mesh)
    K_AIR: float     = 0.026
    K_HEATSINK: float= 200.0

    # Density in kg/m3
    RHO_SILICON: float  = 2330.0
    RHO_IHS: float      = 8960.0
    RHO_PASTE: float    = 2500.0
    RHO_AIR: float      = 1.2
    RHO_HEATSINK: float = 2700.0 

    # Specific Heat Capacity in J/kgK
    C_SILICON: float = 700.0
    C_IHS: float     = 385.0
    C_PASTE: float   = 800.0
    C_AIR: float     = 1005.0
    C_HEATSINK: float= 900.0

@dataclass
class MaterialHeights:
    SILICON_HEIGHT: float = 35  # %
    IHS_HEIGHT: float     = 40 # %
    PASTE_HEIGHT: float   = 5  # %
    RADIATOR_HEIGHT: float= 20  # %

CPU_POWER: float = 595.0 # in Watts

def get_global_data() -> GlobalData:
    return GlobalData(
        SimulationTime=50.0,
        SimulationStepTime=1,
        Conductivity=0.0,
        Alfa=50000.0,
        Tot=25.0,
        InitialTemp=30.0, # For best result set to WaterTemp
        Density=2330.0,
        SpecificHeat=700.0,
        WaterTemp=30.0
    )