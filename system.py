# canthaxanthin simulation and tea
import biosteam as bst
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import flexsolve as flx
from matplotlib.pyplot import plot
from flexsolve import IQ_interpolation
from biosteam.units.design_tools import aeration
from math import ceil, pi
from biosteam import units, Unit, Stream, main_flowsheet
from biosteam.units import Mixer, StorageTank, SolidsSeparator, HXutility
import thermosteam as tmo
from thermosteam import Chemical, Chemicals
from thermosteam.functional import rho_to_V
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

system = bst.main_flowsheet.create_system('canthaxanthin Fermentation')

system.converge_method = 'wegstein'
system.maxiter = 50
system.molar_tolerance = 1e-3
system.diagram(kind = 'thorough', format = 'png')
system.simulate()
equipment_data = []

for unit in system.units:
    equipment_data.append({
        'Unit': unit.ID,
        'Type': unit.__class__.__name__,
        'Purchase cost ($)': unit.purchase_cost,
        'Installed cost ($)': unit.installed_cost,
        'Utility cost ($/h)': unit.utility_cost,
        'Heat duty (kW)': getattr(unit, 'heat_duty', np.nan),
        'Power (kW)': getattr(unit, 'power_utility', np.nan),
    })

equipment_df = pd.DataFrame(equipment_data)

print(equipment_df.to_string(index=False))
# techno-economic assessment and expenditure estimation

class MyTEA(bst.TEA):
    def _FOC(self, FCI):
        return 0.15 * FCI 
tea = MyTEA(
    system=system,
    IRR=0.15,
    duration=(2025, 2045),
    depreciation='MACRS7',
    income_tax=0.30,
    operating_days=330,
    lang_factor=3,
    construction_schedule=(0.4, 0.6),
    startup_months=6,
    startup_FOCfrac=1.0,
    startup_VOCfrac=0.75,
    startup_salesfrac=0.5,
    WC_over_FCI=0.05,
    finance_interest=0.08,
    finance_years=10,
    finance_fraction=0.4,
)

msp = tea.solve_price(pigment)
print(msp)
