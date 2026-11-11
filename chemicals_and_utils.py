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

# pretreatment process and biomass preparation

bst.nbtutorial()

_cal2joule = 4.184
__all__ = ('create_chemicals', 'get_chemical_properties')
def create_chemicals(set_thermo = True):
    chems = Chemicals([])
    def add_chemical(ID, ref = None, **kwargs):
        chemical = Chemical(ID, **kwargs) if ref is None else ref.copy(ID, **kwargs)
        chems.append(chemical)
        return chemical
    def create_chemicals(set_thermo=True):

        chems = Chemicals([])

    def add_chemical(ID, ref=None, **kwargs):
        if ref is None:
            chemical = Chemical(ID, **kwargs)
        else:
            chemical = ref.copy(ID, **kwargs)

        chems.append(chemical)
        return chemical

    # Standard chemicals
    chems = tmo.Chemicals([])

    # Standard chemicals
    for chem in (
        'Ethanol',
        'NaOH',
        'NaCl',
        'H2SO4',
        'AceticAcid',
        'SodiumAcetate',
        'H2O',
        'H2O2',
        'O2',
        'N2',
        'CO2',
    ):
        add_chemical(chem)

    # Glucose is dissolved in the fermentation broth.
    # Lock to liquid phase to prevent an invalid gas-phase
    # thermodynamic calculation.
    glucose = tmo.Chemical('Glucose', phase='l')
    chems.append(glucose)
        
    # Lignocellulosic components

    Lignin = add_chemical(
        'Lignin',
        search_db=False,
        phase='s',
        formula='C8H8O3',
        Hf=-108248 * _cal2joule,
        default=True,
    )

    Cellulose = add_chemical(
        'Cellulose',
        search_db=False,
        phase='s',
        formula='C6H10O5',
        Hf=-233200.06 * _cal2joule,
        default=True,
    )

    Hemicellulose = add_chemical(
        'Hemicellulose',
        search_db=False,
        phase='s',
        formula='C5H8O4',
        Hf=-182100 * _cal2joule,
        default=True,
    )

    # Inert material
    Ash = add_chemical(
        'Ash',
        search_db=True,
        phase='s',
        default=True,
    )

    Solids = add_chemical(
        'Solids',
        search_db=False,
        phase='s',
        MW=1.0,
        rho=1540,
        Cp=1364,
        default=True,
    )
    SilicaGel = add_chemical(
        'SilicaGel',
        search_db=False,
        phase='s',
        formula='SiO2',
        MW=60.084,
        rho=2200,
        default=True,
    )

    # Nutrient / enzyme lumped components

    YeastExtract = add_chemical(
        'YeastExtract',
        search_db=False,
        phase='s',
        formula='CH1.57O0.31N0.29S0.007',
        Hf=-17618 * _cal2joule,
        default=True,
    )

    BT_CGM = add_chemical(
        'BT_CGM',
        search_db=False,
        phase='s',
        # DO NOT assign a molecular formula until its identity/composition
        # is established.
        MW=1.0,
        default=True,
    )

    # Product

    Canthaxanthin = add_chemical(
        'Canthaxanthin',
        search_db=False,
        phase='s',
        formula='C40H52O2',
        default=True,
    )

    # Enzymes: lumped protein representation

    Beta_Glucanase = add_chemical(
        'Beta_Glucanase',
        search_db=False,
        phase='l',
        formula='CH1.59O0.42N0.24S0.01',
        Hf=-17618 * _cal2joule,
        default=True,
    )

    Cellulase = add_chemical(
        'Cellulase',
        search_db=False,
        phase='s',
        formula='CH1.59O0.42N0.24S0.01',
        Hf=-17618 * _cal2joule,
        default=True,
    )

    # Microbial biomass

    Biomass = add_chemical(
        'Biomass',
        search_db=False,
        phase='s',
        formula='CH1.64O0.39N0.23S0.0035',
        Hf=-23200.01 * _cal2joule,
        default=True,
    )
    solid_density = 1540

    for ID in ('Lignin', 'Cellulose', 'Ash', 'Solids'):
        chem = chems[ID]
        chem.V.add_model(
            rho_to_V(solid_density, chem.MW),
            top_priority=True,
        )

    for chem in chems:
        try:
            chem.default()
        except:
            pass

    chems.compile()
    chems.set_alias('Water', 'H2O')
    if set_thermo:
        tmo.settings.set_thermo(chems)

    return chems

def get_chemical_properties(chemicals, T, P, output = False):
    formulas = [chemical.formula for chemical in chemicals]
    MW = [chemical.MW for chemical in chemicals]
    Hfs = [chemical.Hf for chemical in chemicals]
    HHVs = [chemical.HHV for chemical in chemicals]
    LHVs = [chemical.LHV for chemical in chemicals]
    phases = []
    Tbs = []
    Psats = []
    Vs = []
    Cns = []
    mus = []
    kappas = []

    for chemical in chemicals:
            if chemical.locked_state:
                phases.append(chemical.phase_ref)
                Tbs.append('NA')
                try: Psats.append(chemical.Psat(T=T, P=P))
                except: Psats.append('')
                try: Vs.append(chemical.V(T=T, P=P))
                except: Vs.append('')
                try: Cns.append(chemical.Cn(T=T))
                except: Cns.append('')
                try: mus.append(chemical.mu(T=T, P=P))
                except: mus.append('')
                try: kappas.append(chemical.kappa(T=T, P=P))
                except: kappas.append('')
            else:
                ref_phase = chemical.get_phase(T=T, P=P)
                phases.append(f'variable, ref={ref_phase}')
                Tbs.append(chemical.Tb)
                try: Psats.append(chemical.Psat(T=T, P=P))
                except: Psats.append('')
                try: Vs.append(chemical.V(ref_phase, T=T, P=P))
                except: Vs.append('')
                try: Cns.append(chemical.Cn(ref_phase, T=T))
                except: Cns.append('')
                try: mus.append(chemical.mu(ref_phase, T=T, P=P))
                except: mus.append('')
                try: kappas.append(chemical.kappa(ref_phase, T=T, P=P))
                except: kappas.append('')

    properties = pd.DataFrame(
            {'ID': chemicals.IDs,
            'formula': formulas,
            'MW': MW,
            'HHV': HHVs,
            'LHV': LHVs,
            'Hf': Hfs,
            'phase': phases,
            'boiling point': Tbs,
            'Psat': Psats,
            'V': Vs,
            'Cn': Cns,
            'mu': mus,
            'kappa': kappas}
            )

    if output:
        properties.to_excel('chemical_properties.xlsx', sheet_name='properties')

chems = create_chemicals()
