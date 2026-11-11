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

bst.settings.CEPCI = 850
bst.settings.electricity_price = 0.075
steam_utility = bst.settings.get_agent('low_pressure_steam')
bst.settings.heating_agents = [steam_utility]
steam_utility.heat_transfer_efficiency = 0.9
steam_utility.T = 529.2
steam_utility.P = 44e5
steam_utility.regeneration_price = 1.5
bst.main_flowsheet.clear()

price_biomass = {'Rice straw':0.22,
        'Wheat straw':0.22,
        'Sugarcane bagasse':0.22,
        'Sugarcane trash':0.22,
        'Corn cobb':0.22,
        'Waste':0.22,
        }

price_chemical = {
        'Water':0.1,
        'H2SO4':2,
        'Cellulase': 20,
        'Beta_Glucanase': 15,
        'H2O2':1,
        'NaOH':1.66,
        'Steam':2,
        'Ethanol':2,
        'O2': 1,
        'N2': 1,
        'CO2': 1,
        'AceticAcid': 1,
        'SodiumAcetate': 1,
        'Glucose': 0.5,
}

chemicals = bst.settings.chemicals
rice_straw = bst.Stream(
    'rice_straw',
    Cellulose = 380,
    Hemicellulose = 250,
    Lignin = 180,
    Ash = 170,
    Solids = 20,
    units = 'kg/hr',
    price = price_biomass['Rice straw']
)
wheat_straw = bst.Stream(
    'wheat_straw',
    Cellulose = 43,
    Hemicellulose = 27,
    Lignin = 22,
    Ash = 7,
    Solids = 1,
    units = 'kg/hr',
    price = price_biomass['Wheat straw']
)
sugarcane_bagasse = bst.Stream(
    'sugarcane_bagasse',
    Cellulose = 43,
    Hemicellulose = 28,
    Lignin = 23,
    Ash = 4,
    Solids = 2,
    units = 'kg/hr',
    price = price_biomass['Sugarcane bagasse']
)
sugarcane_trash = bst.Stream(
    'sugarcane_trash',
    Cellulose = 38,
    Hemicellulose = 29,
    Lignin = 24,
    Ash = 7,
    Solids = 2,
    units = 'kg/hr',
    price = price_biomass['Sugarcane trash']
)
corn_cobb = bst.Stream(
    'corn_cobb',
    Cellulose = 40,
    Hemicellulose = 35,
    Lignin = 18,
    Ash = 2,
    Solids = 5,
    units = 'kg/hr',
    price = price_biomass['Corn cobb']
)
dilution_water = bst.Stream(
    'dilution_water',
    Water = 1000,
    units = 'kg/hr',
    T = 273,
    price = price_chemical['Water']
)
acid = bst.Stream(
    'acid',
    H2SO4 = 15,
    Water = 985,
    units = 'kg/hr',
    price = (price_chemical['H2SO4'] * 0.015) + (price_chemical['Water'] * 0.98)
)
dilution_water_recycle = bst.Stream()
alkali = bst.Stream(
    'alkali',
    NaOH = 15,
    Water = 985,
    units = 'kg/hr',
    price = (price_chemical['NaOH'] * 0.015) + (price_chemical['Water'] * 0.98)
)
bleaching_agent = bst.Stream(
    'bleaching_agent',
    H2O2 = 200,
    Water = 800,
    units = 'kg/hr',
    price = (price_chemical['H2O2'] * 0.2) + (price_chemical['Water'] * 0.8)
)

def calculate_pH(slurry, acid):
    
    # Total liquid water (kmol)
    water = slurry.imol['Water']
    if water <= 1e-9:
        return 7.0  # neutral fallback

    # Strong acid assumption
    H = acid.imol['H2SO4'] * 2  # H2SO4 -> 2H+
    if H <= 1e-12:
        return 7.0

    # pH = -log10([H+])
    return -np.log10(H / water)
class Pretreatment:
    def __init__(self, feed, ):
        self.metal_separator = units.MagneticSeparator(
            'metal_separator',
            ins = feed,
        )
        self.shred_ricestraw = units.Shredder(
            'shred_ricestraw',
            ins = self.metal_separator.outs[0],
        )
        self.pretreated_biomass = self.shred_ricestraw.outs[0]

ricestraw_biomass = Pretreatment(rice_straw, )

class HemicelluloseSection:
    def __init__(self, feed, acid, dilution_water):
        self.crushed_feed = units.CrushingMill(
            'crushed_feed',
            ins = feed,
            split=dict(
                Cellulose=1,
                Hemicellulose=1,
                Lignin=1,
                Ash=1,
                Solids=1,
                Water=1,
                ),
        )
        self.screened_slurry = units.MixTank(
            'screened_slurry',
            ins = [self.crushed_feed.outs[0], dilution_water, dilution_water_recycle],
            outs = dilution_water_recycle,
        )
        self.heated_slurry = units.HXutility(
            'heated_slurry',
            ins = self.screened_slurry.outs[0],
            T = 393.15, heat_transfer_efficiency = 0.9  # or HX_efficiency depending on API
        )
        self.acid_slurry = units.MixTank(
            'acid_slurry',
            ins = [self.heated_slurry.outs[0], acid],
        )
        self.pump_acid_slurry = units.Pump(
            'pump_acid_slurry',
            ins = self.acid_slurry.outs[0],
            P = 3e5,            # Pa (typical low-pressure slurry pump)
        )
        self.separate_solids = units.Clarifier(
            'separate_solids',
            ins = self.pump_acid_slurry.outs[0],
            split = dict(
                Ash = 0.02,
                Hemicellulose = 0.95,
                Cellulose= 0.02,
                Lignin = 0.02,
                Solids = 0.02,
                Water = 0.85,
            )
        )
        
        self.pump_hemicellulose = units.Pump(
            'pump_hemicellulose',
            ins = self.separate_solids.outs[0],
            P = 3e5, 
        )
        self.store_hemicellulose = units.StorageTank(
            'store_hemicellulose',
            ins = self.pump_hemicellulose.outs[0],
            vessel_material = 'Carbon steel',
            tau = 24
        )
        self.liquor = self.store_hemicellulose.outs[0]
        self.solids = self.separate_solids.outs[1]
        
        @self.crushed_feed.add_specification(run=True)
        def ensure_water():
            s = self.crushed_feed.ins[0]
            if s.imass['Water'] < 1e-3:
                s.imass['Water'] = 1e-3
        
hemi = HemicelluloseSection(ricestraw_biomass.pretreated_biomass, acid, dilution_water)

class LigninSection:
    def __init__(self, feed, alkali, acid, dilution_water):
        self.lignocellulose_slurry = units.MixTank(
            'lignocellulose_slurry',
            ins = [feed,alkali],
                   )
        self.pump_lignocellulose_slurry = units.Pump(
            'pump_lignocellulose_slurry',
            ins = self.lignocellulose_slurry.outs[0],
            P = 3e5, 
        )
        self.heated_lignocellulose_slurry = units.HXutility(
            'heated_lignocellulose_slurry',
            ins = self.pump_lignocellulose_slurry.outs[0],
            T = 353.15, heat_transfer_efficiency = 0.9  # or HX_efficiency depending on API
        )
        self.separate_cellulose = units.SolidsCentrifuge(
            'separate_cellulose',
            ins=self.heated_lignocellulose_slurry.outs[0],
            outs=('cellulose_pulp','lignin_liquor'),
            split={
                'Cellulose':0.98,
                'Water':0.98,
                'Lignin':0.01,
                'Ash':0.01,
                'Solids':0.01,
                'Hemicellulose':0.01
            }
        )
        self.pump_lignin_slurry = units.Pump(
            'pump_lignin_slurry',
            ins = self.separate_cellulose.outs[1],
            P = 3e5, 
        )
        self.precipitate_lignin = units.MixTank(
            'precipitate_lignin',
            ins = [self.pump_lignin_slurry.outs[0], acid],
                    )
        self.remove_water = units.SolidsCentrifuge(
            'remove_water',
            ins = self.precipitate_lignin.outs[0],
            outs = ['lignin', 'wash_water'],
            split = {
                'Lignin': 0.99,
                'Water': 0.01,
            }
        )
        self.lignin_pellet = units.SolidsCentrifuge(
            'lignin_pellet',
            ins=self.remove_water.outs[0],
            outs=('waste_water','lignin_cake'),
            split={
                'Lignin':0.01,
                },

            solids=('Lignin','Ash','Solids')
        )

        self.transport_lignin = units.ConveyingBelt(
            'transport_lignin',
            ins = self.lignin_pellet.outs[1],
        )
        self.store_lignin = units.StorageTank(
            'store_lignin',
            ins = self.transport_lignin.outs[0],
            tau = 24,
            vessel_material = 'Carbon steel',
            
        )
        self.precipitated_lignin = self.store_lignin.outs[0]
        self.pulp_cellulose = self.separate_cellulose.outs[0]
        
lignin = LigninSection(hemi.solids, alkali, acid, dilution_water)

class CelluloseSection:
    def __init__(self, feed, dilution_water):

        self.washing_pulp = units.MixTank(
            'washing_pulp',
            ins=[feed, dilution_water],
        )

        self.washed_pulp = units.SolidsCentrifuge(
            'washed_pulp',
            ins=self.washing_pulp.outs[0],
            outs=('wash_water', 'cellulose_pulp'),
            split={
                'Water': 0.99,
                'Cellulose': 0.01,
            },
        )

        self.transport_cellulose_pulp = units.ConveyingBelt(
            'transport_cellulose_pulp',
            ins=self.washed_pulp.outs[1],
        )

        self.store_cellulose = units.StorageTank(
            'store_cellulose',
            ins=self.transport_cellulose_pulp.outs[0],
            tau=24,
            vessel_material='Carbon steel',
        )

        self.cellulose = self.store_cellulose.outs[0]

cellulose = CelluloseSection(lignin.pulp_cellulose, dilution_water)
cellulose_product = cellulose.cellulose
system = bst.main_flowsheet.create_system('rice_straw_system')

system.converge_method = 'wegstein'
system.maxiter = 50
system.molar_tolerance = 1e-3

system.simulate()
system.results()
system.show()
g = bst.constants.g
titer = 0.5
theta_O2 = 0.5
kW_per_m3 = 0.2955
feed_flow = 100
T_operation = 310
sugar_conc = 0.2
yield_pigment = 0.5

# define feed/streams

# product streams
broth = bst.Stream()
vent = bst.Stream(phase = 'g')
product = bst.MultiStream.from_streams([vent, broth])
air = bst.Stream(
    'air',
    O2=1,
    N2=79/21,
    units='kmol/hr',
    T=T_operation,
    P=101325,
    phase='g',
)
# reactions associated with fermentation and production

cellulose_glucose_conversion = bst.Reaction(
    'Cellulose -> Glucose',
    reactant = 'Cellulose',
    X = 0.5,
    correct_atomic_balance = True,
)
glucose_conversion = units.SinglePhaseReactor(
    'glucose_conversion',
    ins = cellulose_product,
    outs = ('glucose_slurry'),
    tau = 1,
    T = 310, P = 101325,
    reactions = cellulose_glucose_conversion,
    batch = True,
)

glucose_conversion.simulate()
feed = glucose_conversion.outs[0]
Yp = 0.00648
Yb = 0.3333
production_canthaxanthin = bst.Reaction(
    'Glucose -> Canthaxanthin',
    reactant= 'Glucose',
    X = 1,
    #correct_atomic_balance= True,
)
production_canthaxanthin.product_yield(
    'Canthaxanthin', basis = 'wt',
    product_yield= Yp,
)

growth_biomass = bst.Reaction(
    'Glucose -> Biomass',
    reactant= 'Glucose',
    X = 1,
    #correct_atomic_balance= True,
)

growth_biomass.product_yield(
    'Biomass', basis = 'wt',
    product_yield= Yp/Yb
)
respiration_biomass = bst.Reaction(
    'Glucose + O2 -> H2O + CO2',
    reactant = 'Glucose',
    X = 1. - growth_biomass.X,
    #correct_atomic_balance = True
)
fermentation_canthaxanthin = bst.ReactionSystem(
        bst.ParallelReaction([production_canthaxanthin, growth_biomass]),
        respiration_biomass
    )

# bioreactor sizing and design heuristics using bioreactor object so that biosteam calculates the requirements

vessel_radius = 1
V_max = 5
conversion_efficiency = 0.9
working_volume = 3
tau_reaction = 72
tau_cleanup = 12
tau_loading = 12
nue = feed.F_vol
V_total = nue*(tau_loading + tau_cleanup + tau_reaction)/working_volume
N_reactor = ceil(V_total/V_max)
V_reactor = V_total/N_reactor
diameter_vessel = (4*V_reactor/pi/vessel_radius)**(1/3)
height_vessel = diameter_vessel*vessel_radius

# titer specification based on dilution water and air flow rate

T = 32 + 273.15

air = bst.Stream(
        O2=21, N2=79, units='kmol/hr',
        T=T_operation,
        phase='g',
    )
vent = bst.Stream(phase='g')
broth = bst.Stream()
product = bst.MultiStream.from_streams([vent, broth])
product.T = T_operation

def objective(F_water):

        feed.imol['Water'] = F_water

        # Start with fresh feed
        broth.copy_like(feed)

        # Determine the O2 requirement from the reaction stoichiometry
        test_broth = broth.copy()
        fermentation_canthaxanthin.force_reaction(test_broth)

        required_O2 = max(0, -test_broth.imol['O2'])

        air.imol['O2', 'N2'] = [
            required_O2,
            required_O2 * 79 / 21
        ]

        # Fresh broth + air
        broth.mix_from(
            [feed, air],
            energy_balance=False
        )

        # React ONLY ONCE
        fermentation_canthaxanthin(broth)

        vent.empty()
        aeration.vent_broth(vent, broth)

        actual_titer = (
            broth.imass['Canthaxanthin'] / broth.F_vol
        )

        return actual_titer - titer

flx.aitken_secant(objective, feed.imol['Water']) # Solve objective function

    # dissolved oxygen calculations
OUR = -broth.imol['O2']
length_of_filled_vessel = height_vessel*working_volume
air.P = g*broth.rho*length_of_filled_vessel+101325
vent.P = 101325
def objective(air_flow):
        air.set_flow([air_flow*0.21, air_flow*0.79], 'mol/s', ['O2', 'N2'])
        vent.empty()
        broth.mix_from([feed, air], energy_balance = False)
        fermentation_canthaxanthin(broth)

        aeration.vent_broth(vent, broth)
        V_reactive = V_reactor*working_volume
        fraction_operating = tau_reaction/(tau_loading + tau_cleanup + tau_reaction)
        agitation_power = 1e3*kW_per_m3*V_reactive
        F = air.get_total_flow('m3/s')/N_reactor/fraction_operating
        r = 0.5*diameter_vessel
        area = pi*r*r
        superficial_gas_flow = U=F/area
        kLa = aeration.kLa_stirred_Riet(
            agitation_power,V_reactive,U
        )
        P_O2_air = air.get_property('P', 'bar')*air.imol['O2']/air.F_mol
        P_O2_vent = vent.get_property('P', 'bar')*vent.imol['O2']/vent.F_mol
        C_O2_sat_air = aeration.C_O2_L(T_operation, P_O2_air)
        C_O2_sat_vent = aeration.C_O2_L(T_operation, P_O2_vent)
        LMDF = aeration.log_mean_driving_force(C_O2_sat_vent, C_O2_sat_air, theta_O2*C_O2_sat_vent, theta_O2*C_O2_sat_air)
        OTR = kLa*LMDF*broth.rho*V_total*fraction_operating
        return OUR-OTR

minimum_air_flow_rate = OUR/0.21
maximum_air_flow_rate = 10*minimum_air_flow_rate

    # bioreactor object specification

bioreactor = bst.AeratedBioreactor(
        ins = [feed, air],
        outs = ('vent', 'broth'),
        reactions = fermentation_canthaxanthin,
        T = T_operation,
        P = 101325,
        V_wf = conversion_efficiency,
        V_max = V_max,
        tau = tau_reaction,
        tau_0 = tau_cleanup,
        loading_time = tau_loading,
        length_to_diameter = vessel_radius,
        method = 'Riet',
        design = 'Stirred tank',
        heat_exchanger_configuration = 'jacketed',
        kW_per_m3 = kW_per_m3,
        optimize_power = False,
    )


biomass_pellet = bst.SolidsCentrifuge(
    'biomass_pellet',
    ins = bioreactor.outs[1],
    outs = ('effluent', 'biomass_with_product'),
    split = {
                'Water': 0.99,
            },
)
ethanol_extraction = bst.MixTank(
    'ethanol_extraction',
    ins = (biomass_pellet.outs[1],'Ethanol'),
    outs = 'ethanol_slurry',
    
)
canthaxanthin_ethanol_slurry = bst.SolidsCentrifuge(
    'canthaxanthin_ethanol_slurry',
    ins = ethanol_extraction.outs[0],
    outs = ('effluent', 'biomass_with_product'),
    split = {
                'Ethanol': 0.99,
                'Canthaxanthin': 0.99,
            },
)
# ---------------------------------------------------------
# Silica-gel adsorption column for canthaxanthin capture
# ---------------------------------------------------------

rho_silica_bulk = 720       # kg/m3, packed-bed bulk density
epsilon_silica = 0.40       # packed-bed void fraction
dp_silica = 0.0039          # m, ~4-6 mesh silica gel
'''
purified_product = bst.AdsorptionColumn(
    'purified_product',

    ins=canthaxanthin_ethanol_slurry.outs[0],

    outs=(
        'canthaxanthin_effluent',
        'canthaxanthin_eluent',
        'spent_silica'
    ),

    cycle_time=3.0,

    superficial_velocity=9.2,

    isotherm_model='Langmuir',

    # DEBUGGING VALUES ONLY
    isotherm_args=(1e3, 7.0),

    k=0.3,

    void_fraction=0.40,
    rho_adsorbent=720,

    C_final_scaled=0.05,

    # TEMPORARY

    adsorbate='Canthaxanthin',
    adsorbent='SilicaGel',

    particle_diameter=0.0039,

    N_columns=3,

    vessel_material='Stainless steel 316',
    vessel_type='Vertical',
)
'''
pigment = canthaxanthin_ethanol_slurry.outs[0]
glucose = bst.settings.chemicals['Glucose']

