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

msp = tea.solve_price(pigment)
print(msp)

annual_mass={}

# Calculate the minimum selling price
msp = tea.solve_price(pigment)

print(f"Minimum selling price: ${msp:.4f}/kg")

# Assign the calculated price to the product stream
pigment.price = msp

# Check the stream
print("\n===== PRODUCT STREAM CHECK =====")
print(f"Product stream ID: {pigment.ID}")
print(f"Product mass flow: {pigment.F_mass:.6f} kg/hr")
print(f"Product price: ${pigment.price:.4f}/kg")
print(f"Annual product mass: "
      f"{pigment.F_mass * 24 * tea.operating_days:,.2f} kg/year")

# Check whether the stream is registered as a system product
print("\n===== SYSTEM PRODUCT CHECK =====")
for stream in system.products:
    print(
        f"ID: {stream.ID}, "
        f"Mass flow: {stream.F_mass:.6f} kg/hr, "
        f"Price: ${stream.price:.4f}/kg, "
        f"Is pigment stream: {stream is pigment}"
    )

# Recalculate and inspect TEA sales
print("\n===== TEA SUMMARY =====")
print(f"FCI: ${tea.FCI:,.2f}")
print(f"TCI: ${tea.TCI:,.2f}")
print(f"AOC: ${tea.AOC:,.2f}/year")
print(f"Utility Cost: ${tea.utility_cost:,.2f}/year")
print(f"Material Cost: ${tea.material_cost:,.2f}/year")
print(f"Sales: ${tea.sales:,.2f}/year")

cf = tea.get_cashflow_table()
annual_mass['Canthaxanthin'] = (pigment.F_mass * 24 * tea.operating_days)

print(system.products)

print("\n===== TEA SUMMARY =====")

print(f"FCI                 : {tea.FCI:,.2f}")
print(f"TCI                 : {tea.TCI:,.2f}")
print(f"AOC (Total OPEX)    : {tea.AOC:,.2f} $/yr")
print(f"Utility Cost        : {tea.utility_cost:,.2f} $/yr")
print(f"Material Cost       : {tea.material_cost:,.2f} $/yr")
print(f"Sales               : {tea.sales:,.2f} $/yr")

cf = tea.get_cashflow_table()
print(cf)
npv = cf['Net present value (NPV) [MM$]'].sum()

print(f"NPV = {npv:.3f} MM$")

cf = tea.get_cashflow_table()

cum_npv = cf['Cumulative NPV [MM$]']

positive = cum_npv[cum_npv > 0]

if len(positive):
    payback = positive.index[0]
    print(f"Payback year: {payback}")
else:
    print("No payback achieved")
pigment_IRR = tea.solve_IRR()
print(pigment_IRR)
from chaospy import distributions as shape
from warnings import filterwarnings;

model = bst.Model(system)
@model.indicator(units= 'USD/kg')
def MSP():
    return tea.solve_price(pigment)
@model.indicator(units ='10^6*USD')
def TCI():
    return tea.TCI/1e6

@model.indicator(units = '%')
def IRR():
    return tea.solve_IRR()*100
@model.indicator(units = '$')
def NPV():
    return tea.NPV
@model.indicator(units = '$/yr')
def AOC():
    return tea.AOC
@model.indicator(units = '$')
def FCI():
    return tea.FCI

@model.parameter(
    element = 'Glucose',
    units = '$/hr',
    bounds = (0.4,0.8),
    baseline = 0.5,
    distribution = 'triangular',
)
def commercial_glucose_pricing(price):
    price_chemical['Glucose'] = price

@model.parameter(
    element='Canthaxanthin yield',
    units='kg/kg',
    bounds=(0.005, 0.008),
    baseline=Yp,
    distribution='triangular',
    coupled=True,
)
def set_canthaxanthin_yield(y):
    production_canthaxanthin.product_yield(
        'Canthaxanthin',
        basis='wt',
        product_yield=y,
    )

@model.parameter(
    element='Biomass yield parameter',
    units='dimensionless',
    bounds=(0.25, 0.42),
    baseline=Yb,
    distribution='triangular',
    coupled=True,
)
def set_biomass_yield(yb):

    growth_biomass.product_yield(
        'Biomass',
        basis='wt',
        product_yield=Yp / yb
    )


@model.parameter(
    element='Ethanol',
    units='$/kg',
    bounds=(1.5, 2.5),
    baseline=price_chemical['Ethanol'],
    distribution='triangular',
)
def set_ethanol_price(price):
    price_chemical['Ethanol'] = price



@model.parameter(
    element='Electricity',
    units='$/kWh',
    bounds=(0.06, 0.10),
    baseline=bst.settings.electricity_price,
    distribution='triangular',
)
def set_electricity_price(price):
    bst.settings.electricity_price = price

@model.parameter(
    element='LP steam',
    units='$/kg',
    bounds=(1.20, 1.80),
    baseline=steam_utility.regeneration_price,
    distribution='triangular',
)
def set_steam_price(price):
    steam_utility.regeneration_price = price

@model.parameter(
    element='Water',
    units='$/kg',
    bounds=(0.07, 0.13),
    baseline=price_chemical['Water'],
    distribution='triangular',
)
def set_water_price(price):

    price_chemical['Water'] = price

    dilution_water.price = price

    commercial_glucose.price = price

    
np.random.seed(1234)
samples = model.sample(N =100, rule = 'L')
model.load_samples(samples, sort = True)
model.evaluate()
results = model.table.copy()
results

#results.to_excel('biofilm_monte_carlo_2_kg_flow_rate.xlsx')

# montecarlo plots and uncertainty analysis

format_units = tmo.units_of_measure.format_units
ylabel = f"MSP [{format_units('USD/kg')}]"
xlabel = f"TCI [{format_units('10^6 USD')}]"
def plot_uncertainty(table): # This function will be useful later
    fig, ax, axes = bst.plots.plot_kde(
        y=table[MSP.index],
        x=table[TCI.index],
        ylabel=ylabel,
        xlabel=xlabel,
        aspect_ratio=1.1,
    )
plot_uncertainty(model.table)

msp = model.table['-', 'MSP [USD/kg]']

print("Mean MSP:", msp.mean())
print("Median MSP:", msp.median())
print("Minimum MSP:", msp.min())
print("Maximum MSP:", msp.max())
print("Standard deviation:", msp.std())

plt.figure(figsize=(8, 5))

plt.hist(
    msp,
    bins=30,
    density=True,
    edgecolor='black'
)

plt.axvline(
    msp.mean(),
    linestyle='--',
    label=f'Mean = {msp.mean():.2f}'
)

plt.xlabel('MSP [USD/kg]')
plt.ylabel('Probability density')
plt.title('Monte Carlo Distribution of MSP')
plt.legend()
plt.show()

# Spearman correlation plots 

print('\n MSP sensitivity\n')
df_rho, df_p = model.spearman_r()
print('\n Rho values\n', df_rho['-', 'MSP [USD/kg]'])
print('\n P values\n', df_p['-', 'MSP [USD/kg]'])
bst.plots.plot_spearman_1d(df_rho['-', 'MSP [USD/kg]'], index = [i.describe() for i in model.parameters], name = 'MSP [USD/kg]')
print('\n AOC sensitivity\n')
df_rho, df_p = model.spearman_r()
print('\n Rho values\n', df_rho['-', 'AOC [$/yr]'])
print('\n P values\n', df_p['-', 'AOC [$/yr]'])
bst.plots.plot_spearman_1d(df_rho['-', 'AOC [$/yr]'], index = [i.describe() for i in model.parameters], name = 'AOC [$/yr]')
print('\n NPV sensitivity\n')
df_rho, df_p = model.spearman_r()
print('\n Rho values\n', df_rho['-', 'NPV [$]'])
print('\n P values\n', df_p['-', 'NPV [$]'])
bst.plots.plot_spearman_1d(df_rho['-', 'NPV [$]'], index = [i.describe() for i in model.parameters], name = 'NPV [$]')
print('\n TCI sensitivity\n')
df_rho, df_p = model.spearman_r()
print('\n Rho values\n', df_rho['-', 'TCI [10^6*USD]'])
print('\n P values\n', df_p['-', 'TCI [10^6*USD]'])
bst.plots.plot_spearman_1d(df_rho['-', 'TCI [10^6*USD]'], index = [i.describe() for i in model.parameters], name = 'TCI [10^6*USD]')

# Single point sensitivity plots
baseline, lower, upper = model.single_point_sensitivity()
def plot_sps(metric, name, scale=1):

    metric_index = metric.index

    index = [
        i.describe(distribution=False)
        for i in model.parameters
    ]

    bst.plots.plot_single_point_sensitivity(
        scale * baseline[metric_index],
        scale * lower[metric_index],
        scale * upper[metric_index],
        name=name,
        index=index
    )

plot_sps(
    MSP,
    'MSP [USD/kg]',
    scale=1
)
plot_sps(
    TCI,
    'TCI [10^6*USD]',
    scale = 1
)
plot_sps(
    AOC,
    'AOC [$/yr]',
)
plot_sps(
    NPV,
    'NPV [$]',
    scale = 1
)
