# Load pldspectrapy from elsewhere on computer
from packfind import find_package
find_package('pldspectrapy')
import pldspectrapy as pld

# Simulate CH4 and CO2
sim = pld.SpectraSim()

sim.def_environment(300, 1, 10e-2, 1628, 1695) # Temp_Kelvin, Press_atm, pathlength_meter, nm_start, nm_stop
sim.def_molecules(["CH4", "CO2"], [0.002, 0.95])
sim.print_Hitran()

sim.simulate()

# Now plot the results
objPlot = pld.Plotting()
objPlot.plot(sim.x_nm, sim.y_absorbance, strLabels=["CH4", "CO2"])

print("DONE")