from pldspectrapy.sim import SpectraSim
from pldspectrapy.plot import Plotting
import unittest

class TestPldSpectraPy(unittest.TestCase):
    def testPldSim(self):
        sim = SpectraSim()

        sim.defEnviro(300, 1, 1e5, 1620, 1690)
        #sim.defMolecules(["CH4", "CO2"], [0.75, 0.25])
        sim.defMolecules(["CH4"], [2e-6])
        #sim.printHitranMolecules()

        sim.simulate()
        
        plt = Plotting()
        plt.plot(sim.vecWvl,sim.vecAbsorb)


if __name__ == '__main__':
    unittest.main()
    print("PldSpectraPy passed!")