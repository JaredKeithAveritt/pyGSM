from pyGSM.level_of_theories.gaussian import Gaussian
from pygsm.potential_energy_surfaces import PES
from pygsm.optimizers import *
from pygsm.wrappers import Molecule
from pygsm.utilities import *
from coordinate_systems import Topology,PrimitiveInternalCoordinates,DelocalizedInternalCoordinates
import numpy as np


def main(): 
    geom = manage_xyz.read_xyz("molecule.xyz")
    xyz = manage_xyz.xyz_to_np(geom)

    nifty.printcool(" Building the LOT")
    lot = QChem.from_options(
            lot_inp_file="qstart",
            states=[(1,0)],
            geom=geom,
            )

    nifty.printcool(" Building the PES")
    pes = PES.from_options(
            lot=lot,
            ad_idx=0,
            multiplicity=1,
            )

    nifty.printcool("Building the topology")
    atom_symbols  = manage_xyz.get_atoms(geom)
    ELEMENT_TABLE = elements.ElementData()
    atoms = [ELEMENT_TABLE.from_symbol(atom) for atom in atom_symbols]
    top = Topology.build_topology(
            xyz,
            atoms,
            )

    nifty.printcool("Building Primitive Internal Coordinates")
    p1 = PrimitiveInternalCoordinates.from_options(
            xyz=xyz,
            atoms=atoms,
            addtr=False,  # Add TRIC
            topology=top,
            )

    nifty.printcool("Building Delocalized Internal Coordinates")
    coord_obj1 = DelocalizedInternalCoordinates.from_options(
            xyz=xyz,
            atoms=atoms,
            addtr = False,  # Add TRIC
            )

    nifty.printcool("Building Molecule")
    reactant = Molecule.from_options(
            geom=geom,
            PES=pes,
            coord_obj = coord_obj1,
            Form_Hessian=True,
            )

    print(" Done creating molecule")
    optimizer = eigenvector_follow.from_options(Linesearch='backtrack',OPTTHRESH=0.0005,DMAX=0.5,abs_max_step=0.5,conv_Ediff=0.5)

    print("initial energy is {:5.4f}".format(reactant.energy))
    geoms,energies = optimizer.optimize(
            molecule=reactant,
            refE=reactant.energy,
            opt_steps=500,
            verbose=True,
            )

    print("Final energy is {:5.4f}".format(reactant.energy))
    manage_xyz.write_xyz('minimized.xyz',geoms[-1],energies[-1],scale=1.)

if __name__=='__main__':
    main()

