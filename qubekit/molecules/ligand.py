#!/usr/bin/env python3

"""
TODO ligand.py Refactor:
    DO:
        Move module-specific methods such as openmm_coordinates(); read_tdrive(); read_geometric_traj
            to their relevant files/classes
        Fix naming; consistency wrt get/find; clarity on all of the dihedral variables
            (what is dih_start, how is it different to di_starts etc)
        Perform checks after reading input (check_names_are_unique(), validate_info(), etc)
    CONSIDER:
        Remove / replace DefaultsMixin with inheritance, dict or some other solution
        Remove any repeated or unnecessary variables
            Should state be handled in ligand or run?
        Change the structure and type of some variables for clarity
            Do we access via index too often; should we use e.g. SimpleNamespaces/NamedTupleS?
        Be more strict about public/private class/method/function naming?
"""

import decimal
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from xml.dom.minidom import parseString

import networkx as nx
import numpy as np
import qcelemental as qcel
from chemper.graphs.cluster_graph import ClusterGraph
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.toolkit.utils.exceptions import ParameterLookupError
from openmm import unit
from openmm.app import Aromatic, Double, Single, Topology, Triple
from openmm.app.element import Element
from pydantic import Field, validator
from qcelemental.models.types import Array
from rdkit import Chem
from typing_extensions import Literal

import qubekit
from qubekit.forcefield import (
    BaseForceGroup,
    HarmonicAngleForce,
    HarmonicBondForce,
    LennardJones126Force,
    PeriodicImproperTorsionForce,
    PeriodicTorsionForce,
    RBImproperTorsionForce,
    RBProperTorsionForce,
    UreyBradleyHarmonicForce,
    VirtualSiteGroup,
)
from qubekit.molecules.components import Atom, Bond, TorsionDriveData
from qubekit.molecules.utils import RDKit, ReadInput
from qubekit.utils import constants
from qubekit.utils.datastructures import SchemaBase
from qubekit.utils.exceptions import (
    ConformerError,
    FileTypeError,
    MissingParameterError,
    MissingReferenceData,
    StereoChemistryError,
    TopologyMismatch,
)


class Molecule(SchemaBase):
    """Base class for ligands and proteins.

    The class is a simple representation of the molecule as a list of atom and bond objects, many attributes are then
    inferred from these core objects.
    """

    type: Literal["Molecule"] = "Molecule"

    atoms: List[Atom] = Field(
        ..., description="A list of QUBEKit atom objects which make up the molecule."
    )
    bonds: Optional[List[Bond]] = Field(
        None,
        description="The list of QUBEKit bond objects which connect the individual atoms.",
    )
    coordinates: Optional[Array[float]] = Field(
        None,
        description="A numpy arrary of the current cartesian positions of each atom in angstrom, this must be of size (n_atoms, 3)",
    )
    multiplicity: int = Field(
        1,
        description="The integer multiplicity of the molecule which is used in QM calculations.",
    )
    name: str = Field(
        "unk",
        description="An optional name string which will be used in all file IO calls by default.",
    )
    provenance: Dict[str, Any] = Field(
        dict(creator="QUBEKit", version=qubekit.__version__),
        description="Information on the version and method used to create this molecule.",
    )
    extra_sites: VirtualSiteGroup = Field(
        VirtualSiteGroup(),
        description="A force object which records any virtual sistes in the molecule",
    )
    BondForce: Union[HarmonicBondForce] = Field(
        HarmonicBondForce(),
        description="A force object which records bonded interactions between pairs of atoms",
    )
    UreyBradleyForce: UreyBradleyHarmonicForce = Field(
        UreyBradleyHarmonicForce(),
        description="A force object which records Urey-Bradley bond-angle cross terms.",
    )
    AngleForce: Union[HarmonicAngleForce] = Field(
        HarmonicAngleForce(),
        description="A force object which records angle interactions between atom triplets.",
    )
    TorsionForce: PeriodicTorsionForce = Field(
        PeriodicTorsionForce(),
        description="A force object which records torsion interactions between atom quartets, using a periodic function.",
    )
    ImproperTorsionForce: PeriodicImproperTorsionForce = Field(
        PeriodicImproperTorsionForce(),
        description="A force group which records improper torsion interactions between atom quartets using a periodic function.",
    )
    RBTorsionForce: RBProperTorsionForce = Field(
        RBProperTorsionForce(),
        description="A force group which records torsion interactions between atom quartets using a RB function.",
    )
    ImproperRBTorsionForce: RBImproperTorsionForce = Field(
        RBImproperTorsionForce(),
        description="A force object which records improper torsion interactions between atom quartets using a RB function.",
    )
    NonbondedForce: Union[LennardJones126Force] = Field(
        LennardJones126Force(),
        description="A force group which records atom nonbonded parameters.",
    )
    chargemol_coords: Optional[Array[float]] = Field(
        None,
        description="The coordinates used to calculate the chargemol quantities, "
        "this is a reorientated conformation",
    )
    hessian: Optional[Array[float]] = Field(
        None,
        description="The hessian matrix calculated for this molecule at the QM optimised geometry.",
    )
    qm_scans: Optional[List[TorsionDriveData]] = Field(
        None,
        description="The list of reference torsiondrive results which we can fit against.",
    )
    wbo: Optional[Array[float]] = Field(
        None, description="The WBO matrix calculated at the QM optimised geometry."
    )

    @validator("coordinates", "chargemol_coords")
    def _reshape_coords(cls, coordinates: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if coordinates is not None:
            return coordinates.reshape((-1, 3))
        else:
            return coordinates

    @validator("hessian", "wbo", allow_reuse=True)
    def _reshape_matrix(cls, matrix: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if matrix is not None:
            if len(matrix.shape) == 1:
                # the matrix is a flat list
                # so we need to make the matrix to be square
                length = int(np.sqrt(matrix.shape[0]))
                return matrix.reshape((length, length))
        return matrix

    def __init__(
        self,
        atoms: List[Atom],
        bonds: Optional[List[Bond]] = None,
        coordinates: Optional[np.ndarray] = None,
        multiplicity: int = 1,
        name: str = "unk",
        routine: Optional[List[str]] = None,
        provenance: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Init the molecule using the basic information.

        Note:
            This method updates the provanance info.

        Args:
            atoms:
                A list of QUBEKit atom objects in the molecule.
            bonds:
                A list of QUBEKit bond objects in the molecule.
            coordinates:
                A numpy array of the current cartesian positions of each atom, this must be of size (n_atoms, 3)
            multiplicity:
                The integer multiplicity of the molecule which is used in QM calculations.
            name:
                An optional name string which will be used in all file IO calls by default.
            routine:
                The set of strings which encode the routine information used to create the molecule.

            bool; is the current execution starting from the beginning (False) or restarting (True)?
        """
        # the way the molecule was made
        method = routine or ["__init__"]
        if provenance is None:
            new_provenance = dict(
                creator="QUBEKit", version=qubekit.__version__, routine=method
            )
        else:
            # make sure we respect the provenance when parsing a json file
            new_provenance = provenance
            new_provenance["routine"] = new_provenance["routine"]
        name = name or "unk"

        super(Molecule, self).__init__(
            atoms=atoms,
            bonds=bonds,
            multiplicity=multiplicity,
            name=name,
            coordinates=coordinates,
            provenance=new_provenance,
            **kwargs,
        )
        # make sure we have unique atom names
        self._validate_atom_names()

    def __eq__(self, other: "Molecule"):
        return self.to_smiles(isomeric=True, mapped=False) == other.to_smiles(
            isomeric=True, mapped=False
        )

    def to_topology(self) -> nx.Graph:
        """
        Build a networkx representation of the molecule.
        TODO add other attributes to the graph?
        """
        graph = nx.Graph()
        for atom in self.atoms:
            graph.add_node(atom.atom_index)

        for bond in self.bonds:
            graph.add_edge(bond.atom1_index, bond.atom2_index)
        return graph

    def to_file(self, file_name: str) -> None:
        """
        Write the molecule object to file working out the file type from the extension.
        Works with PDB, MOL, SDF, XYZ, Json any other we want?
        """
        if ".json" in file_name:
            with open(file_name, "w") as output:
                output.write(self.json(indent=2))
        else:
            return RDKit.mol_to_file(rdkit_mol=self.to_rdkit(), file_name=file_name)

    def generate_conformers(self, n_conformers: int) -> List[np.ndarray]:
        """
        Generate a list of conformers and return them this includes the input conformer

        Args:
            n_conformers: The number of conformers which should be generated.
        """
        rd_mol = self.to_rdkit()
        return RDKit.generate_conformers(rdkit_mol=rd_mol, conformer_no=n_conformers)

    def to_multiconformer_file(
        self, file_name: str, positions: List[np.ndarray]
    ) -> None:
        """
        Write the molecule to a file allowing multipule conformers.

        As the ligand object only holds one set of coordinates at once a list of coords can be passed here to allow
        multiconformer support.

        Args:
            file_name:
                The name of the file that should be created, the type is inferred by the suffix.
            positions:
                A list of Cartesian coordinates of shape (n_atoms, 3).
        """
        rd_mol = self.to_rdkit()
        rd_mol.RemoveAllConformers()
        # add the conformers
        if not isinstance(positions, list):
            positions = [
                positions,
            ]
        for conformer in positions:
            RDKit.add_conformer(rdkit_mol=rd_mol, conformer_coordinates=conformer)
        return RDKit.mol_to_multiconformer_file(rdkit_mol=rd_mol, file_name=file_name)

    def get_atom_with_name(self, name):
        """
        Search through the molecule for an atom with that name and return it when found
        :param name: The name of the atom we are looking for
        :return: The QUBE Atom object with the name
        """

        for atom in self.atoms:
            if atom.atom_name == name:
                return atom
        raise AttributeError("No atom found with that name.")

    def get_bond_between(self, atom1_index: int, atom2_index: int) -> Bond:
        """
        Try and find a bond between the two atom indices.

        The bond may not have the atoms in the expected order.

        Args:
            atom1_index:
                The index of the first atom in the atoms list.
            atom2_index:
                The index of the second atom in the atoms list.

        Returns:
            The bond object between the two target atoms.

        Raises:
            TopologyMismatch:
                When no bond can be found between the atoms.
        """
        target = [atom1_index, atom2_index]
        for bond in self.bonds:
            if bond.atom1_index in target and bond.atom2_index in target:
                return bond
        raise TopologyMismatch(
            f"There is no bond between atoms {atom1_index} and {atom2_index} in this molecule."
        )

    @property
    def has_unique_atom_names(self) -> bool:
        """
        Check if the molecule has unique atom names or not this will help with pdb file writing.
        """
        atom_names = set([atom.atom_name for atom in self.atoms])
        if len(atom_names) == self.n_atoms:
            return True
        return False

    @property
    def improper_torsions(self) -> Optional[List[Tuple[int, int, int, int]]]:
        """A list of improper atom tuples where the first atom is central."""

        improper_torsions = []
        topology = self.to_topology()
        for node in topology.nodes:
            near = sorted(list(nx.neighbors(topology, node)))
            # if the atom has 3 bonds it could be an improper
            # Check if an sp2 carbon or N
            if len(near) == 3 and (
                self.atoms[node].atomic_symbol == "C"
                or self.atoms[node].atomic_symbol == "N"
            ):
                # Store each combination of the improper torsion
                improper_torsions.append((node, near[0], near[1], near[2]))
        return improper_torsions or None

    @property
    def n_improper_torsions(self) -> int:
        """The number of unique improper torsions."""
        impropers = self.improper_torsions
        if impropers is None:
            return 0
        return len(impropers)

    @property
    def angles(self) -> Optional[List[Tuple[int, int, int]]]:
        """A List of angles from the topology."""

        angles = []
        topology = self.to_topology()
        for node in topology.nodes:
            bonded = sorted(list(nx.neighbors(topology, node)))

            # Check that the atom has more than one bond
            if len(bonded) < 2:
                continue

            # Find all possible angle combinations from the list
            for i in range(len(bonded)):
                for j in range(i + 1, len(bonded)):
                    atom1, atom3 = bonded[i], bonded[j]
                    angles.append((atom1, node, atom3))
        return angles or None

    @property
    def charge(self) -> int:
        """
        Return the integer charge of the molecule as the sum of the formal charge.
        """
        return sum([atom.formal_charge for atom in self.atoms])

    @property
    def n_angles(self) -> int:
        """The number of angles in the molecule."""
        angles = self.angles
        if angles is None:
            return 0
        return len(angles)

    def measure_bonds(self) -> Dict[Tuple[int, int], float]:
        """
        Find the length of all bonds in the molecule for the given conformer in  angstroms.

        Returns:
            A dictionary of the bond lengths stored by bond tuple.
        """

        bond_lengths = {}

        for bond in self.bonds:
            atom1 = self.coordinates[bond.atom1_index]
            atom2 = self.coordinates[bond.atom2_index]
            edge = (bond.atom1_index, bond.atom2_index)
            bond_lengths[edge] = np.linalg.norm(atom2 - atom1)

        return bond_lengths

    @property
    def n_bonds(self) -> int:
        """The number of bonds in the topology."""
        bonds = self.bonds
        if bonds is None:
            return 0
        return len(bonds)

    @property
    def dihedrals(
        self,
    ) -> Optional[Dict[Tuple[int, int], List[Tuple[int, int, int, int]]]]:
        """A list of all possible dihedrals that can be found in the topology."""

        dihedrals = {}
        topology = self.to_topology()
        # Work through the network using each edge as a central dihedral bond
        for edge in topology.edges:
            for start in list(nx.neighbors(topology, edge[0])):
                # Check atom not in main bond
                if start != edge[0] and start != edge[1]:
                    for end in list(nx.neighbors(topology, edge[1])):
                        # Check atom not in main bond
                        if end != edge[0] and end != edge[1] and end != start:
                            if edge not in dihedrals:
                                # Add the central edge as a key the first time it is used
                                dihedrals[edge] = [(start, edge[0], edge[1], end)]

                            else:
                                # Add the tuple to the correct key.
                                dihedrals[edge].append((start, edge[0], edge[1], end))

        return dihedrals or None

    @property
    def n_dihedrals(self) -> int:
        """The total number of dihedrals in the molecule."""
        dihedrals = self.dihedrals
        if dihedrals is None:
            return 0
        return sum([len(torsions) for torsions in dihedrals.values()])

    def find_rotatable_bonds(
        self, smirks_to_remove: Optional[List[str]] = None
    ) -> Optional[List[Bond]]:
        """
        Args:
            smirks_to_remove:
                Optional list of smirks patterns which will be discarded
                from the rotatable bonds
        Find all rotatable bonds in the molecule.
        Remove any groups which are not relevant for torsion scans.
            e.g. methyl / amine groups
        return:
            The rotatable bonds in the molecule to be used for torsion scans.
        """

        rotatable_bond_smarts = "[!$(*#*)&!D1:1]-&!@[!$(*#*)&!D1:2]"

        rotatable_matches = self.get_smarts_matches(rotatable_bond_smarts)
        if rotatable_matches is None:
            return None

        if smirks_to_remove is not None:
            for smirk in smirks_to_remove:
                matches_to_remove = self.get_smarts_matches(smirk)
                if matches_to_remove is not None:
                    for match in matches_to_remove:
                        try:
                            rotatable_matches.remove(match)
                        except ValueError:
                            try:
                                # If the match is not in the list, it may be in backwards
                                rotatable_matches.remove(tuple(reversed(match)))
                            except ValueError:
                                continue

        # gather a list of bond instances to return
        rotatable_bonds = [self.get_bond_between(*bond) for bond in rotatable_matches]

        return rotatable_bonds or None

    @property
    def n_rotatable_bonds(self) -> int:
        """The number of rotatable bonds."""
        rotatable_bonds = self.find_rotatable_bonds()
        if rotatable_bonds is None:
            return 0
        return len(rotatable_bonds)

    def symmetrise_nonbonded_parameters(self) -> bool:
        """
        Symmetrise all non-bonded force group parameters.

        Using the CIP rankings from RDKit apply symmetry to the non-bonded force group.

        Important:
            We respect the predefined parameters in the non-bonded force group which can be symmetrised.
        """
        # group atom types as they are in a different format to other types
        atom_types = {}
        for atom_index, cip_type in self.atom_types.items():
            atom_types.setdefault(cip_type, []).append((atom_index,))
        for atoms in atom_types.values():
            self._symmetrise_parameters(
                force_group=self.NonbondedForce, parameter_keys=atoms
            )

        return True

    def symmetrise_bonded_parameters(self) -> bool:
        """
        Symmetrise all bond and angle force group parameters.

        Using the CIP rankings from RDKit apply symmetry to the bond and angle force groups.

        Important:
            We respect the predefined parameters in the bond/angle force group which can be symmetrised.
        """

        for bonds in self.bond_types.values():
            self._symmetrise_parameters(
                force_group=self.BondForce, parameter_keys=bonds
            )

        if self.n_angles > 0:
            for angles in self.angle_types.values():
                self._symmetrise_parameters(
                    force_group=self.AngleForce, parameter_keys=angles
                )
                if self.UreyBradleyForce.n_parameters > 0:
                    self._symmetrise_parameters(
                        force_group=self.UreyBradleyForce, parameter_keys=angles
                    )

        return True

    def _symmetrise_parameters(
        self, force_group: BaseForceGroup, parameter_keys: List[Tuple[int, ...]]
    ):
        """
        Internal method which applies symmetry to a group of parameter references in a particular force group.

        Args:
            force_group: The force group we should query for parameters.
            parameter_keys: The list of atom indices tuples that the symmetry should be applied to.
        """

        symmetry_attrs = force_group.symmetry_parameters()

        raw_parameter_values = {}
        for parameter_key in parameter_keys:
            param = force_group[parameter_key]
            for attr in symmetry_attrs:
                raw_parameter_values.setdefault(attr, []).append(getattr(param, attr))

        # now average the raw values
        for key, value in raw_parameter_values.items():
            raw_parameter_values[key] = np.array(value).mean()

        # now set back
        for parameter_key in parameter_keys:
            force_group.create_parameter(atoms=parameter_key, **raw_parameter_values)

    def measure_dihedrals(
        self,
    ) -> Optional[Dict[Tuple[int, int, int, int], np.ndarray]]:
        """
        For the given conformation measure the dihedrals in the topology in degrees.
        """
        dihedrals = self.dihedrals
        if dihedrals is None:
            return None

        dih_phis = {}

        for val in dihedrals.values():
            for torsion in val:
                # Calculate the dihedral angle in the molecule using the molecule data array.
                x1, x2, x3, x4 = [self.coordinates[torsion[i]] for i in range(4)]
                b1, b2, b3 = x2 - x1, x3 - x2, x4 - x3
                t1 = np.linalg.norm(b2) * np.dot(b1, np.cross(b2, b3))
                t2 = np.dot(np.cross(b1, b2), np.cross(b2, b3))
                dih_phis[torsion] = np.degrees(np.arctan2(t1, t2))

        return dih_phis

    def measure_angles(self) -> Optional[Dict[Tuple[int, int, int], np.ndarray]]:
        """
        For the given conformation measure the angles in the topology in degrees.
        """
        angles = self.angles
        if angles is None:
            return None

        angle_values = {}

        for angle in angles:
            x1 = self.coordinates[angle[0]]
            x2 = self.coordinates[angle[1]]
            x3 = self.coordinates[angle[2]]
            b1, b2 = x1 - x2, x3 - x2
            cosine_angle = np.dot(b1, b2) / (np.linalg.norm(b1) * np.linalg.norm(b2))
            angle_values[angle] = np.degrees(np.arccos(cosine_angle))

        return angle_values

    @property
    def n_atoms(self) -> int:
        """
        Calculate the number of atoms.
        """
        return len(self.atoms)

    def write_parameters(self, file_name: str):
        """
        Take the molecule's parameter set and write an xml file for the molecule.
        """

        tree = self._build_forcefield().getroot()
        messy = ET.tostring(tree, "utf-8")

        pretty_xml_as_string = parseString(messy).toprettyxml(indent="")

        with open(file_name, "w") as xml_doc:
            xml_doc.write(pretty_xml_as_string)

    def _build_forcefield(self):
        """
        Separates the parameters and builds an xml tree ready to be used.

        TODO how do we support OPLS combination rules.
        Important:
            The ordering here should not be changed due to the way sites have to be added.
        """

        # Create XML layout
        root = ET.Element("ForceField")

        ET.SubElement(
            root,
            "QUBEKit",
            attrib={
                "Version": qubekit.__version__,
                "Date": datetime.now().strftime("%Y_%m_%d"),
            },
        )
        AtomTypes = ET.SubElement(root, "AtomTypes")
        Residues = ET.SubElement(root, "Residues")

        resname = "QUP" if self.__class__.__name__ == "Protein" else "MOL"
        Residue = ET.SubElement(Residues, "Residue", name=resname)
        # declare atom `types` and properties
        for atom in self.atoms:
            atom_type = f"QUBE_{atom.atom_index}"
            ET.SubElement(
                AtomTypes,
                "Type",
                attrib={
                    "name": atom_type,
                    "class": str(atom.atom_index),
                    "element": atom.atomic_symbol,
                    "mass": str(atom.atomic_mass),
                },
            )

            ET.SubElement(
                Residue, "Atom", attrib={"name": atom.atom_name, "type": atom_type}
            )

        # add sites to Atomtypes, topology and nonbonded
        for i, site in enumerate(self.extra_sites, start=1):
            site_name = f"v-site{i}"
            site_class = f"X{i}"
            ET.SubElement(
                AtomTypes,
                "Type",
                attrib={"name": site_name, "class": site_class, "mass": "0"},
            )
            # for some reason we swap name and class here but it works !
            ET.SubElement(
                Residue, "Atom", attrib={"name": site_class, "type": site_name}
            )

        BondForce = ET.SubElement(
            root, self.BondForce.openmm_group(), attrib=self.BondForce.xml_data()
        )
        for parameter in self.BondForce:
            ET.SubElement(
                BondForce, parameter.openmm_type(), attrib=parameter.xml_data()
            )
            ET.SubElement(
                Residue,
                "Bond",
                attrib={"from": str(parameter.atoms[0]), "to": str(parameter.atoms[1])},
            )
        AngleForce = ET.SubElement(
            root, self.AngleForce.openmm_group(), attrib=self.AngleForce.xml_data()
        )
        for parameter in self.AngleForce:
            ET.SubElement(
                AngleForce, parameter.openmm_type(), attrib=parameter.xml_data()
            )
        if self.UreyBradleyForce.n_parameters > 0:
            UBForce = ET.SubElement(
                root,
                self.UreyBradleyForce.openmm_group(),
                attrib=self.UreyBradleyForce.xml_data(),
            )
            for parameter in self.UreyBradleyForce:
                ET.SubElement(
                    UBForce, parameter.openmm_type(), attrib=parameter.xml_data()
                )
        if (
            self.TorsionForce.n_parameters > 0
            or self.ImproperTorsionForce.n_parameters > 0
        ):
            TorsionForce = ET.SubElement(
                root,
                self.TorsionForce.openmm_group(),
                attrib=self.TorsionForce.xml_data(),
            )
            for parameter in self.TorsionForce:
                ET.SubElement(
                    TorsionForce, parameter.openmm_type(), attrib=parameter.xml_data()
                )
            for parameter in self.ImproperTorsionForce:
                ET.SubElement(
                    TorsionForce, parameter.openmm_type(), attrib=parameter.xml_data()
                )
        if (
            self.RBTorsionForce.n_parameters > 0
            or self.ImproperRBTorsionForce.n_parameters > 0
        ):
            RBTorsion = ET.SubElement(
                root,
                self.RBTorsionForce.openmm_group(),
                attrib=self.RBTorsionForce.xml_data(),
            )
            for parameter in self.RBTorsionForce:
                ET.SubElement(
                    RBTorsion, parameter.openmm_type(), attrib=parameter.xml_data()
                )
            for parameter in self.ImproperRBTorsionForce:
                ET.SubElement(
                    RBTorsion, parameter.openmm_type(), attrib=parameter.xml_data()
                )

        # now we add more site info after general bonding
        for i, site in enumerate(self.extra_sites):
            site_data = site.xml_data()
            # we have to add its global index
            site_data["index"] = str(i + self.n_atoms)
            ET.SubElement(Residue, site.openmm_type(), attrib=site_data)

        NonbondedForce = ET.SubElement(
            root,
            self.NonbondedForce.openmm_group(),
            attrib=self.NonbondedForce.xml_data(),
        )
        for parameter in self.NonbondedForce:
            ET.SubElement(
                NonbondedForce, parameter.openmm_type(), attrib=parameter.xml_data()
            )

        for i, site in enumerate(self.extra_sites, start=1):
            site_name = f"v-site{i}"
            ET.SubElement(
                NonbondedForce,
                "Atom",
                attrib={
                    "charge": str(site.charge),
                    "epsilon": "0",
                    "sigma": "1",
                    "type": site_name,
                },
            )

        return ET.ElementTree(root)

    @property
    def bond_types(self) -> Dict[str, List[Tuple[int, int]]]:
        """
        Using the symmetry dict, give each bond a code. If any codes match, the bonds can be symmetrised.
        e.g. bond_symmetry_classes = {(0, 3): (2, 0), (0, 4): (2, 0), (0, 5): (2, 0) ...}
        all of the above bonds (tuples) are of the same type (methyl H-C bonds in same region)
        This dict is then used to produce bond_types.
        bond_types is just a dict where the keys are the string code from above and the values are all
        of the bonds with that particular type.
        """
        atom_types = self.atom_types
        bond_symmetry_classes = {}
        for bond in self.bonds:
            bond_symmetry_classes[(bond.atom1_index, bond.atom2_index)] = (
                atom_types[bond.atom1_index],
                atom_types[bond.atom2_index],
            )

        bond_types = {}
        for key, val in bond_symmetry_classes.items():
            bond_types.setdefault(val, []).append(key)

        bond_types = self._cluster_types(bond_types)
        return bond_types

    @property
    def angle_types(self) -> Dict[str, List[Tuple[int, int, int]]]:
        """
        Using the symmetry dict, give each angle a code. If any codes match, the angles can be symmetrised.
        e.g. angle_symmetry_classes = {(1, 0, 3): (3, 2, 0), (1, 0, 4): (3, 2, 0), (1, 0, 5): (3, 2, 0) ...}
        all of the above angles (tuples) are of the same type (methyl H-C-H angles in same region)
        angle_types is just a dict where the keys are the string code from the above and the values are all
        of the angles with that particular type.
        """
        atom_types = self.atom_types
        angle_symmetry_classes = {}
        if self.n_angles == 0:
            return {}

        for angle in self.angles:
            angle_symmetry_classes[angle] = (
                atom_types[angle[0]],
                atom_types[angle[1]],
                atom_types[angle[2]],
            )

        angle_types = {}
        for key, val in angle_symmetry_classes.items():
            angle_types.setdefault(val, []).append(key)

        angle_types = self._cluster_types(angle_types)
        return angle_types

    @property
    def dihedral_types(self) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """
        Using the symmetry dict, give each dihedral a code. If any codes match, the dihedrals can be clustered and their
        parameters should be the same, this is to be used in dihedral fitting so all symmetry equivalent dihedrals are
        optimised at the same time. dihedral_equiv_classes = {(0, 1, 2 ,3): (1, 1, 2, 1)...} all of the tuples are the
        dihedrals index by topology and the strings are the symmetry equivalent atom combinations.
        """

        if self.n_dihedrals == 0:
            return {}

        atom_types = self.atom_types
        dihedral_symmetry_classes = {}
        for dihedral_set in self.dihedrals.values():
            for dihedral in dihedral_set:
                dihedral_symmetry_classes[tuple(dihedral)] = (
                    atom_types[dihedral[0]],
                    atom_types[dihedral[1]],
                    atom_types[dihedral[2]],
                    atom_types[dihedral[3]],
                )

        dihedral_types = {}
        for key, val in dihedral_symmetry_classes.items():
            dihedral_types.setdefault(val, []).append(key)

        dihedral_types = self._cluster_types(dihedral_types)
        return dihedral_types

    @property
    def improper_types(self) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """Using the atom symmetry types work out the improper types."""

        if self.n_improper_torsions == 0:
            return {}

        atom_types = self.atom_types
        improper_symmetry_classes = {}
        for dihedral in self.improper_torsions:
            improper_symmetry_classes[tuple(dihedral)] = (
                atom_types[dihedral[0]],
                atom_types[dihedral[1]],
                atom_types[dihedral[2]],
                atom_types[dihedral[3]],
            )

        improper_types = {}
        for key, val in improper_symmetry_classes.items():
            improper_types.setdefault(val, []).append(key)

        improper_types = self._cluster_types(improper_types)
        return improper_types

    @staticmethod
    def _cluster_types(equiv_classes):
        """
        Function that helps the bond angle and dihedral class finders in clustering the types based on the forward and
        backward type strings.
        :return: clustered equiv class
        """

        new_classes = {}
        for key, item in equiv_classes.items():
            try:
                new_classes[key].extend(item)
            except KeyError:
                try:
                    new_classes[key[::-1]].extend(item)
                except KeyError:
                    new_classes[key] = item

        return new_classes

    @property
    def atom_types(self) -> Dict[int, int]:
        """Returns a dictionary of atom indices mapped to their class or None if there is no rdkit molecule."""

        return RDKit.find_symmetry_classes(self.to_rdkit())

    def to_rdkit(self) -> Chem.Mol:
        """
        Generate an rdkit representation of the QUBEKit ligand object.

        Here we build the molecule and assign the stereochemistry using the coordinates as we should always have a set of coordinates in the model.
        This allows us to skip complicated local vs global stereo chemistry checks however this could break in future.

        Returns:
            An rdkit representation of the molecule.
        """
        from qubekit.utils.helpers import _assert_wrapper

        # TODO what properties should be put in the rdkit molecule? Multiplicity?
        # make an editable molecule
        rd_mol = Chem.RWMol()
        if self.name is not None:
            rd_mol.SetProp("_Name", self.name)

        # when building the molecule we have to loop multiple times
        # so always make sure the indexing is the same in qube and rdkit
        for atom in self.atoms:
            rd_index = rd_mol.AddAtom(atom.to_rdkit())
            assert rd_index == atom.atom_index

        # now we need to add each bond, can not make a bond from python currently
        for bond in self.bonds:
            rd_mol.AddBond(*bond.indices)
            # now get the bond back to edit it
            rd_bond: Chem.Bond = rd_mol.GetBondBetweenAtoms(*bond.indices)
            rd_bond.SetIsAromatic(bond.aromatic)
            rd_bond.SetBondType(bond.rdkit_type)

        Chem.SanitizeMol(
            rd_mol,
            Chem.SANITIZE_ALL ^ Chem.SANITIZE_ADJUSTHS ^ Chem.SANITIZE_SETAROMATICITY,
        )
        # must use openff MDL model for compatibility
        Chem.SetAromaticity(rd_mol, Chem.AromaticityModel.AROMATICITY_MDL)

        # conformers
        rd_mol = RDKit.add_conformer(
            rdkit_mol=rd_mol, conformer_coordinates=self.coordinates
        )
        Chem.AssignStereochemistryFrom3D(rd_mol)

        # now we should check that the stereo has not been broken
        for rd_atom in rd_mol.GetAtoms():
            index = rd_atom.GetIdx()
            qb_atom = self.atoms[index]
            if qb_atom.stereochemistry is not None:
                with _assert_wrapper(StereoChemistryError):
                    assert qb_atom.stereochemistry == rd_atom.GetProp(
                        "_CIPCode"
                    ), f"StereoChemistry incorrect expected {qb_atom.stereochemistry} got {rd_atom.GetProp('_CIPCode')} for atom {qb_atom}"

        for rd_bond in rd_mol.GetBonds():
            index = rd_bond.GetIdx()
            qb_bond = self.bonds[index]
            if qb_bond.stereochemistry is not None:
                rd_bond.SetStereo(qb_bond.rdkit_stereo)
            rd_stereo = rd_bond.GetStereo()
            if qb_bond.stereochemistry == "E":
                with _assert_wrapper(StereoChemistryError):
                    assert (
                        rd_stereo == Chem.BondStereo.STEREOE
                    ), f"StereoChemistry incorrect expected E got {rd_stereo}"
            elif qb_bond.stereochemistry == "Z":
                with _assert_wrapper(StereoChemistryError):
                    assert (
                        rd_stereo == Chem.BondStereo.STEREOZ
                    ), f"StereoChemistry incorrect expected Z got {rd_stereo}"

        return Chem.Mol(rd_mol)

    def get_smarts_matches(self, smirks: str) -> Optional[List[Tuple[int, ...]]]:
        """
        Get substructure matches for a mapped SMARTS pattern.

        Args:
            smirks:
                The mapped SMARTS pattern that should be used to query the molecule.

        Returns:
            `None` if there are no matches, else a list of tuples of atom indices which match the tagged atoms in
            the SMARTS pattern. These are returned in the same order.
        """
        matches = RDKit.get_smirks_matches(rdkit_mol=self.to_rdkit(), smirks=smirks)
        if not matches:
            return None
        return matches

    def add_qm_scan(self, scan_data: TorsionDriveData) -> None:
        """
        Save the torsion drive data into the ligand object.
        """
        if scan_data.__class__ != TorsionDriveData:
            raise MissingReferenceData(
                "The reference data must be in the form of the torsion drive data class."
            )
        else:
            if self.qm_scans is None:
                self.qm_scans = []
            self.qm_scans.append(scan_data)

    def openmm_coordinates(self) -> unit.Quantity:
        """
        Convert the coordinates to an openMM quantity.

        Build a single set of coordinates for the molecule that work in openMM.
        Note this must be a single conformer, if multiple are given only the first is used.

        Returns:
            A openMM quantity wrapped array of the coordinates in angstrom.
        """
        return unit.Quantity(self.coordinates, unit.angstroms)

    def fix_net_charge(self):
        """
        Ensure the total is exactly equal to the ideal net charge of the molecule.
        If net charge is not an integer value, MM simulations can (ex/im)plode.
        """

        decimal.setcontext(decimal.Context(prec=7))
        round_to = decimal.Decimal(10) ** -6

        for param in self.NonbondedForce:
            param.charge = param.charge.quantize(round_to)

        atom_charges = sum(param.charge for param in self.NonbondedForce)
        extra = self.charge - atom_charges

        if self.extra_sites is not None:
            for site in self.extra_sites:
                site.charge = site.charge.quantize(round_to)
                extra -= site.charge

        if extra:
            last_atom_index = self.n_atoms - 1
            self.NonbondedForce[(last_atom_index,)].charge += extra

    @classmethod
    def from_rdkit(
        cls, rdkit_mol: Chem.Mol, name: Optional[str] = None, multiplicity: int = 1
    ) -> "Molecule":
        """
        Build an instance of a qubekit ligand directly from an rdkit molecule.

        Args:
            rdkit_mol:
                An instance of an rdkit.Chem.Mol from which the QUBEKit ligand should be built.
            name:
                The name that should be assigned to the molecule, this will overwrite any name already assigned.
            multiplicity:
                The multiplicity of the molecule, used in QM calculations.
        """
        if name is None:
            if rdkit_mol.HasProp("_Name"):
                name = rdkit_mol.GetProp("_Name")

        atoms = []
        bonds = []
        # Collect the atom names and bonds
        for rd_atom in rdkit_mol.GetAtoms():
            # make and atom
            qb_atom = Atom.from_rdkit(rd_atom=rd_atom)
            atoms.append(qb_atom)

        # now we need to make a list of bonds
        for rd_bond in rdkit_mol.GetBonds():
            qb_bond = Bond.from_rdkit(rd_bond=rd_bond)
            bonds.append(qb_bond)

        coords = rdkit_mol.GetConformer().GetPositions()
        bonds = bonds or None
        # method use to make the molecule
        routine = ["QUBEKit.ligand.from_rdkit"]
        return cls(
            atoms=atoms,
            bonds=bonds,
            coordinates=coords,
            multiplicity=multiplicity,
            name=name,
            routine=routine,
        )

    def has_ub_terms(self) -> bool:
        """Return `True` if the molecule has Urey-Bradley terms, as there are forces between non-bonded atoms."""
        if self.UreyBradleyForce.n_parameters > 0:
            return True

        return False

    @staticmethod
    def _check_file_name(file_name: str) -> None:
        """
        Make sure that if an unsupported file type is passed we can not make a molecule from it.
        """
        if ".xyz" in file_name:
            raise FileTypeError(
                "XYZ files can not be used to build ligands due to ambiguous bonding, "
                "please use pdb, mol, mol2 or smiles as input."
            )

    @classmethod
    def from_file(cls, file_name: str, multiplicity: int = 1) -> "Molecule":
        """
        Build a ligand from a supported input file.

        Args:
            file_name:
                The abs path to the file including the extension which determines how the file is read.
            multiplicity:
                The multiplicity of the molecule which is required for QM calculations.
        """
        cls._check_file_name(file_name=file_name)
        input_data = ReadInput.from_file(file_name=file_name)
        ligand = cls.from_rdkit(
            rdkit_mol=input_data.rdkit_mol,
            name=input_data.name,
            multiplicity=multiplicity,
        )
        # now edit the routine to include this call
        ligand.provenance["routine"].extend(
            ["QUBEKit.ligand.from_file", os.path.abspath(file_name)]
        )
        return ligand

    @classmethod
    def from_smiles(
        cls, smiles_string: str, name: str, multiplicity: int = 1
    ) -> "Molecule":
        """
        Build the ligand molecule directly from a non mapped smiles string.

        Args:
            smiles_string:
                The smiles string from which a molecule instance should be made.
            name:
                The name that should be assigned to the molecule.
            multiplicity:
                The multiplicity of the molecule, important for QM calculations.
        """
        input_data = ReadInput.from_smiles(smiles=smiles_string, name=name)
        ligand = cls.from_rdkit(
            rdkit_mol=input_data.rdkit_mol, name=name, multiplicity=multiplicity
        )
        # now edit the routine to include this command
        ligand.provenance["routine"].extend(
            ["QUBEKit.ligand.from_smiles", smiles_string]
        )
        return ligand

    def to_openmm_topology(self) -> Topology:
        """
        Convert the Molecule to a OpenMM topology representation.

        We assume we have a single molecule so a single chain is made with a single residue.
        Note this will not work with proteins as we will need to have distinct residues.

        Returns:
            An openMM topology object which can be used to construct a system.
        """
        topology = Topology()
        bond_types = {1: Single, 2: Double, 3: Triple}
        chain = topology.addChain()
        # create a molecule specific residue
        residue = topology.addResidue(name=self.name, chain=chain)
        # add atoms and keep track so we can add bonds
        top_atoms = []
        for atom in self.atoms:
            element = Element.getByAtomicNumber(atom.atomic_number)
            top_atom = topology.addAtom(
                name=atom.atom_name, element=element, residue=residue
            )
            top_atoms.append(top_atom)
        for bond in self.bonds:
            atom1 = top_atoms[bond.atom1_index]
            atom2 = top_atoms[bond.atom2_index]
            # work out the type
            if bond.aromatic:
                b_type = Aromatic
            else:
                b_type = bond_types[bond.bond_order]
            topology.addBond(
                atom1=atom1, atom2=atom2, type=b_type, order=bond.bond_order
            )

        return topology

    def to_smiles(
        self,
        isomeric: bool = True,
        explicit_hydrogens: bool = True,
        mapped: bool = False,
    ) -> str:
        """
        Create a canonical smiles representation for the molecule based on the input settings.

        Args:
            isomeric:
                If the smiles string should encode stereochemistry `True` or not `False`.
            explicit_hydrogens:
                If hydrogens should be explicitly encoded into the smiles string `True` or not `False`.
            mapped:
                If the smiles should encode the original atom ordering `True` or not `False` as this might be different
                from the canonical ordering.

        Returns:
            A smiles string which encodes the molecule with the desired settings.
        """
        return RDKit.get_smiles(
            rdkit_mol=self.to_rdkit(),
            isomeric=isomeric,
            explicit_hydrogens=explicit_hydrogens,
            mapped=mapped,
        )

    def generate_atom_names(self) -> None:
        """
        Generate a unique set of atom names for the molecule.
        """
        atom_names = {}
        for atom in self.atoms:
            symbol = atom.atomic_symbol
            if symbol not in atom_names:
                atom_names[symbol] = 1
            else:
                atom_names[symbol] += 1

            atom.atom_name = f"{symbol}{atom_names[symbol]}"

    def _validate_atom_names(self) -> None:
        """
        Check that the ligand has unique atom names if not generate a new set.
        """
        if not self.has_unique_atom_names:
            self.generate_atom_names()

    def to_qcschema(self, extras: Optional[Dict] = None) -> qcel.models.Molecule:
        """
        build a qcschema molecule from the ligand object, this is useful to interface with QCEngine and QCArchive.
        """
        import copy

        # make sure we have a conformer
        if self.coordinates == [] or self.coordinates is None:
            raise ConformerError(
                "The molecule must have a conformation to make a qcschema molecule."
            )
        coords = copy.deepcopy(self.coordinates)
        # input must be in bohr
        coords *= constants.ANGS_TO_BOHR
        # we do not store explicit bond order so guess at 1
        bonds = [
            (bond.atom1_index, bond.atom2_index, bond.bond_order) for bond in self.bonds
        ]
        mapped_smiles = self.to_smiles(
            isomeric=True, explicit_hydrogens=True, mapped=True
        )
        if extras is not None:
            extras["canonical_isomeric_explicit_hydrogen_mapped_smiles"] = mapped_smiles
        else:
            extras = {
                "canonical_isomeric_explicit_hydrogen_mapped_smiles": mapped_smiles
            }

        symbols = [atom.atomic_symbol for atom in self.atoms]
        schema_info = {
            "symbols": symbols,
            "geometry": coords,
            "connectivity": bonds,
            "molecular_charge": self.charge,
            "molecular_multiplicity": self.multiplicity,
            "extras": extras,
            "fix_com": True,
            "fix_orientation": True,
            "fix_symmetry": "c1",
        }
        return qcel.models.Molecule.from_data(schema_info, validate=True)

    def add_conformer(self, file_name: str) -> None:
        """
        Read the given input file extract  the conformers and save them to the ligand.
        TODO do we want to check that the connectivity is the same?
        """
        input_data = ReadInput.from_file(file_name=file_name)
        if input_data.coords is None:
            # get the coords from the rdkit molecule
            coords = input_data.rdkit_mol.GetConformer().GetPositions()
        else:
            if isinstance(input_data.coords, list):
                coords = input_data.coords[-1]
            else:
                coords = input_data.coords
        self.coordinates = coords

    def to_offxml(self, file_name: str, h_constraints: bool = True):
        """
        Build an offxml for the molecule and raise an error if we do not think this will be possible due to the presence
        of v-sites or if the potential can not be accurately transferred to an equivalent openff potential.

        Note:
            There are a limited number of currently supported potentials in openff without using smirnoff-plugins.
        """

        offxml = self._build_offxml_general(h_constraints=h_constraints)
        self.add_params_to_offxml(offxml=offxml)
        offxml.to_file(filename=file_name)

    def add_params_to_offxml(
        self,
        offxml: ForceField,
        include_torsions: bool = True,
        parameterize: bool = False,
    ):
        """
        Edits the force field in place by adding this molecules force field parameters.

        Args:
            include_torsions:
                If the proper torsion parameters should also be included, this is only used when building transferable
                force fields between fragments and ligands.
            parameterize:
                If any dihedrals passing through a scanned central bond should be marked for optimisation.
        """

        if self.ImproperRBTorsionForce.n_parameters > 0:
            raise NotImplementedError(
                "RB Improper Torsions can not yet be safely converted into offxml format yet."
            )

        self._build_offxml_bonds(offxml=offxml)
        self._build_offxml_angles(offxml=offxml)
        if include_torsions and self.TorsionForce.n_parameters > 0:
            self._build_offxml_torsions(offxml=offxml, parameterize=parameterize)
        if self.ImproperTorsionForce.n_parameters > 0:
            self._build_offxml_improper_torsions(offxml=offxml)
        if self.RBTorsionForce.n_parameters > 0:
            self._build_offxml_rb_torsions(offxml=offxml)
        self._build_offxml_vdw(offxml=offxml)
        self._build_offxml_charges(offxml=offxml)
        self._build_offxml_vs(offxml=offxml)

    def _build_offxml_bonds(self, offxml: ForceField):
        """Edit the offxml in place by adding the bonds for this molecule to the harmonic bond section"""

        rdkit_mol = self.to_rdkit()
        bond_handler = offxml.get_parameter_handler("Bonds")
        bond_types = self.bond_types
        # for each bond type collection create a single smirks pattern
        for bonds in bond_types.values():
            graph = ClusterGraph(
                mols=[rdkit_mol], smirks_atoms_lists=[bonds], layers="all"
            )
            qube_bond = self.BondForce[bonds[0]]
            bond_handler.add_parameter(
                parameter_kwargs={
                    "smirks": graph.as_smirks(),
                    "length": qube_bond.length * unit.nanometers,
                    "k": qube_bond.k * unit.kilojoule_per_mole / unit.nanometers**2,
                }
            )

    def _build_offxml_angles(self, offxml: ForceField):
        """Edit the offxml in place by adding the angles for this molecule to the harmonic angle section"""

        rdkit_mol = self.to_rdkit()
        has_ub_terms = self.has_ub_terms()
        if has_ub_terms:
            angle_handler = offxml.get_parameter_handler("UreyBradley")
        else:
            angle_handler = offxml.get_parameter_handler("Angles")

        angle_types = self.angle_types
        for angles in angle_types.values():
            graph = ClusterGraph(
                mols=[rdkit_mol],
                smirks_atoms_lists=[angles],
                layers="all",
            )
            qube_angle = self.AngleForce[angles[0]]
            angle_data = {
                "smirks": graph.as_smirks(),
                "angle": qube_angle.angle * unit.radian,
            }
            if has_ub_terms:
                angle_data["angle_k"] = (
                    qube_angle.k * unit.kilojoule_per_mole / unit.radians**2
                )
                qube_ub = self.UreyBradleyForce[angles[0]]
                angle_data["bond_length"] = qube_ub.d * unit.nanometers
                angle_data["bond_k"] = (
                    qube_ub.k * unit.kilojoule_per_mole / unit.nanometers**2
                )
            else:
                angle_data["k"] = (
                    qube_angle.k * unit.kilojoule_per_mole / unit.radians**2
                )
            angle_handler.add_parameter(parameter_kwargs=angle_data)

    def _build_offxml_torsions(self, offxml: ForceField, parameterize: bool = False):
        """
        Edit the offxml in place by adding this molecules proper torsion force field parameters.

        Args:
            parameterize:
                If torsions passing through a scanned central bond should be marked for optimisation `True` or not `False`.
        """

        rdkit_mol = self.to_rdkit()
        proper_torsions = offxml.get_parameter_handler("ProperTorsions")
        torsion_types = self.dihedral_types
        if self.qm_scans is None:
            scanned_bonds = []
        else:
            scanned_bonds = [
                torsiondrive_data.central_bond for torsiondrive_data in self.qm_scans
            ]
        for dihedrals in torsion_types.values():
            try:
                qube_dihedral = self.TorsionForce[dihedrals[0]]
                graph = ClusterGraph(
                    mols=[rdkit_mol],
                    smirks_atoms_lists=[dihedrals],
                    layers="all",
                )
                torsion_data = {
                    "smirks": graph.as_smirks(),
                    "k1": qube_dihedral.k1 * unit.kilojoule_per_mole,
                    "k2": qube_dihedral.k2 * unit.kilojoule_per_mole,
                    "k3": qube_dihedral.k3 * unit.kilojoule_per_mole,
                    "k4": qube_dihedral.k4 * unit.kilojoule_per_mole,
                    "periodicity1": qube_dihedral.periodicity1,
                    "periodicity2": qube_dihedral.periodicity2,
                    "periodicity3": qube_dihedral.periodicity3,
                    "periodicity4": qube_dihedral.periodicity4,
                    "phase1": qube_dihedral.phase1 * unit.radians,
                    "phase2": qube_dihedral.phase2 * unit.radians,
                    "phase3": qube_dihedral.phase3 * unit.radians,
                    "phase4": qube_dihedral.phase4 * unit.radians,
                    "idivf1": 1,
                    "idivf2": 1,
                    "idivf3": 1,
                    "idivf4": 1,
                }
                if parameterize:
                    # we need to check if any of the torsions in this symmetry group pass through a scanned bond
                    for dihedral in dihedrals:
                        if (
                            tuple(dihedral[1:3]) in scanned_bonds
                            or tuple(reversed(dihedral[1:3])) in scanned_bonds
                        ):
                            torsion_data["parameterize"] = "k1, k2, k3, k4"
                            torsion_data["allow_cosmetic_attributes"] = True
                            break

                proper_torsions.add_parameter(parameter_kwargs=torsion_data)
            except MissingParameterError:
                # we have no torsion force for this dihedral it could be handled by RB torsions
                continue

    def _build_offxml_rb_torsions(self, offxml: ForceField):
        """
        Edit the offxml in place and add RB Proper torsions using our plugin handler.
        """

        # we need a dummy proper torsion to cover the dihedrals which have RB terms
        if "ProperTorsions" in offxml.registered_parameter_handlers:
            periodic_torsions = offxml.get_parameter_handler("ProperTorsions")
            generic_smirks = "[*:1]~[*:2]~[*:3]~[*:4]"
            try:
                _ = periodic_torsions[generic_smirks]
            except ParameterLookupError:
                # make sure it is inserted at the start of the handlers
                periodic_torsions.add_parameter(
                    parameter_kwargs={
                        "smirks": generic_smirks,
                        "periodicity1": 1,
                        "phase1": 0.0 * unit.degree,
                        "k1": 0.0 * unit.kilojoule_per_mole,
                        "idivf1": 1,
                    },
                    before=0,
                )

        rdkit_mol = self.to_rdkit()
        proper_torsions = offxml.get_parameter_handler("ProperRyckhaertBellemans")
        # we only use these torsion types for flexible non-aromatic ring movements applied by QForce
        # not sure if we need to also group by symmetry?
        torsion_types = self.dihedral_types
        for dihedrals in torsion_types.values():
            try:
                rb_torsion = self.RBTorsionForce[dihedrals[0]]
                graph = ClusterGraph(
                    mols=[rdkit_mol],
                    smirks_atoms_lists=[dihedrals],
                    layers="all",
                )
                proper_torsions.add_parameter(
                    parameter_kwargs={
                        "smirks": graph.as_smirks(),
                        "c0": rb_torsion.c0 * unit.kilojoule_per_mole,
                        "c1": rb_torsion.c1 * unit.kilojoule_per_mole,
                        "c2": rb_torsion.c2 * unit.kilojoule_per_mole,
                        "c3": rb_torsion.c3 * unit.kilojoule_per_mole,
                        "c4": rb_torsion.c4 * unit.kilojoule_per_mole,
                        "c5": rb_torsion.c5 * unit.kilojoule_per_mole,
                    }
                )
            except MissingParameterError:
                # there is rb torsion force for these dihedrals so skip over them?
                continue

    def _build_offxml_improper_torsions(self, offxml: ForceField):
        """Edit the offxml in place and add periodic improper torsions."""

        rdkit_mol = self.to_rdkit()
        improper_torsions = offxml.get_parameter_handler("ImproperTorsions")
        improper_types = self.improper_types
        for torsions in improper_types.values():
            impropers = [
                (improper[1], improper[0], *improper[2:]) for improper in torsions
            ]
            graph = ClusterGraph(
                mols=[rdkit_mol], smirks_atoms_lists=[impropers], layers="all"
            )
            qube_improper = self.ImproperTorsionForce[torsions[0]]
            # we need to multiply each k value by 3 as they will be applied as trefoil see
            # <https://openforcefield.github.io/standards/standards/smirnoff/#impropertorsions> for more details
            # we assume we only have a k2 term for improper torsions via a periodic term
            improper_torsions.add_parameter(
                parameter_kwargs={
                    "smirks": graph.as_smirks(),
                    "k1": qube_improper.k2 * 3 * unit.kilojoule_per_mole,
                    "periodicity1": qube_improper.periodicity2,
                    "phase1": qube_improper.phase2 * unit.radians,
                }
            )

    def _build_offxml_vdw(self, offxml: ForceField):
        """Edit the offxml in place and add the vdw parameters via the normal LJ nonbonded force."""

        rdkit_mol = self.to_rdkit()
        vdw_handler = offxml.get_parameter_handler("vdW")
        atom_types = {}
        for atom_index, cip_type in self.atom_types.items():
            atom_types.setdefault(cip_type, []).append((atom_index,))
        for sym_set in atom_types.values():
            graph = ClusterGraph(
                mols=[rdkit_mol], smirks_atoms_lists=[sym_set], layers="all"
            )
            qube_non_bond = self.NonbondedForce[sym_set[0]]
            vdw_handler.add_parameter(
                parameter_kwargs={
                    "smirks": graph.as_smirks(),
                    "epsilon": qube_non_bond.epsilon * unit.kilojoule_per_mole,
                    "sigma": qube_non_bond.sigma * unit.nanometers,
                }
            )

    @classmethod
    def _build_offxml_general(cls, h_constraints: bool = True):
        """Initiate a custom offxml file with general metadata"""

        offxml = ForceField(allow_cosmetic_attributes=True, load_plugins=True)
        offxml.author = f"QUBEKit_version_{qubekit.__version__}"
        offxml.date = datetime.now().strftime("%Y_%m_%d")

        if h_constraints:
            # add a generic h-bond constraint
            constraints = offxml.get_parameter_handler("Constraints")
            constraints.add_parameter(
                parameter_kwargs={"smirks": "[#1:1]-[*:2]", "id": "h-c1"}
            )

        # add a standard Electrostatic tag
        _ = offxml.get_parameter_handler(
            "Electrostatics", handler_kwargs={"scale14": 0.8333333333, "version": 0.3}
        )

        return offxml

    def _build_offxml_charges(self, offxml: ForceField):
        """Edit the offxml in place by adding the charges as library charges."""

        rdkit_mol = self.to_rdkit()
        library_charges = offxml.get_parameter_handler("LibraryCharges")
        charge_data = dict(
            (f"charge{param.atoms[0] + 1}", param.charge * unit.elementary_charge)
            for param in self.NonbondedForce
        )
        graph = ClusterGraph(
            mols=[rdkit_mol],
            smirks_atoms_lists=[[list([i for i in range(self.n_atoms)])]],
            layers="all",
        )
        charge_data["smirks"] = graph.as_smirks()
        library_charges.add_parameter(parameter_kwargs=charge_data)

    def _build_offxml_volumes(self, offxml: ForceField):
        """Edit the offxml in place by adding the atomic volumes using the plugin handler"""

        rdkit_mol = self.to_rdkit()
        qube_handler = offxml.get_parameter_handler("QUBEKitvdWTS")
        volume_data = dict(
            (f"volume{atom.atom_index + 1}", atom.aim.volume * unit.bohr**3)
            for atom in self.atoms
        )
        graph = ClusterGraph(
            mols=[rdkit_mol],
            smirks_atoms_lists=[[list([i for i in range(self.n_atoms)])]],
            layers="all",
        )
        volume_data["smirks"] = graph.as_smirks()
        qube_handler.add_parameter(parameter_kwargs=volume_data)

    def _build_offxml_vs(self, offxml: ForceField):
        """Edit the offxml in place adding any virtual sites via our custom plugin handler."""

        if self.extra_sites.n_sites == 0:
            return

        rdkit_mol = self.to_rdkit()
        # use our local coordinate vsite plugin
        local_vsites = offxml.get_parameter_handler("LocalCoordinateVirtualSites")
        # we need to work around duplicate smirks patterns so we add them our self
        for i, site in enumerate(self.extra_sites):
            site_type = "local3p" if site.type == "VirtualSite3Point" else "local4p"
            atoms = [site.parent_index, site.closest_a_index, site.closest_b_index]
            if site_type == "local4p":
                atoms.append(site.closest_c_index)

            graph = ClusterGraph(
                mols=[rdkit_mol],
                smirks_atoms_lists=[[atoms]],
                layers="all",
            )
            vsite_parameter = local_vsites._INFOTYPE(
                **{
                    "smirks": graph.as_smirks(),
                    "name": f"site_{i}",
                    "x_local": site.p1 * unit.nanometers,
                    "y_local": site.p2 * unit.nanometers,
                    "z_local": site.p3 * unit.nanometers,
                    "o_weights": site.o_weights,
                    "x_weights": site.x_weights,
                    "y_weights": site.y_weights,
                    "charge": site.charge * unit.elementary_charge,
                    "epsilon": 0 * unit.kilojoule_per_mole,
                    "sigma": 1 * unit.nanometer,
                    "type": "local",
                }
            )
            local_vsites._parameters.append(vsite_parameter)

    def get_atom_with_map_index(self, map_index: int) -> Atom:
        """
        Get the atom in the molecule which has the requested map index.
        """
        for atom in self.atoms:
            if atom.map_index == map_index:
                return atom


class Fragment(Molecule):
    """
    Fragments use bond indices to identify rotatable bonds and can be stored in Ligands that they are related to.
    """

    type: Literal["Fragment"] = "Fragment"
    bond_indices: List[Tuple[int, int]] = Field(
        default_factory=list,
        description="The map indices of the atoms in the parent molecule that are involved in the bond. "
        "The fragment was built around these atoms. Note that one fragment might have more "
        "than one torsion bond for performance reasons.",
    )


class Ligand(Fragment):
    """
    the main data class for QUBEKit which describes a small molecule and its fragments when required.
    """

    type: Literal["Ligand"] = "Ligand"
    fragments: Optional[List[Fragment]] = Field(
        None,
        description="Fragments in the molecule with the bonds around which the fragments were built.",
    )

    @property
    def n_fragments(self) -> int:
        return 0 if self.fragments is None else len(self.fragments)

    def _optimizeable_offxml(self, file_name: str, h_constraints: bool = False):
        """
        Build an optimizeable offxml for the torsion optimisation stage, which correctly handles any fragments of
        the molecule. This logic has been moved from the ForceBalance wrapper to make it more general.
        """

        offxml = self._build_offxml_general(h_constraints=h_constraints)

        if self.n_fragments == 0 or self.qm_scans is not None:
            # If there are no fragments pass the parent molecule through with the correct tags
            self.add_params_to_offxml(
                offxml=offxml, include_torsions=True, parameterize=True
            )
        if self.n_fragments != 0:
            for fragment in self.fragments:
                fragment.add_params_to_offxml(offxml=offxml, include_torsions=False)
                self._build_transferable_torsions(fragment=fragment, offxml=offxml)

        offxml.to_file(filename=file_name)

    def _build_transferable_torsions(self, fragment: Fragment, offxml: ForceField):
        """Edit an offxml in place by adding proper torsions from the fragment molecule and for scanned torsions create
        a trasferable smirks pattern."""
        parent_rdkit = self.to_rdkit()
        fragment_rdkit = fragment.to_rdkit()
        proper_torsions = offxml.get_parameter_handler("ProperTorsions")
        fragment_torsion_types = fragment.dihedral_types
        scanned_bonds = [
            torsiondrive_data.central_bond for torsiondrive_data in fragment.qm_scans
        ]
        for dihedrals in fragment_torsion_types.values():
            qube_dihedral = fragment.TorsionForce[dihedrals[0]]
            mols = [fragment_rdkit]
            smirks_atoms_lists = [dihedrals]

            # check any of the dihedrals in this symmetry group run through a scanned bond
            include_parent = False
            corresponding_parent_dihedrals = []
            frag_dihedral_maps = []

            for dihedral in dihedrals:
                # check to see if this should be a transferable parameter
                if (
                    tuple(dihedral[1:3]) in scanned_bonds
                    or tuple(reversed(dihedral[1:3])) in scanned_bonds
                ):
                    include_parent = True

                # get the map indices of the dihedral in the correct order
                frag_dihedral_map = [fragment.atoms[i].map_index for i in dihedral]
                frag_dihedral_maps.append(frag_dihedral_map)

            if include_parent:
                mols.append(parent_rdkit)
                # find the corresponding dihedral atom indices in the parent with the same ordering
                for frag_dihedral_map in frag_dihedral_maps:
                    corresponding_parent_dihedral_real = [
                        self.get_atom_with_map_index(map_index=i).atom_index
                        for i in frag_dihedral_map
                    ]
                    corresponding_parent_dihedrals.append(
                        tuple(corresponding_parent_dihedral_real)
                    )
                smirks_atoms_lists.append(corresponding_parent_dihedrals)

            graph = ClusterGraph(
                mols=mols,
                smirks_atoms_lists=smirks_atoms_lists,
                layers="all",
            )

            # build the kwargs
            torsion_data = {
                "smirks": graph.as_smirks(),
                "k1": qube_dihedral.k1 * unit.kilojoule_per_mole,
                "k2": qube_dihedral.k2 * unit.kilojoule_per_mole,
                "k3": qube_dihedral.k3 * unit.kilojoule_per_mole,
                "k4": qube_dihedral.k4 * unit.kilojoule_per_mole,
                "periodicity1": qube_dihedral.periodicity1,
                "periodicity2": qube_dihedral.periodicity2,
                "periodicity3": qube_dihedral.periodicity3,
                "periodicity4": qube_dihedral.periodicity4,
                "phase1": qube_dihedral.phase1 * unit.radians,
                "phase2": qube_dihedral.phase2 * unit.radians,
                "phase3": qube_dihedral.phase3 * unit.radians,
                "phase4": qube_dihedral.phase4 * unit.radians,
                "idivf1": 1,
                "idivf2": 1,
                "idivf3": 1,
                "idivf4": 1,
            }
            if include_parent:
                # add the tags to mark as optimisable
                torsion_data["parameterize"] = "k1, k2, k3, k4"
                torsion_data["allow_cosmetic_attributes"] = True

            proper_torsions.add_parameter(parameter_kwargs=torsion_data)
