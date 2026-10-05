"""Outils pour la récupération de données et le calcul de descripteurs."""

import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from langchain_core.tools import tool
from typing import Optional


@tool
def search_pubchem_by_name(molecule_name: str) -> dict:
    """Recherche une molécule dans PubChem par son nom.
    
    Args:
        molecule_name: Nom de la molécule (ex: "aspirin", "caffeine")
    
    Returns:
        Dictionnaire avec le CID et les propriétés, ou une erreur.
    """
    try:
        compounds = pcp.get_compounds(molecule_name, 'name')
        if not compounds:
            return {"error": f"Molécule '{molecule_name}' non trouvée dans PubChem"}
        
        compound = compounds[0]
        return {
            "cid": compound.cid,
            "molecular_formula": compound.molecular_formula,
            "molecular_weight": compound.molecular_weight,
            "canonical_smiles": compound.connectivity_smiles,
            "xlogp": compound.xlogp,
            "tpsa": compound.tpsa,
            "h_bond_donor_count": compound.h_bond_donor_count,
            "h_bond_acceptor_count": compound.h_bond_acceptor_count,
        }
    except Exception as e:
        return {"error": f"Erreur lors de la recherche: {str(e)}"}


@tool
def get_pubchem_properties(cid: int) -> dict:
    """Récupère les propriétés d'une molécule PubChem par son CID.
    
    Args:
        cid: Identifiant PubChem Compound
    
    Returns:
        Dictionnaire avec les propriétés demandées.
    """
    try:
        compound = pcp.Compound.from_cid(cid)
        return {
            "cid": compound.cid,
            "molecular_weight": compound.molecular_weight,
            "xlogp": compound.xlogp,
            "tpsa": compound.tpsa,
            "h_bond_donor_count": compound.h_bond_donor_count,
            "h_bond_acceptor_count": compound.h_bond_acceptor_count,
            "canonical_smiles": compound.connectivity_smiles,
        }
    except Exception as e:
        return {"error": f"Erreur pour CID {cid}: {str(e)}"}


@tool
def compute_rdkit_descriptors(smiles: str) -> dict:
    """Calcule les descripteurs moléculaires RDKit à partir d'un SMILES.
    
    Args:
        smiles: Représentation SMILES de la molécule
    
    Returns:
        Dictionnaire avec les descripteurs calculés.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"SMILES invalide: '{smiles}'"}
    
    return {
        "smiles": smiles,
        "molecular_weight": round(Descriptors.MolWt(mol), 2),
        "mol_logp": round(Descriptors.MolLogP(mol), 2),
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "h_donors": Lipinski.NumHDonors(mol),
        "h_acceptors": Lipinski.NumHAcceptors(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "ring_count": Lipinski.RingCount(mol),
        "heavy_atom_count": mol.GetNumHeavyAtoms(),
    }


@tool
def predict_property_from_descriptors(
    mol_weight: float,
    mol_logp: float,
    tpsa: float,
    h_donors: int,
    h_acceptors: int,
    target_property: str
) -> dict:
    """Prédit une propriété physico-chimique à partir des descripteurs RDKit.
    
    Args:
        mol_weight: Masse molaire
        mol_logp: Coefficient de partition octanol/eau
        tpsa: Surface polaire topologique
        h_donors: Nombre de donneurs de liaisons hydrogène
        h_acceptors: Nombre d'accepteurs de liaisons hydrogène
        target_property: Propriété cible ("logp", "solubility", "permeability")
    
    Returns:
        Prédiction avec justification.
    """
    if target_property.lower() == "logp":
        return {
            "property": "logp",
            "predicted_value": mol_logp,
            "method": "RDKit MolLogP",
            "reliability": "high",
            "justification": f"Calcul direct RDKit: {mol_logp}"
        }
    
    elif target_property.lower() == "solubility":
        # Heuristique basée sur la règle de Lipinski
        if mol_logp > 5 or mol_weight > 500:
            solubility = "faible"
            reason = f"LogP élevé ({mol_logp}) et/ou masse élevée ({mol_weight})"
        elif mol_logp < 0 or tpsa > 100:
            solubility = "haute"
            reason = f"LogP faible ({mol_logp}) et polarité élevée (TPSA={tpsa})"
        else:
            solubility = "moyenne"
            reason = f"Propriétés intermédiaires (LogP={mol_logp}, TPSA={tpsa})"
        
        return {
            "property": "solubility",
            "predicted_value": solubility,
            "method": "Lipinski heuristic",
            "reliability": "medium",
            "justification": reason
        }
    
    elif target_property.lower() == "permeability":
        # Heuristique pour la perméabilité membranaire
        if 1 < mol_logp < 4 and tpsa < 90 and h_donors < 5:
            permeability = "haute"
            reason = "Respecte les critères de perméabilité optimale"
        else:
            permeability = "faible"
            reason = f"LogP={mol_logp} (optimal: 1-4), TPSA={tpsa} (optimal: <90)"
        
        return {
            "property": "permeability",
            "predicted_value": permeability,
            "method": "Lipinski heuristic",
            "reliability": "medium",
            "justification": reason
        }
    
    else:
        return {
            "error": f"Propriété '{target_property}' non supportée",
            "supported": ["logp", "solubility", "permeability"]
        }


# Liste des outils disponibles
ALL_TOOLS = [
    search_pubchem_by_name,
    get_pubchem_properties,
    compute_rdkit_descriptors,
    predict_property_from_descriptors,
]
