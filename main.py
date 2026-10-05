"""Point d'entrée principal avec tests de validation."""

import os
import sys
import time
from dotenv import load_dotenv
from agents import create_llm, create_chem_supervisor

load_dotenv()


def run_single_test(app, query: str, test_name: str) -> dict:
    """Exécute un test unique et affiche les résultats."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"Requête: {query}")
    print('='*60)
    
    initial_state = {
        "messages": [{"role": "user", "content": query}],
        "molecule_name": None,
        "molecule_cid": None,
        "smiles": None,
        "target_property": None,
        "pubchem_data": None,
        "rdkit_descriptors": None,
        "prediction_result": None,
        "suggestions": [],
        "status": "pending",
        "errors": [],
    }
    
    start_time = time.time()
    
    try:
        result = app.invoke(
            initial_state,
            config={"recursion_limit": 30}
        )
        elapsed = time.time() - start_time
        
        print(f"\n--- Résultats (temps: {elapsed:.1f}s) ---")
        print(f"Statut: {result.get('status', 'unknown')}")
        
        if result.get("pubchem_data"):
            print(f"Données PubChem: {result['pubchem_data']}")
        
        if result.get("rdkit_descriptors"):
            print(f"Descripteurs RDKit: {result['rdkit_descriptors']}")
        
        if result.get("prediction_result"):
            print(f"Prédiction: {result['prediction_result']}")
        
        if result.get("suggestions"):
            print(f"Suggestions: {result['suggestions']}")
        
        if result.get("errors"):
            print(f"Erreurs: {result['errors']}")
        
                # Afficher la trace complète des messages (traçabilité)
        messages = result.get("messages", [])
        if messages:
            print(f"\n--- Trace complète ({len(messages)} messages) ---")
            for i, msg in enumerate(messages):
                role = getattr(msg, 'type', msg.__class__.__name__)
                content = getattr(msg, 'content', str(msg))
                
                # Tronquer les contenus trop longs
                if isinstance(content, str) and len(content) > 400:
                    content = content[:400] + "... [tronqué]"
                
                print(f"\n[{i}] {role}: {content}")
                
                # Afficher les appels d'outils si présents
                tool_calls = getattr(msg, 'tool_calls', None)
                if tool_calls:
                    for tc in tool_calls:
                        name = tc.get('name', '?') if isinstance(tc, dict) else getattr(tc, 'name', '?')
                        args = tc.get('args', {}) if isinstance(tc, dict) else getattr(tc, 'args', {})
                        print(f"    → Outil appelé: {name}")
                        print(f"      Arguments: {args}")
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\nERREUR après {elapsed:.1f}s: {type(e).__name__} - {str(e)}")
        return {"error": str(e)}


def main():
    """Fonction principale."""
    print("="*60)
    print("SYSTÈME MULTI-AGENTS POUR L'ANALYSE MOLÉCULAIRE")
    print("Architecture: LangGraph Supervisor")
    print("="*60)
    
    # Vérifier qu'Ollama est accessible
    print("\nVérification de la connexion Ollama...")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m['name'] for m in response.json().get("models", [])]
        print(f"Modèles disponibles: {models}")
    except Exception as e:
        print(f"AVERTISSEMENT: Ollama non accessible - {e}")
        return
    
    # Créer le système
    llm = create_llm()
    app = create_chem_supervisor(llm)
    
    # Choix du test à exécuter via argument de ligne de commande
    # Usage: python main.py 1   → exécute seulement le test 1
    #        python main.py all → exécute tous les tests
    test_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    tests = {
        "1": ("Analyse la molécule 'aspirin' et donne-moi ses propriétés PubChem.",
              "TEST 1 - Récupération PubChem par nom"),
        "2": ("Calcule les descripteurs RDKit pour le SMILES: CC(=O)OC1=CC=CC=C1C(=O)O",
              "TEST 2 - Descripteurs RDKit par SMILES"),
        "3": ("Prédis le LogP pour le SMILES: CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
              "TEST 3 - Prédiction LogP (caféine)"),
        "4": ("Prédis la solubilité pour le SMILES: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
              "TEST 4 - Prédiction solubilité (ibuprofène)"),
        "5": ("Calcule les descripteurs pour le SMILES: C1=CC=CC",
              "TEST 5 - Robustesse (SMILES invalide)"),
        "6": ("Analyse la molécule 'xyzabc123' dans PubChem.",
              "TEST 6 - Robustesse (molécule inexistante)"),
    }
    
    if test_arg == "all":
        for key in sorted(tests.keys()):
            query, name = tests[key]
            run_single_test(app, query, name)
    elif test_arg in tests:
        query, name = tests[test_arg]
        run_single_test(app, query, name)
    else:
        print(f"Test inconnu: {test_arg}. Utilise 1-6 ou 'all'.")
        return
    
    print("\n" + "="*60)
    print("FIN DES TESTS")
    print("="*60)


if __name__ == "__main__":
    main()
