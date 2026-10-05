"""Interface utilisateur Streamlit pour le système multi-agents chimique."""

import streamlit as st
import time
import json
from datetime import datetime

from agents import create_llm, create_chem_supervisor
from state import ChemAgentState

# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================

st.set_page_config(
    page_title="Multi-Agent Chemical Analyzer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLES CSS PERSONNALISÉS
# ============================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1F4E79;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #F0F8FF;
        border-left: 5px solid #1F4E79;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .audit-box {
        background-color: #FAFAFA;
        border: 1px solid #DDD;
        padding: 1rem;
        border-radius: 5px;
        font-family: monospace;
        font-size: 0.85rem;
        max-height: 500px;
        overflow-y: auto;
    }
    .metric-box {
        background-color: #FFFFFF;
        border: 2px solid #1F4E79;
        padding: 0.8rem;
        border-radius: 8px;
        text-align: center;
    }
    .success-tag {
        background-color: #70AD47;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .error-tag {
        background-color: #C00000;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# INITIALISATION DE L'ÉTAT DE SESSION
# ============================================================

if "app" not in st.session_state:
    with st.spinner("Initialisation du système multi-agents..."):
        try:
            llm = create_llm()
            st.session_state.app = create_chem_supervisor(llm)
            st.session_state.llm_ready = True
        except Exception as e:
            st.session_state.llm_ready = False
            st.session_state.init_error = str(e)

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# FONCTION D'EXÉCUTION DU WORKFLOW
# ============================================================

def run_analysis(query: str) -> dict:
    """Exécute le workflow multi-agents et retourne le résultat."""
    
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
        result = st.session_state.app.invoke(
            initial_state,
            config={"recursion_limit": 30}
        )
        elapsed = time.time() - start_time
        result["_elapsed"] = elapsed
        result["_success"] = True
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "_elapsed": elapsed,
            "_success": False,
            "_error": str(e),
            "messages": [],
        }


def extract_audit_trace(messages: list) -> list:
    """Extrait la trace d'audit des messages."""
    trace = []
    for i, msg in enumerate(messages):
        role = getattr(msg, 'type', msg.__class__.__name__)
        content = getattr(msg, 'content', str(msg))
        
        entry = {
            "index": i,
            "role": role,
            "content": content,
            "tool_calls": [],
        }
        
        # Extraire les appels d'outils
        tool_calls = getattr(msg, 'tool_calls', None)
        if tool_calls:
            for tc in tool_calls:
                if isinstance(tc, dict):
                    entry["tool_calls"].append({
                        "name": tc.get("name", "?"),
                        "args": tc.get("args", {}),
                    })
                else:
                    entry["tool_calls"].append({
                        "name": getattr(tc, "name", "?"),
                        "args": getattr(tc, "args", {}),
                    })
        
        trace.append(entry)
    
    return trace


# ============================================================
# EN-TÊTE
# ============================================================

st.markdown('<div class="main-header">🧪 Multi-Agent Chemical Analyzer</div>', 
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">LangGraph Supervisor · openai/gpt-oss-120b via Groq · RDKit · PubChem</div>', 
            unsafe_allow_html=True)

# Vérifier que le LLM est initialisé
if not st.session_state.llm_ready:
    st.error(f"❌ Erreur d'initialisation : {st.session_state.init_error}")
    st.stop()


# ============================================================
# BARRE LATÉRALE - PARAMÈTRES
# ============================================================

with st.sidebar:
    st.header("⚙️ Paramètres")
    
    input_mode = st.radio(
        "Mode d'entrée",
        ["Nom de molécule", "SMILES", "CID PubChem"],
        help="Choisissez comment identifier votre molécule"
    )
    
    target_property = st.selectbox(
        "Propriété cible",
        ["logp", "solubility", "permeability"],
        format_func=lambda x: {
            "logp": "LogP (lipophilicité)",
            "solubility": "Solubilité aqueuse",
            "permeability": "Perméabilité membranaire",
        }[x]
    )
    
    st.divider()
    
    st.markdown("### 📊 Statistiques de session")
    st.metric("Analyses effectuées", len(st.session_state.history))
    
    if st.session_state.history:
        avg_time = sum(h["elapsed"] for h in st.session_state.history) / len(st.session_state.history)
        st.metric("Temps moyen", f"{avg_time:.1f}s")
        
        success_rate = sum(1 for h in st.session_state.history if h["success"]) / len(st.session_state.history) * 100
        st.metric("Taux de réussite", f"{success_rate:.0f}%")
    
    st.divider()
    
    if st.button("🗑️ Effacer l'historique"):
        st.session_state.history = []
        st.rerun()


# ============================================================
# ZONE D'ENTRÉE
# ============================================================

st.markdown("### 📥 Entrée")

col1, col2 = st.columns([3, 1])

with col1:
    if input_mode == "Nom de molécule":
        user_input = st.text_input(
            "Nom de la molécule",
            value="aspirin",
            placeholder="Ex: aspirin, caffeine, ibuprofen...",
            key="input_name"
        )
    elif input_mode == "SMILES":
        user_input = st.text_input(
            "SMILES",
            value="CC(=O)OC1=CC=CC=C1C(=O)O",
            placeholder="Ex: CC(=O)OC1=CC=CC=C1C(=O)O",
            key="input_smiles"
        )
    else:  # CID
        user_input = st.text_input(
            "CID PubChem",
            value="2244",
            placeholder="Ex: 2244",
            key="input_cid"
        )

with col2:
    st.write("")
    st.write("")
    run_button = st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True)


# ============================================================
# CONSTRUCTION DE LA REQUÊTE
# ============================================================

def build_query(mode: str, value: str, prop: str) -> str:
    """Construit la requête à envoyer au superviseur."""
    prop_label = {
        "logp": "LogP",
        "solubility": "solubilité",
        "permeability": "perméabilité",
    }[prop]
    
    if mode == "Nom de molécule":
        return f"Analyse la molécule '{value}' et prédis sa {prop_label}."
    elif mode == "SMILES":
        return f"Prédis la {prop_label} pour le SMILES: {value}"
    else:
        return f"Analyse la molécule avec CID {value} et prédis sa {prop_label}."


# ============================================================
# EXÉCUTION
# ============================================================

if run_button and user_input:
    query = build_query(input_mode, user_input, target_property)
    
    st.markdown("### 🔄 Traitement en cours")
    
    with st.spinner(f"Exécution du workflow multi-agents pour : {user_input}"):
        result = run_analysis(query)
    
    # Sauvegarder dans l'historique
    st.session_state.history.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "input": user_input,
        "mode": input_mode,
        "property": target_property,
        "elapsed": result.get("_elapsed", 0),
        "success": result.get("_success", False),
    })
    
    # ============================================================
    # SORTIE PRINCIPALE
    # ============================================================
    
    st.markdown("### 📊 Résultat")
    
    if result.get("_success"):
        messages = result.get("messages", [])
        
        if messages:
            last_msg = messages[-1]
            content = getattr(last_msg, 'content', str(last_msg))
            
            # Métriques
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⏱️ Temps d'exécution", f"{result['_elapsed']:.1f}s")
            with col2:
                st.metric("📨 Messages échangés", len(messages))
            with col3:
                st.metric("✅ Statut", "Succès")
            
            # Résultat principal
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.markdown("#### Réponse finale du système")
            st.markdown(content)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error(f"❌ Erreur : {result.get('_error', 'Erreur inconnue')}")
    
    # ============================================================
    # AUDIT DU PROCESSUS
    # ============================================================
    
    st.markdown("### 🔍 Audit du processus")
    
    with st.expander("📜 Voir la trace complète des messages", expanded=True):
        if result.get("_success") and result.get("messages"):
            trace = extract_audit_trace(result["messages"])
            
            for entry in trace:
                role = entry["role"]
                content = entry["content"]
                
                # Tronquer les contenus longs
                if isinstance(content, str) and len(content) > 800:
                    content = content[:800] + "... [tronqué]"
                
                # Couleur selon le rôle
                if role == "human":
                    color = "🔵"
                elif role == "ai":
                    color = "🟢"
                elif role == "tool":
                    color = "🟡"
                else:
                    color = "⚪"
                
                st.markdown(f"**{color} [{entry['index']}] {role}**")
                st.markdown(f"```\n{content}\n```")
                
                # Afficher les appels d'outils
                if entry["tool_calls"]:
                    for tc in entry["tool_calls"]:
                        st.markdown(f"→ **Outil appelé :** `{tc['name']}`")
                        st.markdown(f"   Arguments : `{json.dumps(tc['args'], ensure_ascii=False)}`")
                
                st.divider()
        else:
            st.info("Aucune trace disponible.")

    # ============================================================
    # EXPORT
    # ============================================================
    
    st.markdown("### 💾 Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export JSON
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "input_mode": input_mode,
            "input_value": user_input,
            "target_property": target_property,
            "elapsed_seconds": result.get("_elapsed"),
            "success": result.get("_success"),
            "messages": [
                {
                    "role": getattr(m, 'type', 'unknown'),
                    "content": str(getattr(m, 'content', m)),
                }
                for m in result.get("messages", [])
            ],
        }
        
        st.download_button(
            label="📥 Télécharger le rapport (JSON)",
            data=json.dumps(export_data, indent=2, ensure_ascii=False),
            file_name=f"rapport_{user_input}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )
    
    with col2:
        # Export texte
        text_report = f"""RAPPORT D'ANALYSE MULTI-AGENTS
================================
Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Requête : {query}
Mode d'entrée : {input_mode}
Valeur : {user_input}
Propriété cible : {target_property}
Temps d'exécution : {result.get('_elapsed', 0):.1f}s
Statut : {'Succès' if result.get('_success') else 'Échec'}

RÉPONSE FINALE
--------------
{result.get('messages', [{}])[-1].content if result.get('messages') else 'N/A'}

TRACE COMPLÈTE
--------------
"""
        for entry in extract_audit_trace(result.get("messages", [])):
            text_report += f"\n[{entry['index']}] {entry['role']}:\n{entry['content']}\n"
        
        st.download_button(
            label="📄 Télécharger le rapport (TXT)",
            data=text_report,
            file_name=f"rapport_{user_input}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )


# ============================================================
# HISTORIQUE
# ============================================================

if st.session_state.history:
    st.markdown("### 📚 Historique de la session")
    
    for h in reversed(st.session_state.history[-10:]):
        status = "✅" if h["success"] else "❌"
        st.markdown(
            f"{status} **{h['timestamp']}** · "
            f"`{h['input']}` ({h['mode']}) · "
            f"{h['property']} · "
            f"{h['elapsed']:.1f}s"
        )
