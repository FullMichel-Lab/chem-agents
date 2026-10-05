# Multi-Agent Chemical Analyzer

A LangGraph-based multi-agent system for molecular property prediction and optimization.

## Overview

This project implements a Supervisor architecture in LangGraph where a central
routing agent orchestrates four specialized agents:

- etrieve_data : Fetches molecular properties from PubChem
- compute_descriptors : Computes RDKit molecular descriptors
- predict_property : Predicts physicochemical properties (LogP, solubility, permeability)
- suggest_optimization : Proposes structural modifications

## Architecture

The system uses openai/gpt-oss-120b via the Groq API for reasoning and tool calling.

## Installation

### 1. Clone the repository

\\\ash
git clone https://github.com/YOUR_USERNAME/chem-agents.git
cd chem-agents
\\\

### 2. Create a virtual environment

\\\ash
python -m venv venv
\\\

**Windows:**
\\\powershell
.\venv\Scripts\Activate.ps1
\\\

**Linux/macOS:**
\\\ash
source venv/bin/activate
\\\

### 3. Install dependencies

\\\ash
pip install -r requirements.txt
\\\

### 4. Configure environment variables

Copy .env.example to .env and fill in your Groq API key:

\\\
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=openai/gpt-oss-120b
\\\

Get a free API key at https://console.groq.com/keys

## Usage

### Command-line interface

\\\ash
python main.py all      # Run all tests
python main.py 5        # Run a specific test
\\\

### Web interface (Streamlit)

\\\ash
streamlit run app.py
\\\

Then open http://localhost:8501 in your browser.

## Features

- Multi-agent orchestration with LangGraph Supervisor
- Full audit trail of every agent interaction
- Robustness to invalid inputs (SMILES, non-existent molecules)
- Export of analysis reports in JSON and TXT
- Model-agnostic architecture

## Results

100% success rate on 6 functional and robustness tests.
Average execution time: 13.9 seconds per query.

## License

MIT License

## Citation

If you use this work, please cite:

[Your name]. (2026). Beyond the Tool: A LangGraph-Based Multi-Agent System
for Molecular Property Prediction and Optimization. Adv. Chemistry 2027.
