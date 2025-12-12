# MDTeamGPT 🏥

**A Self-Evolving Multi-Agent Framework for Medical Multi-Disciplinary Team (MDT) Consultations.**

MDTeamGPT leverages Large Language Models (LLMs) to simulate a full medical consultation team. It mitigates context collapse in long dialogues using a **Residual Discussion Structure** and employs a **Self-Evolving Mechanism** via dual knowledge bases (CorrectKB & ChainKB) to accumulate medical experience.

## ✨ Key Features

  * **👨‍⚕️ Multi-Role Specialists:** Dynamically assigns specialists (e.g., Cardiologist, Neurologist) based on the patient case via a Primary Care Triage agent.
  * **🧠 Context Engineering:** A **Lead Physician** agent synthesizes discussions into structured residual context (*Consistency, Conflict, Independence, Integration*) to reduce cognitive load.
  * **🔄 Self-Evolution:**
      * **CorrectKB:** Stores successful reasoning patterns.
      * **ChainKB:** Stores reflection and error analysis from incorrect diagnoses.
  * **🛡️ Safety & Ethics:** A dedicated reviewer ensures consensus convergence and output safety.
  * **🛠️ External Tools:** Integrated Web Search and PubMed for evidence-based grounding.

## 🚀 Quick Start

### 1\. Prerequisites

  * Python 3.8+
  * OpenAI API Key (or compatible endpoint like DashScope)

### 2\. Installation

```bash
git clone https://github.com/KaiChenNJ/MDTeamGPT.git
cd MDTeamGPT
pip install -r requirements.txt
```

### 3\. Run the System

```bash
streamlit run app.py
```

## 📂 Project Structure

  * **`app.py`**: Main Streamlit interface for case entry, configuration, and visualization.
  * **`workflow.py`**: LangGraph state machine defining the flow (Triage $\to$ Consultation $\to$ Safety Check).
  * **`agents.py`**: Definitions for all agent roles, prompts, and strictly formatted outputs.
  * **`knowledge_base.py`**: Manages vector storage (FAISS) for dual-memory experience retrieval.
  * **`tools.py`**: Search and PubMed tool integrations.

## 🎓 Training Mode

1.  Toggle **Training Mode** in the UI.
2.  Input the **Ground Truth** (Correct Answer).
3.  The system automatically grades the consultation:
      * **Correct:** Saves reasoning to `CorrectKB`.
      * **Incorrect:** Performs Chain-of-Thought reflection and saves to `ChainKB`.
