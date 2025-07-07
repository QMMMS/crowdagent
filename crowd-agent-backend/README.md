# CrowdAgent Backend

This directory contains the source code for the backend of the **CrowdAgent** system. This core logic, built with [LangGraph](https://www.langchain.com/langgraph), implements the multi-agent framework responsible for managing the entire data annotation lifecycle.

The backend orchestrates the interactions between specialized agents—including Scheduling, Quality Assurance (QA), and Financing—to dynamically manage a diverse workforce of LLMs, SLMs, and human annotators.

---

## 🚀 Setup

Follow these steps to prepare the backend environment.

### 1. Prerequisites

Before running the system, ensure the following requirements are met.

**Step 1: Hardware**

A **GPU is required** for the Small Language Model (SLM) annotators to function correctly and efficiently.

**Step 2: Complete Configuration**

Fill in the necessary `TODO` information within the source code.

**Step 3: Download SLM Weights**

The SLM annotators rely on pre-trained models from Hugging Face.

1. Refer to the `files/download_huggingface_model.ipynb` notebook for a guided download process.

2. Download the necessary model weights and place them in the `src/slm_function/cached_models/` directory. The required models are:
   
   - `models--facebook--convnextv2-tiny-22k-384`
   
   - `models--google-bert--bert-base-uncased`
   
   - `models--roberta-base`

**Step 4: Install Custom Dependencies**

The system uses a custom `crowdlib` package. Install it manually using pip:

```Bash
pip install files/crowdlib-0.7.0-py3-none-any.whl
```

> **⚠️ Important Notice:** The `crowdlib` package is protected. Use in any other project is strictly prohibited.

### 2. How to Run

The backend is not a standalone, continuously running server. It is **triggered by the frontend**.

To start a new annotation task, you must:

1. First, deploy and run the **frontend application**.

2. Navigate to the **Task Configuration** page in the UI.

3. Configure all required parameters for your task (e.g., budget, dataset, annotator selection).

4. Click the **"Submit"** button to initiate the backend process.

---

## ✨ System Architecture

The backend operates as a decentralized multi-agent system that simulates a virtual crowdsourcing company. Each agent has a distinct role and communicates with others to optimize for annotation quality, cost, and efficiency.

- **Annotation Agents**: These are the operational units that perform the labeling.
  
  - **LLM Annotators**: Generate large-scale initial labels using diverse prompting strategies.
  
  - **SLM Annotators**: Serve as efficient label purifiers, trained to distill true labels from noisy data.
  
  - **Human Annotators**: Provide expert judgment on the most challenging or ambiguous samples.

- **Quality Assurance (QA) Agent**: This agent is responsible for quality control. It aggregates labels from multiple sources using Bayesian inference, models the capabilities of each annotator, and generates feedback to improve future annotation rounds.

- **Financing Agent**: Tracks all project costs in real-time, from API calls and GPU usage to human labor. It provides continuous cost-effectiveness analysis to inform scheduling decisions.

- **Scheduling Agent**: Acts as the central coordinator. It dynamically assigns tasks to the most suitable annotator by synthesizing data from the QA and Financing agents, annotator profiles, and overall project goals.
