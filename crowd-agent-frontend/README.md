# CrowdAgent Frontend

This directory contains the source code for the frontend user interface of the **CrowdAgent** system, built with [Streamlit](https://streamlit.io/).

This interactive interface allows users to configure annotation tasks, monitor agent interactions in real-time, and visualize key project metrics.

---

## 🚀 Getting Started

Follow these steps to set up and run the frontend application locally.

### 1. Prerequisites

Before launching the application, please complete the following setup steps:

**Step 1: Complete Configuration**

Fill in the necessary `TODO` information within the source code. This typically includes API keys, server endpoints, or other configuration variables.

**Step 2: Download Example Assets (Optional)**

The system is pre-configured for a multi-modal four-class classification task. You can download the required image assets to run this default task.

- **Download:** `images.zip` from [this Google Drive link](https://drive.google.com/drive/folders/1DIk9pqm0Fl39mOuTjSuZ7LpaeXmNj_B9?usp=sharing).

- **Place** the zip file into the `/asserts/example_files/` directory.

**Step 3: Build the Agent Map Component**

The interface uses a custom component to visualize agent interactions. You need to build it from the source.

Navigate to the component's directory and run the build commands:

Bash

```
cd components/agent_map_component/frontend
npm install
npm run build
```

### 2. Running the Application

Once the prerequisites are met, you can launch the Streamlit application.

**Option A: Run Directly**

Execute the following command in your terminal from the `frontend` directory:

Bash

```
streamlit run home.py
```

**Option B: Run in the Background**

Use the provided bash script to run the application as a background process:

Bash

```
bash run.bash
```

This will start the server, and you can access the UI in your web browser, typically at `http://localhost:8501`.

---

## ✨ Features

The user interface is organized into several key pages, providing comprehensive control and insight into the annotation process.

### **📝 Task Configuration**

Define your annotation project by specifying the task type, setting the budget, and configuring the class labels. You can also select which annotation sources (LLMs, SLMs, Humans) to use and set up connections to third-party crowdsourcing platforms.

### **🤖 Agents Interaction**

Gain a transparent view into the multi-agent system's operations. This page displays real-time and historical interactions, including annotator progress, difficult sample analysis, automatically generated agent profiles, and the decision logic of the core agents.

### **📊 Annotation Details**

Track the annotation process round-by-round. This page presents critical statistics like deployed annotators, generated labels, round-specific accuracy, and costs. It also allows for manual management of human annotation tasks, including data download and results upload.

### **📈 Dashboard**

Access a centralized control panel that visualizes a complete overview of your project's progress. It features intuitive charts for key metrics like accuracy, estimated cost savings, and confidence distribution. You can also download the final, aggregated dataset directly from this page.
