# CrowdAgent: Multi-Agent Managed Multi-Source Annotation System

This repository contains the official implementation for the system demo paper, **"CrowdAgent: Multi-Agent Managed Multi-Source Annotation System"**.

Our demonstration video is available at [https://youtu.be/fcnpanaEUqo](https://youtu.be/fcnpanaEUqo).

**Note:** This project is currently under development. The codebase and documentation are subject to change.

------

## 📖 Overview

High-quality annotated data is a cornerstone of modern Natural Language Processing (NLP). While recent methods leverage diverse annotation sources—including Large Language Models (LLMs), Small Language Models (SLMs), and human experts—they often lack the holistic **process control** required to manage these sources dynamically.

Inspired by real-world crowdsourcing companies, **CrowdAgent** is a multi-agent system that provides end-to-end process control over the entire data annotation lifecycle. It integrates task assignment, data annotation, and quality/cost management into a unified framework. By implementing a novel methodology that rationally assigns tasks, CrowdAgent enables LLMs, SLMs, and human experts to work synergistically, demonstrating superior efficiency and accuracy across diverse multimodal classification tasks.

This repository provides the source code for both the backend agent system and the frontend user interface.

## ✨ Key Features

- **Multi-Agent Framework:** Utilizes specialized agents for Scheduling, Quality Assurance (QA), and Financing to automate workflow management.

- **Multi-Source Annotation:** Synergistically combines LLMs, SLMs, and human experts to leverage their respective strengths.

- **Dynamic Process Control:** Moves beyond static rules, enabling intelligent, real-time task scheduling based on cost, quality, and annotator performance.

- **Interactive User Interface:** A user-friendly platform for task configuration, real-time monitoring of agent interactions, and visualization of key metrics.
