# 🧠 NeuroInsight : A GNN + VLM + XAI Framework for 3D MRI Analysis & QnA
*A Graph Neural Network + Multimodal AI System for Brain Tumor Segmentation and Interpretation*

---

## 📖 Overview

**NeuroInsight** is a deep learning web application that integrates a **Graph Neural Network (GNN)** for 3D MRI brain tumor segmentation with a **LLaVA Vision-Language model** for interactive interpretation.  
It transforms raw MRI scans (`.nii`/`.nii.gz`) into **graph-based tumor segmentation maps**, explains model predictions using **GNNExplainer**, and allows users to **chat with a multimodal AI assistant** about the scan.

---

## 🧩 Key Features

- ⚙️ **MRI to Graph Conversion** using supervoxel segmentation (SLIC) + Region Adjacency Graph (RAG)
- 🧠 **GNN-based segmentation** for identifying tumor subregions:
  - Background / Healthy tissue  
  - Necrotic Core  
  - Edema  
  - Enhancing Tumor
- 🔍 **Explainable AI** using GNNExplainer (visualizes important nodes/edges)
- 💬 **Interactive Chat** powered by **LLaVA** (via Ollama API) for radiology-style Q&A
- 🎛️ **Streamlit Frontend** for an intuitive and visual workflow

---

## 🧱 Architecture Overview

             
             ┌─────────────────────────────┐
             │      3D FLAIR MRI (.nii)    │
             └──────────────┬──────────────┘
                            │
                            ▼
               🧩 Preprocessing & Graph Building
               - Normalize MRI intensity
               - Apply SLIC → Supervoxels
               - Build RAG → Nodes + Edges
               - Extract node features:
                 [Intensity, Size, Centroid (x,y,z)]
                            │
                            ▼
                     🧠 Graph Neural Network
               (GCNConv Layers + BatchNorm + Dropout)
               - Learns relationships between regions
               - Classifies each node into tumor types
                            │
                            ▼
                      🔍 GNN Explainer
               - Highlights important nodes/edges
               - Generates interpretable subgraph
                            │
                            ▼
                 🧾 Classification Summary (by Node)
               - Background / Healthy %
               - Necrotic Core %
               - Edema %
               - Enhancing Tumor %
                            │
                            ▼
                  🖼️ 2D MRI Visualization
                            │
                            ▼
                 💬 LLaVA Vision-Language Chat
               - Image + GNN summary fed to LLaVA
               - Multimodal Q&A about the scan

---

## ⚙️ Model Architecture

###  **Graph Convolutional Network (GCN)**

| Layer | Type | Input → Output | Purpose |
|--------|------|----------------|----------|
| 1 | GCNConv | 5 → 32 | Extract local region features |
| 2 | GCNConv | 32 → 16 | Aggregate neighborhood context |
| 3 | GCNConv | 16 → 5 | Classify each node (0–4 classes) |

**Activation:** ReLU  
**Normalization:** BatchNorm1d  
**Regularization:** Dropout (p=0.5)  
**Output:** `log_softmax` per node

### 🧩 Classes

| Label | Meaning | Description |
|--------|----------|--------------|
| 0 | Background / Healthy | Normal brain tissue |
| 1 | Necrotic Core | Dead tissue in tumor center |
| 2 | Edema | Peritumoral swelling |
| 4 | Enhancing Tumor | Active proliferating tumor |

---

## 🔬 Data Flow

1. **Input:** `.nii` MRI file  
2. **Processing:**
   - Convert MRI volume → normalized array  
   - Generate supervoxels (`slic`)  
   - Build adjacency graph (`rag_mean_color`)  
   - Create PyTorch Geometric `Data(x, edge_index, pos)`  
3. **Inference:**  
   - Model predicts per-node tumor class  
   - GNNExplainer computes feature/edge importance  
4. **Output:**  
   - Node-level segmentation summary  
   - Explanation graph  
   - 2D slice preview  
5. **LLaVA Integration:**  
   - GNN summary + image → LLaVA via Ollama API  
   - User can query scan in natural language  

---

## 📊 Evaluation Metrics

| Metric | Formula | Description |
|---------|----------|--------------|
| **Accuracy** | (TP + TN) / (All) | Fraction of correctly classified nodes |
| **Dice Score (F1)** | 2TP / (2TP + FP + FN) | Measures overlap between predicted & actual tumor |
| **IoU (Jaccard)** | TP / (TP + FP + FN) | Measures intersection-over-union |
| **Precision / Recall** | Standard metrics per class | Identify sensitivity vs specificity |

---

## 💬 Multimodal Integration (LLaVA via Ollama)

After segmentation:
- A **2D MRI slice** and the **GNN summary text** are encoded and sent to **LLaVA**.
- The assistant answers user queries such as:
  - “Where is the tumor located?”
  - “Is it enhancing or necrotic?”
  - “What percentage of tissue is affected?”

This allows **multimodal reasoning** — image + text understanding, similar to an AI radiologist’s assistant.

---

## 🖥️ Streamlit Web App

### 🎛️ User Interface
| Section | Function |
|----------|-----------|
| **1. GNN Analysis** | Upload MRI, run segmentation, view GNN summary & explainer graphs |
| **2. LLaVA Chat** | Chat with the multimodal assistant about scan results |

---

# Built for the intersection of medical imaging and graph intelligence — combining neuroscience, deep learning, and natural language reasoning.
