import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
import matplotlib.pyplot as plt
import networkx as nx
import torch_geometric.utils
import nibabel as nib
import numpy as np
from skimage.segmentation import slic
from skimage.graph import rag_mean_color
from scipy.ndimage import center_of_mass
import glob
import os
import requests
import base64
from PIL import Image
import io
import tempfile
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
plt.ioff()

# ===================================================================
# PART 1: YOUR GNN MODEL DEFINITION (from training script)
# ===================================================================
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes=5):
        super(GCN, self).__init__()
        # Convolutions
        self.conv1 = GCNConv(num_node_features, 32)
        self.conv2 = GCNConv(32, 16)
        self.conv3 = GCNConv(16, num_classes)

        # Batch Normalization Layers
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.bn2 = torch.nn.BatchNorm1d(16)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Layer 3
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def forward_explainer(self, x, edge_index):
        """Forward method for GNNExplainer"""
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Layer 3
        x = self.conv3(x, edge_index)
        return x  # Return raw logits for explainer

class GCNWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, **kwargs):
        return self.model.forward_explainer(x, edge_index)

# ===================================================================
# PART 2: YOUR DATA PROCESSING FUNCTION (from data script)
# ===================================================================
def _split(n, train=0.6, val=0.2):
    """Helper for training, not used in inference but needed by process_fn"""
    idx = torch.randperm(n)
    tr = int(n * train)
    va = int(n * val)
    tm, vm, tsm = (torch.zeros(n, dtype=torch.bool) for _ in range(3))
    tm[idx[:tr]] = True
    vm[idx[tr:tr+va]] = True
    tsm[idx[tr+va:]] = True
    return tm, vm, tsm

def process_files_to_graph(flair_path, seg_path=None, is_healthy=False, tumor_threshold=0.15):
    """
    This is your exact data processing function.
    I've modified it to return the 'flair_data' as well for 2D slicing.
    """
    # ... (all print statements removed for a cleaner UI experience)
    y = None
    flair_img = nib.load(flair_path)
    flair_data = flair_img.get_fdata().astype(np.float32)

    eps = 1e-8
    flair_norm = (flair_data - flair_data.min()) / (flair_data.max() - flair_data.min() + eps)

    supervoxels = slic(
        flair_norm, n_segments=2000, compactness=0.1,
        start_label=1, channel_axis=None
    )
    
    graph = rag_mean_color(flair_norm, supervoxels)
    node_ids = sorted(graph.nodes())
    old2new = {old: new for new, old in enumerate(node_ids)}

    uniq = np.unique(supervoxels)
    cents = center_of_mass(np.ones_like(flair_norm), supervoxels, uniq)
    centroid_map = dict(zip(uniq, cents))

    feats, pos = [], []
    for nid in node_ids:
        nd = graph.nodes[nid]
        intensity = nd['mean color']
        if isinstance(intensity, (list, np.ndarray)):
            intensity = float(intensity[0])
        size = float(nd['pixel count'])
        c = centroid_map.get(nid, (0, 0, 0))
        feats.append([intensity, size, float(c[0]), float(c[1]), float(c[2])])
        pos.append(list(c))

    x = torch.tensor(feats, dtype=torch.float)
    pos_tensor = torch.tensor(pos, dtype=torch.float)

    edges = [[old2new[u], old2new[v]] for u, v in graph.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # --- Label generation (not needed for inference) ---
    if seg_path is None and not is_healthy:
        pass # This is the inference path, y will remain None

    data = Data(x=x, edge_index=edge_index, pos=pos_tensor)
    if y is not None:
        data.y = y
        train_mask, val_mask, test_mask = _split(len(node_ids))
        data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask

    # Return both the graph and the original image data
    return data, flair_data

# ===================================================================
# PART 3: YOUR GNN INFERENCE PIPELINE (modified from 'Part 4' script)
# ===================================================================
def get_gnn_pipeline_results(flair_path, model, device, process_fn):
    """
    Runs the GNN pipeline and returns a text summary and the 3D image.
    This replaces the GNNExplainer part with the text summary, which
    is more useful for LLaVA.
    """
    st.write("🧠 [GNN] Step 1/3: Preprocessing NIfTI to Graph...")
    try:
        data, flair_data_3d = process_fn(flair_path, seg_path=None, is_healthy=False)
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return None, None, None

    st.write("🧠 [GNN] Step 2/3: Running GNN model for segmentation...")
    with torch.no_grad():
        model.eval()
        data = data.to(device)
        out = model(data)
        all_predicted_labels = out.argmax(dim=1)

    # --- Generate GNN Explainer Results ---
    st.write("🧠 [GNN] Step 3/4: Running GNN Explainer...")
    
    # Find tumor nodes to explain
    tumor_nodes_idx = torch.where((all_predicted_labels > 0) & (all_predicted_labels != 3))[0]
    
    explanations = []
    if tumor_nodes_idx.numel() > 0:
        # Setup GNNExplainer with wrapper
        model_for_explain = GCNWrapper(model).to(device)
        
        gnnexplainer_algorithm = GNNExplainer(
            epochs=50,
            lr=0.01,
            edge_mask_type="object"
        )
        
        model_config = ModelConfig(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw"
        )
        
        explainer = Explainer(
            model=model_for_explain,
            algorithm=gnnexplainer_algorithm,
            explanation_type="phenomenon",
            model_config=model_config,
            edge_mask_type="object"
        )
        
        # Explain first few tumor nodes
        for node_idx in tumor_nodes_idx[:3]:  # Limit to 3 nodes
            try:
                explanation = explainer(
                    data.x,
                    data.edge_index,
                    index=node_idx.item(),
                    target=all_predicted_labels
                )
                
                graph_fig = generate_explanation_graph(
                    explanation, data, node_idx.item(), 
                    all_predicted_labels[node_idx].item(), all_predicted_labels
                )
                
                explanations.append({
                    'node_idx': node_idx.item(),
                    'prediction': all_predicted_labels[node_idx].item(),
                    'edge_mask': explanation.edge_mask,
                    'explanation': explanation,
                    'graph_fig': graph_fig
                })
            except Exception as e:
                st.warning(f"Could not explain node {node_idx}: {e}")
                continue
    else:
        st.info("No tumor nodes found to explain.")
    
    # --- Generate Classification Summary ---
    st.write("🧠 [GNN] Step 4/4: Generating text summary...")
    node_counts = all_predicted_labels.cpu().unique(return_counts=True)
    class_map = {0: 'Background/Healthy', 1: 'Necrotic Core',
                 2: 'Edema', 3: 'UNUSED', 4: 'Enhancing Tumor'}

    summary_lines = ["**GNN Classification Summary (by Supervoxel Count):**"]
    tumor_nodes = 0
    total_nodes = data.num_nodes

    for label, count in zip(node_counts[0], node_counts[1]):
        label_int = label.item()
        count_int = count.item()
        class_name = class_map.get(label_int, f'Class {label_int}')
        percentage = (count_int / total_nodes) * 100
        summary_line = f"- {class_name:<20}: {count_int} nodes ({percentage:.2f}%)"
        summary_lines.append(summary_line)
        if label_int in [1, 2, 4]:
            tumor_nodes += count_int
    
    summary_lines.append("---")
    summary_lines.append(f"**Total Tumor Nodes (1, 2, 4): {tumor_nodes} ({ (tumor_nodes / total_nodes) * 100:.2f}%)**")
    
    text_summary = "\n".join(summary_lines)
    return text_summary, flair_data_3d, explanations

# ===================================================================
# PART 4: IMAGE & LLAVA CHAT HELPERS
# ===================================================================
OLLAMA_API_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llava:7b-v1.6-vicuna-q2_K" # Or your specific LLaVA model

def get_2d_slice(flair_data_3d):
    """
    Takes a 3D NIfTI data array and returns a 2D PNG-ready slice.
    This extracts the middle slice on the Z-axis.
    """
    # Get the middle slice index
    z_slice_index = flair_data_3d.shape[2] // 2
    
    # Extract the 2D slice
    slice_2d = flair_data_3d[:, :, z_slice_index]
    
    # Normalize the slice to 0-255 for image display
    if slice_2d.max() > slice_2d.min():
        slice_norm = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min())
    else:
        slice_norm = slice_2d # Avoid division by zero if flat
        
    slice_uint8 = (slice_norm * 255).astype(np.uint8)
    
    # Rotate the image 90 degrees (common for MRI display)
    slice_rotated = np.rot90(slice_uint8)
    
    return slice_rotated

def convert_np_to_base64(np_array):
    """Converts a NumPy array (2D) to a Base64 string"""
    img = Image.fromarray(np_array)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def generate_explanation_graph(explanation, data, node_to_explain, pred_label, all_predicted_labels, threshold=0.3):
    """Generate graph visualization exactly as in xgnnex.txt"""
    try:
        edge_mask = explanation.edge_mask
        mask = edge_mask > threshold
        
        edges_to_draw = data.edge_index[:, mask]
        nodes_to_draw = torch.unique(edges_to_draw)
        
        if nodes_to_draw.numel() == 0:
            return None
            
        relabel_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(nodes_to_draw)}
        
        remapped_edge_index = torch.stack([
            torch.tensor([relabel_map[idx.item()] for idx in edges_to_draw[0]]),
            torch.tensor([relabel_map[idx.item()] for idx in edges_to_draw[1]])
        ])
        
        plot_data = Data(
            x=data.x[nodes_to_draw],
            edge_index=remapped_edge_index,
            y=all_predicted_labels[nodes_to_draw].cpu()
        )
        
        g = torch_geometric.utils.to_networkx(plot_data, node_attrs=['y'])
        
        cmap = plt.cm.Set1
        label_to_color_index = {0: 0, 1: 1, 2: 2, 4: 3}
        color_indices = [label_to_color_index.get(g.nodes[n]['y'], 0) for n in g.nodes()]
        
        edge_importances = edge_mask[mask].cpu().numpy()
        widths = [w * 5 for w in edge_importances]
        
        explained_node_new_index = relabel_map.get(node_to_explain, -1)
        node_sizes = [600 if n == explained_node_new_index else 200 for n in g.nodes()]
        
        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(g)
        
        nx.draw_networkx(
            g, pos=pos, node_color=color_indices, cmap=cmap,
            node_size=node_sizes, width=widths, with_labels=False,
            vmin=0, vmax=8
        )
        
        healthy_patch = plt.Line2D([0], [0], marker='o', color='w',
                                   label='Healthy (Label 0)',
                                   markerfacecolor=cmap(0), markersize=12)
        nc_patch = plt.Line2D([0], [0], marker='o', color='w',
                                 label='Necrotic Core (Label 1)',
                                 markerfacecolor=cmap(1), markersize=12)
        edema_patch = plt.Line2D([0], [0], marker='o', color='w',
                                 label='Edema (Label 2)',
                                 markerfacecolor=cmap(2), markersize=12)
        tumor_patch = plt.Line2D([0], [0], marker='o', color='w',
                                 label='Enhancing Tumor (Label 4)',
                                 markerfacecolor=cmap(3), markersize=12)

        plt.legend(handles=[healthy_patch, nc_patch, edema_patch, tumor_patch],
                   title="Node Labels", loc="best")
        
        if explained_node_new_index != -1:
            ax = plt.gca()
            target_pos = pos[explained_node_new_index]
            ax.annotate(
                f"Explained Node ({node_to_explain})",
                xy=target_pos,
                xytext=(target_pos[0] + 0.15, target_pos[1] + 0.15),
                textcoords='data',
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                fontsize=12, fontweight='bold'
            )
        
        plt.title(f"GNNExplainer - Node {node_to_explain} (Threshold: {threshold})")
        return plt.gcf()
        
    except Exception as e:
        st.error(f"Error generating graph: {e}")
        return None

def call_ollama(prompt, image_b64, gnn_summary):
    """
    Calls the Ollama LLaVA model with the prompt, image, and GNN summary.
    """
    
    # We create a "system prompt" to give LLaVA context
    system_prompt = f"""
    You are a helpful radiologist's assistant. You will be given a 2D slice from a 3D FLAIR MRI.
    Your task is to answer questions about this image.

    To help you, a GNN (Graph Neural Network) has already analyzed the 3D scan and produced the following segmentation summary.
    Use this summary as a primary source of truth, even if the 2D slice isn't perfectly clear.

    --- GNN SUMMARY ---
    {gnn_summary}
    --- END SUMMARY ---

    Now, please answer the user's question based on the image and this summary.
    """

    # Prepare messages for the chat API
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": prompt,
            "images": [image_b64]
        }
    ]

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False
            }
        )
        response.raise_for_status()
        ollama_response = response.json()
        
        return ollama_response['message']['content']

    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Please make sure Ollama is running (`ollama run llava...`)"
    except Exception as e:
        return f"Error communicating with Ollama: {e}"

# ===================================================================
# PART 5: MODEL LOADING
# ===================================================================


# ===================================================================
# PART 6: STREAMLIT UI
# ===================================================================

@st.cache_resource
def load_model_cached(model_path="best_model_state.pth"):
    """Cached model loading for Streamlit"""
    st.write("Loading GNN model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found! Please save your model as '{model_path}'")
        return None, None

    model = GCN(num_node_features=5, num_classes=5).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None, None
        
    model.eval()
    st.success(f"GNN Model loaded successfully on {device}.")
    return model, device

# Streamlit App
st.set_page_config(layout="wide", page_title="Radiologist's Assistant")
st.title("🧠 Radiologist's GNN Assistant")

# Load Model
model, device = load_model_cached()
if model is None:
    st.stop()
    
# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "gnn_results" not in st.session_state:
    st.session_state.gnn_results = None

# Main UI Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. GNN Analysis")
    st.info("Upload a 3D FLAIR MRI file (`.nii` or `.nii.gz`) to begin.")
    
    uploaded_file = st.file_uploader("Choose an MRI file", type=["nii", "nii.gz"])
    
    if uploaded_file is not None and st.button("Analyze MRI"):
        # Determine correct file extension
        file_extension = ".nii.gz" if uploaded_file.name.endswith(".nii.gz") else ".nii"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            with st.spinner("Running GNN Pipeline..."):
                text_summary, flair_data_3d, explanations = get_gnn_pipeline_results(
                    flair_path=tmp_path,
                    model=model,
                    device=device,
                    process_fn=process_files_to_graph
                )
            
            if text_summary and flair_data_3d is not None and explanations is not None:
                st.success("GNN Analysis Complete!")
                
                slice_2d_np = get_2d_slice(flair_data_3d)
                image_b64 = convert_np_to_base64(slice_2d_np)
                
                st.session_state.gnn_results = {
                    "summary": text_summary,
                    "image_np": slice_2d_np,
                    "image_b64": image_b64,
                    "explanations": explanations
                }
                st.session_state.messages = []

        except Exception as e:
            st.error(f"Error: {e}")
            
        finally:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # Display Results
    if st.session_state.gnn_results:
        results = st.session_state.gnn_results
        st.subheader("Analysis Results")
        
        st.image(results["image_np"], caption="2D MRI Slice", use_column_width=True)
        
        with st.expander("Show GNN Summary"):
            st.markdown(results["summary"])
            
        with st.expander("Show GNN Explainer Results"):
            if "explanations" in results and results["explanations"]:
                for i, exp in enumerate(results["explanations"]):
                    st.write(f"**Node {exp['node_idx']} (Predicted Class: {exp['prediction']})**")
                    st.write(f"Edge importance (mean): {exp['edge_mask'].mean():.4f}")
                    st.write(f"Edge importance (max): {exp['edge_mask'].max():.4f}")
                    st.write(f"Important edges: {(exp['edge_mask'] > 0.3).sum().item()}")
                    
                    if 'graph_fig' in exp and exp['graph_fig'] is not None:
                        st.pyplot(exp['graph_fig'])
                        plt.close(exp['graph_fig'])
                    
                    st.write("---")
            else:
                st.write("No explanations generated (no tumor nodes found).")

with col2:
    st.header("2. LLaVA Chat Assistant")
    
    if not st.session_state.gnn_results:
        st.info("Please upload and analyze an MRI in Step 1 to activate chat.")
    else:
        st.info("Ollama (LLaVA) must be running locally.")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask about this scan..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("LLaVA is thinking..."):
                    results = st.session_state.gnn_results
                    response = call_ollama(
                        prompt,
                        results["image_b64"],
                        results["summary"]
                    )
                    st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})

