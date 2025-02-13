import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ollama
import gradio as gr
from sklearn.cluster import KMeans
from sklearn.utils import parallel_backend

# Set environment variable to avoid joblib CPU count issue
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Load Data
def load_data(file_path):
    return pd.read_csv(file_path)

# Determine Numeric Features
def get_numeric_features(df):
    return df.select_dtypes(include=['number']).columns.tolist()

# Cluster Data
def cluster_data(df, n_clusters=3):
    if n_clusters < 1:
        return df, "Error: Number of clusters must be at least 1."
    
    features = get_numeric_features(df)
    df = df.dropna(subset=features)  # Drop rows with missing values in key features
    if len(features) < 2:
        return df, "Error: Not enough numerical features for clustering."
    
    with parallel_backend('threading', n_jobs=1):  # Force joblib to use threading
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(df[features])
    
    return df, "Clustering completed."

# Generate Structured Business Analysis Summary using Ollama
def generate_summary(df):
    prompt = f"""
Perform a detailed business analysis on the following dataset summary:
{df.describe()}

Note: The dataset may not provide direct, immediately actionable insights without external context. However, it offers a foundation for understanding sales performance. Use the available data to extract potential trends, patterns, and recommendations. If direct insights are limited, suggest further investigation, additional data collection, or external comparisons.

Please produce the output in three parts in valid JSON format with the following keys:
- "reasoning": Provide the detailed internal reasoning and analysis process, including key trends and patterns identified.
- "graph_prompt": Describe an ideal graph or set of graphs that would best illustrate these trends and insights.
- "final_report": Summarize the insights in a clear, structured, and business-friendly report with actionable recommendations.

Ensure the output is in valid JSON format.
"""
    response = ollama.chat(model='mistral', messages=[{"role": "user", "content": prompt}])
    try:
        structured_output = json.loads(response['message']['content'])
    except Exception as e:
        structured_output = {
            "reasoning": "",
            "graph_prompt": "",
            "final_report": response['message']['content']
        }
    return structured_output

# Visualize Clusters
def plot_clusters(df):
    features = get_numeric_features(df)
    if len(features) < 2:
        return "Error: Not enough numeric features to visualize clusters."
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df[features[0]], y=df[features[1]], hue=df['cluster'], palette='viridis')
    plt.title("Cluster Visualization")
    plt.savefig("cluster_plot.png")
    return "cluster_plot.png"

# Answer Questions about Data
def answer_question(df, question):
    response = ollama.chat(model='mistral', messages=[{"role": "user", "content": f"Answer this question based on the dataset:\n{question}\nDataset Summary:\n{df.describe()}"}])
    return response['message']['content']

# Web UI with Gradio
def gradio_interface(file_path, question, n_clusters):
    try:
        df = load_data(file_path)
        df, cluster_message = cluster_data(df, max(1, int(n_clusters)))  # Ensure at least 1 cluster
        structured_summary = generate_summary(df)
        plot_path = plot_clusters(df)
        answer = answer_question(df, question)
    
        reasoning = structured_summary.get("reasoning", "")
        graph_prompt = structured_summary.get("graph_prompt", "")
        final_report = structured_summary.get("final_report", "")
    
        return cluster_message, reasoning, graph_prompt, final_report, plot_path, answer
    except Exception as e:
        return f"Error: {str(e)}", "", "", "", "", ""

iface = gr.Interface(
    fn=gradio_interface,
    inputs=[ "file", "text", "number" ],
    outputs=[ "text", "text", "text", "text", "image", "text" ],
    title="AI-Powered Data Analysis & Clustering",
    description=(
        "Upload your dataset, generate clusters, visualize them, and get AI-driven insights.\n\n"
        "The analysis is split into three parts:\n"
        "1. Model Reasoning\n"
        "2. Graph Prompt\n"
        "3. Final Business Report\n"
    )
)

if __name__ == "__main__":
    iface.launch()
