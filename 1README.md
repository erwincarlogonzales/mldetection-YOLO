# 🔬 A Framework for Training, Tracking, and Benchmarking Custom YOLO Models
**An End-to-End MLOps Workflow for Reproducible Object Detection Research**

<div align="center">

[![MLOps](https://img.shields.io/badge/Methodology-MLOps_Workflow-purple?style=for-the-badge)](https://mlflow.org/)
[![YOLO](https://img.shields.io/badge/Architecture-YOLOv8_&_v11-blue?style=for-the-badge)](https://ultralytics.com/)
[![Performance](https://img.shields.io/badge/Performance-97.7%25_Memory_Reduction-green?style=for-the-badge)](https://onnx.ai/)
[![Deployment](https://img.shields.io/badge/Deployment-ONNX_&_TensorRT-orange?style=for-the-badge)](https://onnx.ai/)

*A complete MLOps workflow from dataset ingestion to optimized deployment with comprehensive benchmarking and experiment tracking.*

**⚡ KEY FINDING: The ONNX model format provides the best balance of performance and efficiency, achieving a 97.7% memory reduction compared to PyTorch while maintaining a 4.44 FPS throughput.**

</div>

---

## 🎯 **Framework Overview**

Developing and deploying machine learning models often involves challenges with reproducibility, performance comparison, and experiment management. This project addresses these issues by providing a systematic MLOps pipeline that covers the entire model lifecycle. The framework is designed to:

* Standardize the training process for multiple YOLO architectures
* Provide a rigorous, data-driven method for comparing and selecting a deployment format
* Automate experiment tracking to ensure all results, parameters, and artifacts are logged and reproducible
* Base deployment decisions on empirical evidence rather than guesswork

---

## 🏗️ **MLOps Architecture**

The workflow is a sequential pipeline that ensures a reproducible and well-documented path from data to a deployment recommendation.

### **End-to-End Workflow**
```
📊 Dataset (Roboflow) → 🤖 Multi-YOLO Training → 📈 MLflow Tracking →
🔧 Model Export → ⚡ Performance Benchmarking → 🚀 Deployment Recommendation
```

### **Core Components**

#### **1. Multi-Architecture Training Pipeline**
* Supports both **YOLOv8n & YOLOv11n** architectures within a unified interface
* Automates the logging of hyperparameters and performance metrics to MLflow
* Generates and stores visual artifacts like confusion matrices and PR curves

#### **2. Centralized Experiment Tracking**
* Integrates with **MLflow** for comprehensive experiment management
* Automatically stores all artifacts, including models, plots, and logs
* Uses the **MLflow Model Registry** for model versioning and promotion

#### **3. Model Optimization & Export**
* Exports models to multiple formats: PyTorch, **ONNX** (FP32, FP16, INT8), **TensorRT**, and **TFLite**
* Includes dynamic quantization for memory optimization

#### **4. Empirical Performance Benchmarking**
* Measures and compares **latency, throughput, and memory consumption** for each model format
* Provides quantitative data to support the final deployment recommendation

---

## 📊 **Live Experiment Dashboard**

All training runs, metrics, and artifacts are tracked in real-time and are publicly accessible for review and collaboration.

**🔗 [View MLflow Experiments on DagsHub](https://dagshub.com/erwincarlogonzales/yolo-object-counter-mlflow.mlflow/#/experiments/10)**

---

## 🚀 **Empirical Analysis and Results**

### **Model Training Performance**
The YOLOv8n model was trained and validated, achieving a **mean Average Precision (mAP@0.5) of 0.988**. The training process showed consistent convergence without significant overfitting, and the final model demonstrated high true-positive rates across all classes.

![Training and Validation Curves](training_results/results.png)
*Figure 1: Training curves illustrating model convergence and validation performance.*

![Confusion Matrix](training_results/confusion_matrix.png)
*Figure 2: Confusion matrix showing high true-positive rates for most classes.*

### **Deployment Benchmarking**
A performance benchmark was conducted to compare the PyTorch, ONNX, and TFLite formats on key deployment metrics.

| Model Format | Latency (ms) | Throughput (FPS) | Memory (MB) |
|:---|:---:|:---:|:---:|
| PyTorch | 262.82 | 3.80 | 180.47 |
| **ONNX** | **225.23** | **4.44** | **4.05** |
| TFLite | 207.22 | 4.83 | 8.54 |

### **Discussion and Recommendation**
The results show a clear trade-off between different deployment formats.
* **TFLite** is the fastest, with the lowest latency (207.22 ms) and highest throughput (4.83 FPS)
* **ONNX** offers the most balanced profile. It is exceptionally memory-efficient (4.05 MB) while delivering strong performance (4.44 FPS)

Given its superior balance of speed and minimal memory footprint, the **ONNX model is the recommended format for production deployment**.

---

## 🛠️ **Implementation Guide**

### **Prerequisites**
Set the following as secrets in your Google Colab environment:
* `GITHUB_TOKEN`
* `ROBOFLOW_API_KEY`
* `MLFLOW_TRACKING_USERNAME`
* `MLFLOW_TRACKING_PASSWORD`

### **Running the Pipeline**
1. **Train a Model**: Execute either the `YOLO_..._YOLOv8n.ipynb` or `YOLO_..._YOLOv11n.ipynb` notebook to run the training and export pipeline
2. **Benchmark Performance**: Run the `model_benchmarking.ipynb` notebook to analyze the exported models in the `/models` directory. The results will be saved to a `benchmark_results_[timestamp]` directory

---

## 📝 **Known Issues**
* The dynamically quantized ONNX model fails during benchmark tests with batch sizes greater than one and should be exported with fully dynamic input axes for batch inference
* The conversion to a TensorRT INT8 engine failed within the Colab environment and requires further investigation