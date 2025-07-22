# ⚡ YOLO MLOps Training & Benchmarking Framework
**Complete Pipeline for Training, Tracking, and Optimizing YOLO Object Detection Models**

<div align="center">

[![MLOps](https://img.shields.io/badge/MLOps-Experiment_Tracking-purple?style=for-the-badge)](https://mlflow.org/)
[![YOLO](https://img.shields.io/badge/YOLO-v8_&_v11-blue?style=for-the-badge)](https://ultralytics.com/)
[![Benchmarking](https://img.shields.io/badge/Benchmarking-Performance_Analysis-green?style=for-the-badge)](https://onnx.ai/)
[![Colab](https://img.shields.io/badge/Google-Colab_Ready-orange?style=for-the-badge)](https://colab.research.google.com/)

*End-to-end workflow for training YOLO models, tracking experiments with MLflow, and benchmarking deployment formats*

**🎯 Key Finding: ONNX format provides optimal balance of speed and memory efficiency with 97.7% memory reduction**

</div>

---

## 🎯 **Project Overview**

This framework provides a systematic approach to YOLO model development, from training to deployment optimization. Built for Google Colab, it streamlines the entire ML workflow while providing rigorous performance analysis to guide deployment decisions.

### **What This Framework Provides**
- **Unified training pipeline** for YOLOv8n and YOLOv11n architectures
- **Automated experiment tracking** with MLflow integration
- **Multi-format model export** (PyTorch, ONNX, TensorRT, TFLite)
- **Comprehensive benchmarking** to compare deployment performance
- **Reproducible workflows** with version control and artifact management

---

## 🏗️ **Architecture & Workflow**

### **Complete ML Pipeline**
```
📊 Roboflow Dataset → 🤖 YOLO Training → 📈 MLflow Logging → 
🔧 Model Export → ⚡ Performance Benchmarking → 📋 Deployment Recommendation
```

### **Key Components**
- **Training Notebooks**: Separate pipelines for YOLOv8n and YOLOv11n
- **MLflow Integration**: Automatic logging of metrics, parameters, and artifacts
- **Model Export**: Convert trained models to optimized inference formats
- **Benchmarking Module**: Systematic performance comparison across formats

---

## 📊 **Live Experiment Tracking**

**🔗 [View MLflow Experiments on DagsHub](https://dagshub.com/erwincarlogonzales/yolo-object-counter-mlflow.mlflow/#/experiments/10)**

All training runs, metrics, and model artifacts are tracked and accessible through the MLflow dashboard.

---

## 📈 **Training Results**

### **Model Performance**
The YOLOv8n model achieved **98.8% mAP@0.5** with strong convergence patterns:

![Training and Validation Curves](training_results/results.png)
*Training and validation curves showing consistent learning without overfitting*

![Confusion Matrix](training_results/confusion_matrix.png)
*Confusion matrix demonstrating high classification accuracy across object classes*

### **Performance Analysis**
![Precision-Recall Curve](training_results/PR_curve.png)
*Precision-recall curves achieving 98.8% mAP@0.5*

![F1-Confidence Curve](training_results/F1_curve.png)
*F1-score optimization showing peak performance at 0.490 confidence threshold*

### **Detailed Metrics**
<div align="center">

![Precision-Confidence Curve](training_results/P_curve.png) ![Recall-Confidence Curve](training_results/R_curve.png)

*Precision and recall performance across different confidence thresholds*

</div>

![Validation Batch 0 Predictions](training_results/val_batch0_labels.jpg)
*Model predictions on validation data showing accurate detection and classification*

---

## ⚡ **Benchmarking Results**

### **Performance Comparison**

| Model Format | Latency (ms) | Throughput (FPS) | Memory (MB) | Use Case |
|--------------|--------------|------------------|-------------|----------|
| PyTorch | 262.82 | 3.80 | 180.47 | Training/Research |
| **ONNX** | **225.23** | **4.44** | **4.05** | **Production** |
| TFLite | 207.22 | 4.83 | 8.54 | Mobile/Edge |

### **Key Findings**
- **ONNX format recommended for production**: Best balance of speed (4.44 FPS) and memory efficiency (97.7% reduction vs PyTorch)
- **TFLite fastest for single inference**: Lowest latency but higher memory usage than ONNX
- **PyTorch baseline**: Acceptable for research but inefficient for deployment

---

## 🚀 **Getting Started**

### **Prerequisites**
Configure these secrets in Google Colab:
```
GITHUB_TOKEN - Personal access token for repository access
ROBOFLOW_API_KEY - API key for dataset downloads  
MLFLOW_TRACKING_USERNAME - DagsHub username
MLFLOW_TRACKING_PASSWORD - DagsHub access token
```

### **Usage**
**1. Training:**
- Open either `YOLO_..._YOLOv8n.ipynb` or `YOLO_..._YOLOv11n.ipynb`
- Run all cells to execute complete training and export pipeline

**2. Benchmarking:**
- Ensure exported models are in `/models` directory
- Open `model_benchmarking.ipynb`
- Run all cells to generate performance analysis

**3. Results:**
- Check MLflow dashboard for experiment tracking
- Review `/benchmark_results_[timestamp]/` for detailed metrics

---

## 📁 **Repository Structure**

```
├── YOLO_Detection_Counting_MLflow_Experiments_YOLOv8n.ipynb
├── YOLO_Detection_Counting_MLflow_Experiments_YOLOv11n.ipynb
├── model_benchmarking.ipynb
├── training_results/
│   ├── results.png
│   ├── confusion_matrix.png
│   ├── PR_curve.png
│   ├── F1_curve.png
│   ├── P_curve.png
│   ├── R_curve.png
│   └── val_batch0_labels.jpg
├── models/
│   ├── yolov8n_best_int8_dynamic.onnx
│   ├── yolov8n_float16.tflite
│   └── yolov8n_pytorch_float16.pt
├── benchmark_results_[timestamp]/
│   ├── results.csv
│   └── results.json
└── README.md
```

---

## 🔧 **Features**

### **Training Pipeline**
- **Multi-architecture support** for YOLOv8n and YOLOv11n
- **Automated dataset integration** from Roboflow
- **Comprehensive metric logging** (mAP, precision, recall)
- **Visual artifact generation** (confusion matrices, curves)

### **Model Export**
- **Multiple format support**: ONNX (FP32/FP16/INT8), TensorRT, TFLite
- **Quantization options** for memory-constrained environments
- **Automated export pipeline** with error handling

### **Performance Benchmarking**
- **Latency measurement** for inference timing
- **Memory footprint analysis** for deployment planning
- **Throughput evaluation** for scalability assessment
- **Comparative analysis** across all export formats

---

## 🔮 **Known Issues & Future Work**

### **Current Limitations**
- **ONNX dynamic batching**: INT8 quantized models fail with batch size > 1
- **TensorRT export**: INT8 engine creation fails in Colab environment
- **Limited to single GPU**: Multi-GPU training not implemented

### **Potential Improvements**
- Fix ONNX model export for dynamic input shapes
- Investigate TensorRT INT8 calibration issues
- Add support for custom dataset formats beyond Roboflow
- Implement automated hyperparameter tuning

---

## 💡 **Why This Matters**

This framework addresses common challenges in ML model deployment:

**Problem**: Teams often deploy models without systematic performance analysis  
**Solution**: Comprehensive benchmarking provides data-driven deployment decisions

**Problem**: Inconsistent training and tracking across different model architectures  
**Solution**: Unified pipeline with automated experiment logging

**Problem**: Manual model optimization and format conversion  
**Solution**: Automated export to multiple optimized formats with performance comparison

---

## 📋 **Technical Stack**

- **Training**: YOLOv8n/YOLOv11n with Ultralytics
- **Experiment Tracking**: MLflow with DagsHub hosting
- **Model Export**: ONNX, TensorRT, TensorFlow Lite
- **Benchmarking**: Custom Python scripts with performance profiling
- **Environment**: Google Colab with GPU acceleration

---

<div align="center">

**🔬 Systematic • 📊 Data-Driven • 🚀 Practical**

*Building reliable ML workflows with performance validation*

</div>