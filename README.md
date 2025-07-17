# A Framework for Training, Tracking, and Benchmarking Custom YOLO Object Detection Models

## Abstract

This document outlines a comprehensive framework for the development and evaluation of custom object detection models utilizing the YOLO (You Only Look Once) architecture. The project presents an integrated workflow implemented within a Google Colab environment, designed to facilitate reproducible research and streamlined deployment. The methodology encompasses the entire machine learning lifecycle, including dataset integration from Roboflow, model training for YOLOv8n and YOLOv11n variants, comprehensive experiment tracking using MLflow, and subsequent model optimization via export to high-performance inference formats such as ONNX and TensorRT. A key component of this framework is a dedicated benchmarking module for the empirical analysis of latency, throughput, and memory consumption of the exported model artifacts. The results of this analysis indicate that while the TensorFlow Lite format yields the lowest single-instance inference latency, the ONNX format provides the most advantageous balance of computational performance and memory efficiency, establishing it as the optimal candidate for deployment.

---

## 1. Methodological Framework

The project is structured as an end-to-end MLOps workflow, integrating several key components to ensure a systematic and reproducible approach to model development.

* **Multi-Architecture Support**: The framework is designed to train and evaluate multiple YOLO variants, specifically the YOLOv8n and YOLOv11n architectures.
* **Integrated Experiment Tracking**: All training sessions are logged using MLflow. This includes the automatic logging of hyperparameters, performance metrics (e.g., mean Average Precision (mAP), Precision, Recall), and visual artifacts such as confusion matrices and validation predictions.
* **Advanced Model Export**: The system provides functionality to convert the trained PyTorch models into several optimized formats suitable for inference, including ONNX (with FP32, FP16, and INT8 dynamic quantization) and NVIDIA TensorRT engines.
* **Empirical Performance Benchmarking**: A dedicated Jupyter Notebook (`model_benchmarking.ipynb`) is included for a quantitative comparison of the exported models. This module evaluates latency, throughput, and memory usage to inform deployment decisions.
* **Model Lifecycle Management**: The workflow incorporates the use of the MLflow Model Registry to version, manage, and promote models from experimentation to production stages.

---

## 2. Project Resources

* **Experiment Tracking Dashboard**: All experimental runs, including parameters, metrics, and artifacts, are centrally managed and can be reviewed at the following MLflow tracking URI:
    * [**View MLflow Experiments on DagsHub**](https://dagshub.com/erwincarlogonzales/yolo-object-counter-mlflow.mlflow/#/experiments/10)

---

## 3. Repository Structure

The repository contains the following key components, organized to separate concerns between training, benchmarking, and storage of artifacts.

* **`YOLO_..._YOLOv8n.ipynb`**: A Jupyter notebook containing the complete implementation of the training, export, and logging pipeline for the YOLOv8n architecture.
* **`YOLO_..._YOLOv11n.ipynb`**: A Jupyter notebook providing the identical pipeline, adapted for the YOLOv11n architecture.
* **`model_benchmarking.ipynb`**: A standalone notebook for conducting performance analysis of the various exported model formats.
* **`/models`**: A directory containing the exported model files (`.pt`, `.onnx`, `.tflite`) that serve as inputs for the benchmarking notebook.
* **`/benchmark_results_...`**: An output directory containing the results of the performance benchmarks in both `.csv` and `.json` formats.
* **`README.md`**: This document, providing a comprehensive overview of the project.
* **`LICENSE`**: The project's MIT License file.

---

## 4. Implementation Guide

To replicate the experiments and utilize the framework, the following steps should be performed in a Google Colab environment.

### 4.1. Prerequisite: Configuration of Environment Credentials

The framework requires access to external services for version control, dataset acquisition, and experiment tracking. These must be configured as secrets within the Google Colab environment to ensure security.

* `GITHUB_TOKEN`: A GitHub personal access token with repository access rights.
* `ROBOFLOW_API_KEY`: An API key from a Roboflow account for programmatic dataset downloads.
* `MLFLOW_TRACKING_USERNAME`: The associated username for the MLflow tracking server (e.g., DagsHub).
* `MLFLOW_TRACKING_PASSWORD`: The corresponding access token or password for the MLflow tracking server.

### 4.2. Execution of the Training and Export Pipeline

1.  Select either the YOLOv8n or YOLOv11n training notebook.
2.  Execute the notebook cells sequentially. The initial cells will configure the environment by cloning the repository and installing dependencies.
3.  The subsequent cells will execute the full pipeline: dataset download, model training, metric and artifact logging to MLflow, and exporting the final model to all specified formats.

### 4.3. Execution of the Performance Benchmarks

1.  Ensure that the desired models exported from the training pipeline are present in the `/models` directory.
2.  Open the `model_benchmarking.ipynb` notebook.
3.  In the configuration cell, verify that the `MODEL_PATHS` dictionary correctly points to the models to be evaluated.
4.  Execute the cells sequentially to run the benchmarks. The results, including quantitative data tables and visualizations, will be generated and saved to a new `benchmark_results_[timestamp]` directory.

---

## 5. Empirical Analysis and Results

A performance analysis was conducted on the exported YOLOv8n models to determine the optimal format for deployment. The results for single-instance inference are summarized below.

| Model Format | Mean Latency (ms) | Throughput (FPS) | Memory Footprint (MB) |
| :------------- | :------------------ | :----------------- | :---------------------- |
| PyTorch        | 262.82              | 3.80               | 180.47                  |
| ONNX           | 225.23              | 4.44               | 4.05                    |
| TFLite         | 207.22              | 4.83               | 8.54                    |

![model_benchmarks](benchmark_results_20250717_101239/benchmark_results.png)

### 5.1. Discussion of Results

The empirical data reveals a significant performance differential between the native PyTorch training format and the optimized inference formats. The TFLite model achieved the lowest latency, making it the fastest for single-image processing. However, the most notable finding lies in resource utilization. The ONNX model demonstrated exceptional memory efficiency, consuming only 4.05 MB of memory—a 97.7% reduction compared to the 180.47 MB required by the PyTorch model. This efficiency is paramount for deployment in resource-constrained environments.

Furthermore, an analysis of the PyTorch model's scalability showed that its throughput peaked at a batch size of 8, with performance degrading at a batch size of 16. This indicates the presence of a computational or memory-related bottleneck, further highlighting its unsuitability for optimized, high-volume inference tasks.

### 5.2. Recommendation

Based on this quantitative analysis, the **ONNX model is formally recommended for production deployment**. It provides a superior synthesis of high-speed performance and minimal memory resource consumption. This makes it an ideal candidate for a wide range of applications, from edge computing on devices with limited memory to scalable, cost-effective deployments in the cloud.

---

## 6. Future Work and Known Issues

For the sake of academic transparency and future research directions, the following limitations and areas for improvement are noted:

* **Resolution of ONNX Dynamic Batching Incompatibility**: The dynamically quantized ONNX model currently fails during benchmark tests with batch sizes greater than one. Future work should involve exporting the ONNX model with fully dynamic input axes to enable variable batch size inference.
* **Investigation of TensorRT INT8 Export Failures**: The conversion to a TensorRT INT8 engine consistently resulted in a session failure within the Colab environment. This necessitates further investigation into the INT8 calibration process and potential hardware or software incompatibilities.
* **Integration of TensorRT Benchmarking**: The benchmarking notebook should be extended to include performance tests for the exported TensorRT `.engine` files, providing a more complete comparison, especially for NVIDIA hardware environments.

---

## 7. License

This project is distributed under the terms of the MIT License. Please refer to the `LICENSE` file for full details.