# TODO:
    - add benchmarking pytorch, onnx, and engine

# Custom Object Detection with YOLO (Ultralytics) 🚀

This project provides a streamlined Google Colab workflow for training custom object detection models using Ultralytics YOLO, integrating MLflow for experiment tracking, and exporting models to ONNX and TensorRT formats. Perfect for your dissertation project, bro! 🧑‍🎓

## 🎯 Goal

Train a custom object detection model for specific hardware components, track experiments with MLflow, and convert the trained model to ONNX and TensorRT for optimized deployment.

## ✨ Features

-   **Custom Object Detection**: Train YOLOv8n or YOLOv11n models on your own dataset.
-   **Google Colab Integration**: Optimized for Google Colab, making setup and execution a breeze.
-   **Roboflow Dataset Integration**: Easily download and prepare your dataset from Roboflow.
-   **MLflow Experiment Tracking**: Log all your training metrics, parameters, and artifacts to MLflow for comprehensive experiment management.
-   **Model Export**: Convert your trained YOLO models to various formats:
    -   .onnx (FP32, FP16, INT8 Dynamic Quantization)
    -   .engine (TensorRT FP16, TensorRT INT8 - *Note: TensorRT INT8 might require runtime restarts and specific hardware configurations, so keep an eye out for that!* 👀)
-   **Reusable Workflow**: Designed for easy adaptation to other object detection projects.

## 📦 Project Structure (Colab Notebook Sections)

The Colab notebooks (`YOLO_Detection_Counting_MLflow_Experiments_YOLOv8n.ipynb` and `YOLO_Detection_Counting_MLflow_Experiments_YOLOv11n.ipynb`) are structured into clear, logical sections:

### ⚙️ Setup & Dataset Preparation

-   Install necessary libraries: `ultralytics`, `roboflow`, `mlflow`, `onnx`, `onnxruntime`.
-   Clone the GitHub repository to your Colab environment.
-   Download your custom dataset from Roboflow in YOLO format.
-   Verify dataset configuration (e.g., `data.yaml` classes).

### 🚀 MLflow & Model Configuration

-   Configure MLflow tracking URI and credentials (e.g., for DagsHub).
-   Set up experiment name and model parameters (epochs, image size, batch size, patience).
-   Initialize the YOLO model (e.g., `yolov8n.pt` or `yolo11n.pt`).

### 📈 Training & MLflow Export

-   Execute the training loop for your YOLO model.
-   Automatically log key metrics (mAP@0.5-0.95, Precision, Recall) to MLflow.
-   Log essential training artifacts (confusion matrix, PR curves, results CSV) to MLflow.
-   Export the trained `.pt` model to:
    -   ONNX (FP32)
    -   ONNX (INT8 Dynamic Quantization)
    -   ONNX (FP16)
    -   TensorRT Engine (FP16)
    -   TensorRT Engine (INT8) - *This one can be tricky, as mentioned in the notebooks!*

## 🛠️ Usage

1.  **Open in Google Colab**: Click the "Open In Colab" badge in the respective notebook (`YOLO_Detection_Counting_MLflow_Experiments_YOLOv8n.ipynb` or `YOLO_Detection_Counting_MLflow_Experiments_YOLOv11n.ipynb`).
2.  **GitHub Setup**: Follow the initial steps to clone the repository and configure Git.
3.  **Roboflow API Key**: Ensure you have your Roboflow API key configured as a Colab secret for dataset download.
4.  **MLflow Credentials**: Set up your MLflow tracking URI and credentials (username/password) as Colab secrets to enable experiment logging.
5.  **Run Cells**: Execute the notebook cells sequentially. The code is designed to be lean and direct, so no complex error handling is in there. If something breaks, hit me up! 🤙
6.  **Review Results**: Check MLflow for detailed experiment logs and download your exported ONNX and TensorRT models for deployment.

## 🤝 Contribution

Feel free to fork this repo and adapt it to your needs! If you've got cool improvements or bug fixes, hit me up! 🚀

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](http://googleusercontent.com/AB781512/erwincarlogonzales/mldetection-yolo/erwincarlogonzales-mldetection-YOLO-73fd82a097cfcfff83b3876c47b3db758949fc75/LICENSE) for details.