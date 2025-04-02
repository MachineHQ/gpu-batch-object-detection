[![Machine](./docs/github-repo-banner.png)](https://machine.dev/)

Machine supercharges your GitHub Workflows with seamless GPU acceleration. Say goodbye to the tedious overhead of managing GPU runners and hello to streamlined efficiency. With Machine, developers and organizations can effortlessly scale their AI and machine learning projects, shifting focus from infrastructure headaches to innovation and speed.

# GPU-Accelerated Object Detection

This repository provides a complete, automated example of performing GPU-accelerated object detection using the DETR model (`facebook/detr-resnet-50`) and GitHub Actions powered by Machine.dev. It leverages Hugging Face Transformers to detect and annotate objects from images in the COCO2017 dataset.

---

### ✨ **Key Features**

- **⚡ GPU Acceleration:** Efficiently perform object detection using GPUs via Machine.dev.
- **📥 Seamless Data Integration:** Automatically fetch metadata and images from the `phiyodr/coco2017` dataset.
- **🔍 Automated Detection Pipeline:** Detect, annotate, and export results without manual intervention.
- **📂 Results in CSV:** Generate a structured CSV artifact with detection details for easy review.
- **🚀 Easy Deployment:** Use this repository as a GitHub template to quickly start your own GPU-accelerated workflows.

---

### 📁 **Repository Structure**

```
├── .github/workflows/
│   └── batch-object-detection.yml  # Workflow definition
└── object_detection.py             # Main detection script
```

---

### ▶️ **Getting Started**

#### 1. **Use This Repository as a Template**
Click the **Use this template** button at the top of this page to create your own copy.

#### 2. **Set Up GPU Runners**
Ensure your repository uses Machine GPU-powered runners. No additional configuration is required if you're already using Machine.dev.

#### 3. **Run the Workflow**
- Trigger via commit or manually using GitHub Actions (**workflow_dispatch**).
- The workflow downloads and processes the first 1000 images from COCO2017.

#### 4. **Review Outputs**
- Annotated images and `detections.csv` will be saved in the `detection_output/` directory.
- The CSV file is uploaded as a GitHub Actions artifact for convenient access.

The workflow itself is defined in `.github/workflows/batch-object-detection.yml` and is fairly routine:

```yaml
name: Batch Object Detection

on:
  workflow_dispatch:

jobs:
  detect_objects:
    name: Detect Objects
    runs-on:
      - machine
      - gpu=T4
      - tenancy=spot
      - cpu=4
      - ram=16
      - architecture=x64
    steps:

      - uses: actions/checkout@v4

      - name: Set up Python 3.10
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install Dependencies
        run: |
          pip install -r requirements.txt

      - name: Run Object Detection
        run: python3 object_detection.py

      - name: Upload Detection Results CSV
        uses: actions/upload-artifact@v4
        with:
          name: detection-results-csv
          path: detection_output/detections.csv
```

We have configured the detect_objects job to run on a Machine GPU runner with the desired specs.The workflow then installs dependencies, runs the object detection script, and uploads the resulting CSV as an artifact.

---

### 🔑 **Prerequisites**

- GitHub account
- Access to [Machine](https://machine.dev) GPU-powered runners

_No local installation necessary—all processes run directly within GitHub Actions._

---

### 📄 **License**

This repository is available under the [MIT License](LICENSE).

---

### 📌 **Notes**

- This repository is currently open for use as a template. While public forks are encouraged, we are not accepting Pull Requests at this time.

_For questions or concerns, please open an issue._
