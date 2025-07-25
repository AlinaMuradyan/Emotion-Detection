# 🧠 Emotion Detection

This project detects human emotions from facial images using Python and machine learning. It includes a trained model (`model.pkl`), an inference script (`face.py`), and a Jupyter notebook for experimentation.

---

## 📁 Project Structure

```
Emotion-Detection/
├── face.py                 # Script for running emotion detection
├── model.pkl               # Trained model for prediction
├── emotions/               # Folder for emotion mappings or resources
├── Untitled.ipynb          # Notebook for model training/testing
├── .ipynb_checkpoints/     # Auto-generated Jupyter checkpoints
└── venv/                   # Virtual environment (ignored)
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/AlinaMuradyan/Emotion-Detection.git
cd Emotion-Detection
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install required libraries manually

```bash
pip install opencv-python numpy pickle5
pip install pandas matplotlib scikit-learn
```

---

## 🎯 Usage

### Run emotion detection

```bash
python face.py
```

> Make sure your `model.pkl` and any required files are in the correct location.

### Or run the notebook

```bash
jupyter notebook Untitled.ipynb
```

---

## 🔒 .gitignore (Recommended)

Add a `.gitignore` file to avoid pushing unnecessary files:

```
venv/
.ipynb_checkpoints/
__pycache__/
*.pyc
*.pkl
```

---

## 📌 Notes

- `face.py`: Runs emotion detection using webcam
- `model.pkl`: Trained model binary file
- `Untitled.ipynb`: Jupyter notebook with training/testing code
- `emotions/`: Resources for training the ML model

---

## 📜 License

This project is for educational and experimental use.

---

## 🙋‍♀️ Maintained by

[Alina Muradyan](https://github.com/AlinaMuradyan)
