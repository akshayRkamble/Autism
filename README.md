## Autism Spectrum Disorder (ASD) Prediction (Streamlit)

This project provides a **local Streamlit web app** to predict ASD using:
- **Questionnaire answers** (tabular features)
- **A facial image** (simple grayscale pixel features)

It also includes scripts to **preprocess data**, **split datasets**, and **train/save models**.

## Project structure (important files)

- **App**: `src/app.py`
- **Training (questionnaire)**: `src/train_questionnaire_models.py` → saves `questionnaire_rf.pkl`, `questionnaire_scaler.pkl` to project root
- **Training (image)**: `src/train_image_models.py` → saves `image_rf.pkl`, `image_scaler.pkl` (see note below)
- **Preprocess questionnaire**: `src/preprocess_questionnaire.py`
- **Preprocess images → features**: `src/preprocess_images.py` (creates `Image/train_image_features.npz`)
- **Split train/test**: `src/split_datasets.py` (creates `Autism Screening/questionnaire_train_test.npz` and `Image/image_train_test.npz`)

## Requirements

- Windows 10/11 (works on other OS too with small command changes)
- Python **3.10+**
- Recommended: create a virtual environment

## Setup (fresh machine)

Open **PowerShell** in the project folder (the folder that contains `src/`, `Image/`, `Autism Screening/`):

```powershell
cd F:\Autism
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once (then retry activate):

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Option A (recommended): Run the app using pre-trained models

If the project root already contains these files, you can run the app immediately:
- `questionnaire_rf.pkl`
- `questionnaire_scaler.pkl`
- `image_rf.pkl`
- `image_scaler.pkl`

Run:

```powershell
cd F:\Autism
.\.venv\Scripts\Activate.ps1
streamlit run .\src\app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Option B: Train models locally (from the included data)

### 1) (Optional) Preprocess + split datasets

If you want to regenerate the `.npz` train/test split files:

```powershell
cd F:\Autism
.\.venv\Scripts\Activate.ps1
python .\src\preprocess_questionnaire.py
python .\src\preprocess_images.py
python .\src\split_datasets.py
```

Expected outputs:
- `Autism Screening\Autism_Data_processed.csv`
- `Autism Screening\questionnaire_train_test.npz`
- `Image\train_image_features.npz`
- `Image\image_train_test.npz`

### 2) Train questionnaire model

```powershell
cd F:\Autism
.\.venv\Scripts\Activate.ps1
python .\src\train_questionnaire_models.py
```

This should create (or overwrite) in the project root:
- `questionnaire_rf.pkl`
- `questionnaire_scaler.pkl`

### 3) Train image model

```powershell
cd F:\Autism
.\.venv\Scripts\Activate.ps1
python .\src\train_image_models.py
```

Note: the current `src/train_image_models.py` file contains duplicated/nested code blocks; depending on your Python execution path, it may not always save `image_rf.pkl`/`image_scaler.pkl` to the project root.  
If the files get created inside `src/`, run:

```powershell
python .\src\move_image_models.py
```

## Troubleshooting

### “can’t open file 'F:\\src' …”

Run scripts using the correct path from the project root:

```powershell
python .\src\train_image_models.py
```

### Streamlit starts but prediction fails (missing model files)

Make sure these exist in the **project root** (same folder as `src/`):
- `questionnaire_rf.pkl`, `questionnaire_scaler.pkl`
- `image_rf.pkl`, `image_scaler.pkl`

If they don’t exist, run the training steps in **Option B**.
