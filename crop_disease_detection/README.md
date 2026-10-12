# AI-Based Crop Disease Detection

Plant leaf disease classifier with an EfficientNet-B0 PyTorch model, FastAPI
backend, and SQLite-backed user/diagnosis history.

## Setup

Run these commands from this directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Set `SECRET_KEY` in `.env` to a unique random value of at least 32 characters.
For example, generate one with `python -c "import secrets; print(secrets.token_urlsafe(48))"`
and paste the result into `.env`. Production refuses missing or placeholder keys.

Users sign in with their registered email address and password. Passwords are
stored as bcrypt hashes; successful sign-in returns a bearer token.

## Dataset and training

The project uses seven tomato and potato classes from Kaggle's New Plant Diseases
Dataset. This downloads/prepares the `train/` classes into `ml/dataset/` and the
separate official `valid/` classes into `ml/validation_dataset/`:

```powershell
python Download/Download_dataset.py
```

Train the model (ImageNet weights are downloaded on the first run; training is
much faster with a CUDA-capable GPU), then evaluate only on Kaggle's independent
validation split:

```powershell
python ml/training/train_model.py --epochs 5 --batch-size 16
python ml/training/evaluate_model.py
```

The evaluator requires all seven independent validation folders and does not
fall back to a random holdout from training data. To test the backend and data
pipeline, run `python -m pytest`.

The current checkpoint scored **98.07% accuracy on 3,320 images** from Kaggle's
separate validation split. This is a held-out dataset result, not a guarantee of
accuracy on field photos. GitHub Actions runs the automated tests on pushes and
pull requests.

The trained checkpoint is written to `model/crop_disease_model.pth` and includes
class names, validation accuracy, and preprocessing metadata.

## Run the application

Start the API and frontend from this directory:

```powershell
python -m uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/` for the CropCare frontend or
`http://127.0.0.1:8000/docs` for interactive API documentation. Create an
account or sign in, then upload a tomato or potato leaf photo. The frontend
stores the bearer token in the browser and displays the user's recent diagnoses.

## Automated tests

Install test dependencies with `python -m pip install -r requirements-dev.txt`
and run `python -m pytest`. The tests use a temporary SQLite database and a
mock predictor for upload API cases; they do not alter the local database or
reload the model.

## Deploy to Render

The repository includes a `render.yaml` Blueprint for a paid, always-on web
service. To deploy:

1. Push the **contents of this `crop_disease_detection` folder** to a GitHub
   repository. Include `model/crop_disease_model.pth`; `.gitignore` excludes the
   training/validation datasets, virtual environments, local database, uploaded
   images, and `.env` secrets.
2. In Render, choose **New → Blueprint**, connect the GitHub repository, and
   select the branch containing `render.yaml`.
3. Review and create the service. The Blueprint pins Python, installs
   `requirements.txt`, starts Uvicorn on Render's `$PORT`, generates `SECRET_KEY`,
   and attaches a persistent disk for SQLite and uploaded images.
4. Open the service's `onrender.com` URL after the first deploy finishes.

The Blueprint uses Render's paid `1c-2g` plan so the web service does not sleep
when idle. Free web services sleep after 15 minutes without traffic and lose
local files on restarts, so they are unsuitable for uninterrupted availability
and persistent SQLite/uploads. The attached disk requires a single service
instance. Set a payment/spend limit in Render to control costs. Never commit
`.env`; Render generates the production secret automatically.

## Project layout

- `backend/`: FastAPI routes, authentication, persistence, image handling, and inference.
- `frontend/`: client application.
- `ml/`: PlantVillage preparation, training, and evaluation scripts.
- `model/`: runtime model checkpoint.
