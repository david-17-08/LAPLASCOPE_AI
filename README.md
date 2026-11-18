# LAPLASCOPE AI

**Periodic Signal Generation and Response Visualisation Using Python Based on Laplace Transform Techniques**

An interactive Streamlit engineering lab for periodic signals, Laplace transforms, continuous-time LTI response, pole-zero analysis, and frequency response.

## Features

- Sine, cosine, square, triangular, sawtooth, pulse-train, and custom Fourier-series waveforms.
- Forward causal, periodic-extension, and finite-duration Laplace transforms; inverse transform, partial fractions, and value-theorem limits.
- First- and second-order presets or user-entered proper transfer functions.
- Step, impulse, ramp, sinusoidal, and square-wave response simulation using SciPy.
- Start, pause, and reset controls for a decimated time playback of simulated input/output/impulse response.
- Interactive Plotly time plots, pole-zero maps, and frequency response.
- CSV, PNG, JSON parameter, and PDF mathematical-analysis exports.
- Restricted expression parser; arbitrary Python expressions are not evaluated.

## Windows setup (Python 3.11+)

In the VS Code terminal:

```powershell
cd LAPLASCOPE_AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local Streamlit address printed in the terminal. If PowerShell blocks activation, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Tests

```powershell
.\.venv\Scripts\Activate.ps1
pytest -q
```

Tests cover periodic waveform samples, sample-size guards, causal/periodic/finite transforms, inverse transforms, safe expression parsing, partial fractions, value theorems, symbolic-vs-numerical step response, impulse scaling, pole stability, frequency response, and PNG/PDF exports.

## Mathematical and numerical scope

For a one-sided periodic extension of period `T`, the transform is

`F(s) = integral_0^T f(t) exp(-s t) dt / (1 - exp(-s T))`.

The finite-duration option uses only `integral_0^T f(t) exp(-s t) dt`; it does not repeat the finite record. For bounded periodic signals, the periodic formula generally converges for `Re(s) > 0`, subject to the actual integral.

The system simulator assumes a zero-initial-condition, continuous-time, proper transfer function and uses SciPy numerical simulation. Rise/settling/overshoot/error metrics are estimated from a unit-step simulation across the selected time window. A too-short horizon can under-report settling time. Symbolic transforms may remain unevaluated when the entered expression has no simple closed form.

## Project layout

```text
LAPLASCOPE_AI/
├── app.py
├── modules/
├── visualization/
├── reports/
├── tests/
├── requirements.txt
└── README.md
```

## Deployment

### Streamlit Community Cloud

The project is prepared for Streamlit Community Cloud: `app.py` is the entry point, `requirements.txt` lists the Python dependencies, and `.streamlit/config.toml` contains the app theme and server settings. The local virtual environment uses Python 3.13.14; select Python 3.13 in the Cloud deployment settings so the hosted runtime matches.

1. Create a GitHub repository for this project and upload the contents of the `LAPLASCOPE_AI` folder. Choose repository visibility deliberately: a public repository makes the source code public.
2. From the VS Code terminal, in the `LAPLASCOPE_AI` folder, initialize and push the project (replace the URL with your repository URL):

   ```powershell
   git init
   git add .
   git commit -m "Prepare LAPLASCOPE AI for deployment"
   git branch -M main
   git remote add origin https://github.com/<your-account>/LAPLASCOPE_AI.git
   git push -u origin main
   ```

   If the folder is already a Git repository, skip `git init`. If `origin` is already configured, skip `git remote add origin`.
3. Sign in at [share.streamlit.io](https://share.streamlit.io), connect GitHub, and select **Create app**.
4. Choose the repository, the `main` branch, and `app.py` as the app file. In **Advanced settings**, choose Python 3.13. No app secrets are currently required.
5. Deploy and wait for Streamlit Community Cloud to report that the app is running. Then open the `streamlit.app` URL it provides. Do not treat the URL as live until the host confirms a successful deployment.

The `.gitignore` excludes `.venv` and `.streamlit/secrets.toml`; do not force-add either or publish credentials. After deployment, push future code changes to `main` to update the app. Community Cloud deployment requires GitHub and Streamlit account authorization; do not share passwords or access tokens.
