# Annex 20 free prediction

Run `streamlit run app.py` from this directory after installing `requirements.txt`. The app predicts velocity and concentration from inlet velocity and a source location. It uses the existing Annex 20 velocity FNO and two-stage concentration FNO–CNN.

The app loads inference-only checkpoint tensors and normalization metadata. It verifies checkpoint SHA-256 digests before loading. The deployed branch uses encrypted checkpoint files. Set `ANNEX20_MODEL_KEY` and `ANNEX20_ACCESS_CODE` in the hosting service's secrets. The demo prompts visitors for the access code on that branch.
