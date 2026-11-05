---
title: FLOODCAST
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.31.0"
python_version: "3.12"
app_file: space_app.py
pinned: false
short_description: River discharge forecasts and flood early warnings
---

# FLOODCAST (Team Midnight Cowens)

Daily river discharge prediction from weather and catchment attributes, with flood early warnings.

- `/` is the web app: Am I safe?, Officer dashboard, Batch (test.csv to predictions.csv, submission
  checks, evaluation metrics), Manual input.
- `/gradio/` is the classic app in English, Hindi and Kannada.

Code, model card and instructions: see the GitHub repository. This Space is deployed from it with
`scripts/deploy_space.py`.
