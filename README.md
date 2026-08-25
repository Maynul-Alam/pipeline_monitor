# 🛢️ Pipeline Leak Detection System

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.62.0-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Render-purple.svg)](https://your-app-url.onrender.com)

An interactive **Pipeline Leak Detection** simulation that monitors pressure, flow, temperature, and volume in real‑time. Designed to demonstrate anomaly detection concepts for gas/water pipelines.

## 🚀 Live Demo

🔗 [**Try the Live App**](https://your-app-url.onrender.com)

## 📸 Screenshots

![Dashboard](docs/images/dashboard.png)
*Real‑time monitoring dashboard*

![Alert](docs/images/alert.png)
*Leak detection alert with severity scoring*

## ✨ Features

- ⏱️ **Real‑time simulation** – pressure, flow, temperature, volume
- 🚨 **Automatic leak detection** – random or manual injection
- 📊 **Severity scoring** – 0–100 scale based on deviation
- 📈 **Live metrics** – instant feedback on critical parameters
- 📋 **Detection log** – timestamped events with severity
- 📤 **CSV export** – download log for compliance/reporting
- 📱 **Mobile‑friendly** – responsive UI for all devices
- 🎨 **Dark/Light mode** – comfortable for control room use

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Streamlit |
| Data Processing | NumPy, Pandas |
| Visualisation | Matplotlib |
| Deployment | Render / Streamlit Cloud |

## 📁 Project Structure

pipeline_monitor/
├── app/ # Streamlit web app
│ ├── app.py # Main dashboard
│ └── utils.py # Helper functions
├── src/ # Core logic
│ └── data_preprocessing.py
├── data/ # Sample data
├── models/ # Trained models (optional)
├── requirements.txt # Python dependencies
├── README.md # This file
└── LICENSE # MIT License

## 🏃 Local Setup

### Prerequisites
- Python 3.10 or higher
- pip / conda

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/pipeline_monitor.git
   cd pipeline_monitor
2.Create and activate a virtual environment

bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

3.Install dependencies

bash
pip install -r requirements.txt

4.Run the app locally

bash
streamlit run app/app.py

Open your browser at http://localhost:8501

   
---

## 📄 LICENSE (MIT)

Create a file called `LICENSE` in your repository root:

```txt
MIT License

Copyright (c) 2026 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.62.0-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Render-purple.svg)](https://your-app-url.onrender.com)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
