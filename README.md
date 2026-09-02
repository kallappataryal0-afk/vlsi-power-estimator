⚡ VLSI Power Estimation using Machine Learning
This project implements a Machine Learning-based approach to predict power consumption in digital VLSI circuits, replacing slow traditional simulation tools like SPICE or Cadence. By leveraging ML, this tool provides instant power estimations, enabling rapid design space exploration.

🚀 Live Web App: [http://10.185.45.151:8501](https://vlsi-power-estimator-fnslxymig8pgdse4vfchwr.streamlit.app/)


📊 Key Findings
This project replicates and extends the IEEE paper: "Efficient Power Estimation and Analysis of Digital Circuits Using Machine Learning Techniques."

While the reference paper found Ridge Regression to be optimal for a small dataset, my project yielded a critical extension:

Linear Models (Ridge Regression): Maxed out at an R² score of 0.61 (61%) because they cannot map the non-linear physics of VLSI circuits (e.g., P∝C⋅V 
2⋅f).
Deep Learning (MLP Regressor): Emerged as the best-performing model, achieving an R² score of 0.96 (96%). The Neural Network's hidden layers successfully mapped these complex curves, proving deep learning is required for modern circuit datasets.
<img width="1376" height="742" alt="image" src="https://github.com/user-attachments/assets/d9a4f032-be55-4dfe-af4a-78c752a5d6f0" />

💻 Tech Stack
Language: Python
Machine Learning: Scikit-Learn, XGBoost
Data Manipulation: Pandas, NumPy
Visualization: Matplotlib, Seaborn
Web GUI Framework: Streamlit


🌐 Web Application (GUI)
I developed an interactive web application using Streamlit. Engineers can:

Upload raw CSV data files (simulating reports extracted from EDA tools).
Select from multiple ML models (Ridge, Random Forest, XGBoost, MLP).
Instantly view performance metrics (R², MAE) and prediction graphs.
Download the resulting power predictions.


📁 Repository Structure
vlsi-power-estimator/
│
├── app.py # Streamlit Web GUI code
├── requirements.txt # Python dependencies
├── digital_circuit_power_consumption_5000.csv # Main training dataset
├── industry_benchmark_1000.csv # Complex industry test dataset
└── images/ # Generated graphs and plots


## 🛠️ How to Run Locally
1. Clone this repository.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt

streamlit run app.py

Open your browser to http://localhost:8501 and upload the CSV dataset!



🏭 Real-World Industry Integration
In a real semiconductor manufacturing flow, engineers do not manually input circuit parameters. This project includes a Python parsing script template that automatically extracts the 18 required features from raw EDA text reports, formats them, and feeds them directly into the ML model—creating a fully automated, server-side power estimation pipeline.





















   







