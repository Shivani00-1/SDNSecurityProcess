# SDNSecurityProcess
AI-powered security framework for SDN-based cloud-native 5G networks. Uses statistical (EMA, ARIMA) and deep learning (MLP, CNN1D, CNN2D) models for real-time attack detection, DDoS prevention, and monitoring. Includes Python-MySQL backend, SDN Floodlight integration, and web UI.

# SDNSecurityProcess

**Experiments with Digital Security Processes over SDN-Based Cloud-Native 5G Core Networks**  
An AI-powered intrusion detection and prevention framework for SDN-based cloud-native 5G networks. The system integrates statistical models (EMA, ARIMA) and deep learning models (MLP, CNN1D, CNN2D) to detect and mitigate DDoS and other malicious network traffic in real-time.  
It includes a Python-MySQL backend, SDN Floodlight controller integration, and a browser-based UI for monitoring and control.

---

## 📌 Features
- **Real-Time Monitoring** – Collects live SDN traffic statistics and resource usage.
- **AI-Based Attack Detection** – Uses CNN2D for high-accuracy detection of malicious traffic.
- **Decision-Making Module** – Automatically blocks malicious IPs at the SDN level.
- **Web-Based Interface** – User-friendly UI for dataset processing, model execution, and attack monitoring.
- **Model Comparison** – Benchmarks EMA, MLP, CNN1D, and CNN2D for accuracy and speed.

---

## 🎯 Project Aim
To design and implement a **security framework** using AI and SDN monitoring for effective attack detection and prevention in 5G core networks.

---

## 📂 Dataset
- **Source:** [Kaggle – DoS/DDoS Attacks on 5G Networks](https://www.kaggle.com/datasets/iagobs/dosddos-attacks-on-5g-networks)
- **Description:** Includes network protocol type, port numbers, packet lengths, and flow statistics.
- **Usage:** 80% training, 20% testing for model evaluation.

---

## 🛠 Tech Stack
- **Programming Language:** Python 3.7.2
- **Frameworks/Libraries:** Pandas, NumPy, Matplotlib, Scikit-learn, Keras, h5py
- **Database:** MySQL
- **SDN Controller:** Floodlight (OpenFlow Protocol)
- **Deployment:** Local Python web server

---

## ⚙️ Installation & Setup

### 1️⃣ Prerequisites
- Python **3.7.2**
- MySQL Server
- `pip` package manager

### 2️⃣ Install Required Packages
```bash
pip install -r requirements.txt

3️⃣ Configure Database
sql
Copy
Edit
-- Inside MySQL console:
SOURCE path/to/database.txt;
4️⃣ Run the Web Server
Windows:
bash
Copy
Edit
double-click runWebServer.bat
Or:
bash
Copy
Edit
python manage.py runserver
5️⃣ Access Application
Open your browser and go to:

arduino
Copy
Edit
http://127.0.0.1:8000/index.html
🚀 Usage Flow
Register/Login to the application.

Load and split dataset (80% training, 20% testing).

Train models (EMA, MLP, CNN1D, CNN2D) and view performance metrics.

Upload test data for attack detection.

System decides whether to allow or block packets.

📊 Results
CNN2D outperformed all other models in detection accuracy.

Statistical models (EMA, ARIMA) offered faster execution but lower accuracy.

AI models provided better overall security performance.

📌 Future Improvements
Real-time cloud deployment with live SDN hardware.

Support for additional attack categories.

Integration with 6G-SANDBOX testbed.

📜 License
This project is licensed under the MIT License.

👩‍💻 Author
Shivani Yadav
BTech – Electronics & Communication Engineering
Final Year Project Leader | AI & Network Security Enthusiast

⭐ Contribute & Support
If you find this project helpful:

⭐ Star the repository

🐛 Report issues via the Issues tab

📬 Contact for collaborations


