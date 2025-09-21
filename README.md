# AI-Based Student Dropout Prediction and Counseling System

## **Project Overview**

This project is a **Student Dropout Prediction Dashboard** that helps educational institutes **identify students at risk of dropping out** early, using attendance, marks, and fee data. The system provides:

* A **clean, interactive web interface** for uploading student data.
* **Rule-based predictive logic** to flag high-risk students.
* **Visualizations** including cards, bar charts, and pie charts to summarize risk distribution.
* **Downloadable CSV** of risk predictions for further analysis.
* **Modern email integration** to alert mentors about high-risk students with a preformatted table.

The system is designed to be **fully functional without complex ML training**, yet it demonstrates the principles of **AI-powered prediction**, as it identifies patterns and provides actionable insights.

---

## **Project Structure**

```
student-dropout-dashboard/
│
├─ app.py                 # Main Streamlit app
├─ students.csv           # Optional historical data
├─ requirements.txt       # Python dependencies
├─ README.md              # Project documentation
└─ sample_excel_files/    # Example Excel files (Attendance, Marks, Fees)
```

--Other files like create_data and ml_server and train_model are for creation of the students.csv and to validate the ml ..

---

## **How the Project Works**

### **1️⃣ Upload Student Data**

* The user uploads three Excel files:

  1. Attendance data
  2. Marks/Grades data
  3. Fee pending data
* Each Excel file is processed and cleaned automatically. Missing values are replaced with zeros.

### **2️⃣ Data Processing**

* The system merges all three datasets into a single table using `student_id` as the key.
* Each student is assigned a **risk label** based on simple thresholds:

  * Attendance < 70% → high-risk
  * Average Marks < 50 → high-risk
  * Fee Pending > 3000 → high-risk

> This is a **predictive AI-like logic**: although it is rule-based, it mimics how AI identifies students likely to drop out based on multiple indicators.

### **3️⃣ Visualization**

* **Dashboard Cards:** Show total students, high-risk students, and low-risk students.
* **Charts:** Bar chart and pie chart for risk distribution.
* **Detailed Table:** Highlights high-risk students for quick review.

### **4️⃣ CSV Export**

* After analysis, the user can download a **CSV file** containing all students with their calculated risk, probabilities, and other details.
* This makes it easy to maintain records or integrate with other systems.

### **5️⃣ Email Integration**

* Mentors can be notified via **email** using the built-in **mailto link**.
* The email includes:

  * Pre-filled **subject line**: "High-Risk Students Alert"
  * **HTML table** of high-risk students with attendance, marks, and fee pending
  * Custom message for context
* No SMTP configuration or password is required, making it secure and easy to use. Clicking the button opens the default email client with everything pre-filled.

### **6️⃣ Optional Historical Data**

* The system can optionally use `students.csv` to store previous analyses.
* This allows tracking trends over time and improving decision-making.

---

## **Why This is AI-Powered**

* While the current system uses **rule-based thresholds**, it demonstrates **predictive analytics**, which is the core of AI:

  * **Data-Driven Insights:** The system looks at multiple indicators (attendance, marks, fees) simultaneously.
  * **Pattern Recognition:** Identifies students at risk before actual dropout occurs.
  * **Actionable Alerts:** Provides predictions to mentors for timely intervention.

In a more advanced version, a **machine learning model** can replace the thresholds to learn patterns from historical data, making the system truly self-learning.

---

## **How to Run the Project**

### **1. Install Python dependencies**

```bash
pip install -r requirements.txt
```

### **2. Run the Streamlit app**

```bash
streamlit run app.py
```

### **3. Use the Dashboard**

1. Upload the Attendance, Marks, and Fees Excel files.
2. Enter mentor emails (comma-separated).
3. View risk cards, charts, and detailed table.
4. Download predictions as CSV.
5. Click the **Open Email Client** button to alert mentors.

---

## **Dependencies**

* `streamlit` – Web app interface
* `pandas` – Data processing
* `numpy` – Numeric computations
* `scikit-learn` – For optional ML model (if used later)
* `urllib` – For generating mailto links

---

## **Future Enhancements**

* Replace rule-based thresholds with **trained ML models** for better accuracy.
* Integrate **automatic email sending** via SMTP (optional).
* Add **visual trend analysis** over multiple semesters.
* Support **real-time dashboard updates** as new data is uploaded.

---

✅ **Result:** A secure, interactive, and modern dashboard that helps institutes **intervene early and reduce student dropout rates** without requiring additional budget or training.

student-dropout-dashboard/
│
├─ app.py              # Main Streamlit app
├─ students.csv        # Optional historical data
├─ requirements.txt    # Python dependencies
├─ README.md           # Documentation
└─ sample_excel_files/ # Example Excel files: Attendance, Marks, Fees


           ┌────────────────────┐
           │ Upload Excel Files │
           │ Attendance, Marks, │
           │ Fees               │
           └─────────┬─────────┘
                     │
                     ▼
           ┌────────────────────┐
           │ Merge & Process    │
           │ Data by student_id │
           └─────────┬─────────┘
                     │
                     ▼
           ┌────────────────────┐
           │ Risk Calculation   │
           │ (Rule-Based)       │
           │ attendance < 70    │
           │ marks < 50         │
           │ fees > 3000        │
           └─────────┬─────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
 ┌───────────────┐        ┌───────────────┐
 │ Visual Cards  │        │ Detailed Table│
 │ & Charts      │        │ Highlight Risk│
 └───────┬───────┘        └───────┬───────┘
         │                       │
         ▼                       ▼
 ┌──────────────────────────┐   ┌──────────────────────────┐
 │ Download Predictions CSV │   │ Open Email Client for    │
 │ for records              │   │ Mentors (mailto link)   │
 └──────────────────────────┘   └──────────────────────────┘
