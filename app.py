import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import joblib
from sklearn.ensemble import RandomForestClassifier
import urllib.parse

# ----------------------------
# Helper Functions
# ----------------------------
def preprocess_file(file, required_cols):
    df = pd.read_excel(file)
    df.columns = df.columns.str.strip().str.lower()
    df = df[required_cols]
    for col in required_cols[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

def retrain_model(existing_data, new_data):
    df = pd.concat([existing_data, new_data], ignore_index=True)
    df = df.drop_duplicates(subset='student_id', keep='last')
    X = df[['attendance', 'avg_marks', 'fee_pending']]
    y = df['dropout']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "dropout_model.pkl")
    return model, df

def predict_row(model, row):
    payload = {'attendance': float(row['attendance']),
               'avg_marks': float(row['avg_marks']),
               'fee_pending': float(row['fee_pending'])}
    X = pd.DataFrame([payload])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return pred, proba[0], proba[1]

def generate_mailto_link(receiver_emails, high_risk_df):
    subject = "🚨 High-Risk Students Alert"

    # Only include student IDs
    body_lines = ["📢 High-Risk Students Report\n"]
    body_lines.append(f"Total High-Risk Students: {len(high_risk_df)}\n")
    body_lines.append("Student IDs:\n")
    
    # List all IDs (or first 50 if too many)
    ids_to_send = high_risk_df['student_id'].astype(str).tolist()
    if len(ids_to_send) > 50:
        ids_to_send = ids_to_send[:50] + ["..."]
    body_lines.append(", ".join(ids_to_send))
    
    # Footer
    body_lines.append("\n⚠️ Please take timely action for these students.")
    body_lines.append("\nBest Regards,")
    body_lines.append("🎓 AI Dropout Dashboard")

    # URL encode
    body_encoded = "%0A".join([urllib.parse.quote(line) for line in body_lines])
    mailto_link = f"mailto:{receiver_emails}?subject={urllib.parse.quote(subject)}&body={body_encoded}"
    return mailto_link

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Student Dropout Dashboard", layout="wide")
st.title("🎓 AI Dropout Prediction Dashboard")
st.markdown("### Upload student data and get predictions with a colorful, interactive UI.")

# File upload
col1, col2, col3 = st.columns(3)
with col1:
    attendance_file = st.file_uploader("📊 Upload Attendance Excel", type=["xlsx"])
with col2:
    marks_file = st.file_uploader("📝 Upload Marks Excel", type=["xlsx"])
with col3:
    fees_file = st.file_uploader("💰 Upload Fees Excel", type=["xlsx"])

# Receiver emails
st.subheader("📧 Email Settings for High-Risk Students")
receiver_emails_input = st.text_area("Mentor Emails (comma-separated)", "")

# ----------------------------
# Process & Merge Files
# ----------------------------
if attendance_file and marks_file and fees_file:
    attendance_df = preprocess_file(attendance_file, ['student_id', 'attendance'])
    marks_df = preprocess_file(marks_file, ['student_id', 'avg_marks'])
    fees_df = preprocess_file(fees_file, ['student_id', 'fee_pending'])

    df = attendance_df.merge(marks_df, on='student_id', how='left').merge(fees_df, on='student_id', how='left')
    df['avg_marks'] = df['avg_marks'].fillna(0)

    # Load old dataset
    try:
        old_df = pd.read_csv("students.csv")
    except FileNotFoundError:
        old_df = pd.DataFrame(columns=['student_id','attendance','avg_marks','fee_pending','dropout'])

    # Label dropout
    df['dropout'] = ((df['attendance'] < 70) | (df['avg_marks'] < 50) | (df['fee_pending'] > 3000)).astype(int)

    # Retrain model
    model, combined_df = retrain_model(old_df, df)
    combined_df.to_csv("students.csv", index=False)

    # Predictions
    result_rows = []
    for _, row in df.iterrows():
        pred, prob_continue, prob_dropout = predict_row(model, row)
        result_rows.append({
            'student_id': row['student_id'],
            'attendance': row['attendance'],
            'avg_marks': row['avg_marks'],
            'fee_pending': row['fee_pending'],
            'risk': pred,
            'prob_continue': prob_continue,
            'prob_dropout': prob_dropout
        })
    result_df = pd.DataFrame(result_rows)

    # ----------------------------
    # Risk Summary Cards
    # ----------------------------
    high_risk = result_df[result_df['risk']==1]
    low_risk = result_df[result_df['risk']==0]

    st.subheader("💡 Risk Summary")
    card_col1, card_col2, card_col3 = st.columns(3)
    card_col1.markdown(f"<div style='background-color:#F0F0F5; padding:15px; border-radius:10px; text-align:center;'>"
                       f"<h4>Total Students</h4><h2>{len(result_df)}</h2></div>", unsafe_allow_html=True)
    card_col2.markdown(f"<div style='background-color:#FF4C4C; padding:15px; border-radius:10px; text-align:center;'>"
                       f"<h4>High-Risk Students</h4><h2>{len(high_risk)}</h2></div>", unsafe_allow_html=True)
    card_col3.markdown(f"<div style='background-color:#4CAF50; padding:15px; border-radius:10px; text-align:center;'>"
                       f"<h4>Low-Risk Students</h4><h2>{len(low_risk)}</h2></div>", unsafe_allow_html=True)

    # ----------------------------
    # Charts
    # ----------------------------
    st.subheader("📊 Risk Distribution")
    chart_data = result_df['risk'].value_counts().rename({0:'Low Risk',1:'High Risk'})
    st.bar_chart(chart_data)
    st.subheader("📈 Pie Chart of Risk")
    st.pyplot(chart_data.plot.pie(autopct='%1.1f%%', colors=["#E94848","#68D85E"]).figure)

    # ----------------------------
    # Detailed Table
    # ----------------------------
    st.subheader("📝 Detailed Predictions")
    def highlight_risk(row):
        return ['background-color: #FF4C4C' if col=='risk' and row[col]==1 else '' for col in row.index]
    st.dataframe(result_df.style.apply(highlight_risk, axis=1))

    # ----------------------------
    # Download CSV
    # ----------------------------
    csv_buffer = StringIO()
    result_df.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download Predictions CSV", csv_buffer.getvalue(), "predictions.csv", "text/csv")

    # ----------------------------
    # Open Email Client (mailto)
    # ----------------------------
    if st.button("Open Email Client for High-Risk Students"):
        if high_risk.empty:
            st.warning("No high-risk students to email.")
        elif not receiver_emails_input:
            st.warning("Please enter mentor emails.")
        else:
            receivers = ",".join([email.strip() for email in receiver_emails_input.split(",")])
            mailto_link = generate_mailto_link(receivers, high_risk)
            st.markdown(f'<a href="{mailto_link}" target="_blank">'
                        f'<button style="padding:10px 20px; background-color:#FF4C4C; color:white; border:none; border-radius:5px; cursor:pointer;">'
                        f'📧 Open Email Client</button></a>', unsafe_allow_html=True)
