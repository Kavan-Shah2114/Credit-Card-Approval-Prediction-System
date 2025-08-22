import streamlit as st
import pandas as pd
import joblib
import altair as alt
from fpdf import FPDF
import datetime

# --- Page Config ---
st.set_page_config(page_title="Credit Approval Predictor", layout="wide")

# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('credit_card_model.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        model_columns = joblib.load('model_columns.pkl')
        return model, label_encoder, model_columns
    except FileNotFoundError:
        st.error("Model files not found. Please run the training script first.")
        return None, None, None

model, label_encoder, model_columns = load_model()

# --- Sidebar Info for Business Users ---
st.sidebar.header("📘 About this Tool")
st.sidebar.markdown("""
This Credit Approval Prediction tool analyzes applicant financial and credit history to 
estimate the **likelihood of loan approval**.  

### Prediction Categories
**P1 – Very High Probability**  
Low risk, strong profile. Suitable for approval in most cases.

**P2 – High Probability**  
Low risk, good profile. Generally safe to approve.

**P3 – Moderate Probability**  
Some financial concerns; manual review recommended before approval.

**P4 – Low Probability**  
High risk; approval not recommended except in special cases.

---

### Business Guidance
- **P1 & P2:** Generally safe to approve with minimal further checks.  
- **P3:** Requires detailed review and possible risk mitigation before approval.  
- **P4:** Typically reject unless strong compensating factors exist.

---
""")

# --- Input Form ---
def get_user_input():
    with st.expander("📄 Personal Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            net_monthly_income = st.number_input('Net Monthly Income', min_value=0, value=25000)
            time_with_employer = st.number_input('Time with Current Employer (Months)', min_value=0, value=60)
            marital_status = st.selectbox('Marital Status', ['Married', 'Single'])
            gender = st.selectbox('Gender', ['M', 'F'])
        with col2:
            age_oldest_tl = st.number_input('Age of Oldest TL (Months)', min_value=0, value=72)
            age_newest_tl = st.number_input('Age of Newest TL (Months)', min_value=0, value=12)
            education = st.selectbox('Education Level', ['SSC', '12TH', 'GRADUATE', 'POST-GRADUATE', 'PROFESSIONAL', 'OTHERS', 'UNDER GRADUATE'])

    with st.expander("🏦 Loan & Credit History"):
        col3, col4 = st.columns(2)
        with col3:
            cc_tl = st.number_input('Number of Credit Card Loans', min_value=0, value=0)
            pl_tl = st.number_input('Number of Personal Loans', min_value=0, value=1)
            home_tl = st.number_input('Number of Home Loans', min_value=0, value=0)
            tot_missed_pmnt = st.number_input('Total Missed Payments', min_value=0, value=0)
        with col4:
            num_times_60p_dpd = st.number_input('Num Times 60+ Days Past Due', min_value=0, value=0)
            pct_tl_open_L6M = st.slider('TLs Opened in Last 6M (%)', 0.0, 1.0, 0.1)
            pct_tl_closed_L6M = st.slider('TLs Closed in Last 6M (%)', 0.0, 1.0, 0.0)

    with st.expander("📌 Product Enquiries & Flags"):
        last_prod_enq2 = st.selectbox('Last Product Enquired', ['PL', 'ConsumerLoan', 'others', 'AL', 'HL', 'CC'])
        first_prod_enq2 = st.selectbox('First Product Enquired', ['PL', 'ConsumerLoan', 'others', 'AL', 'HL', 'CC'])

        hl_flag = st.selectbox('Has Home Loan?', ['Yes', 'No'])
        hl_flag = 1 if hl_flag == 'Yes' else 0

        gl_flag = st.selectbox('Has Gold Loan?', ['Yes', 'No'])
        gl_flag = 1 if gl_flag == 'Yes' else 0

        cc_flag = st.selectbox('Has Credit Card?', ['Yes', 'No'])
        cc_flag = 1 if cc_flag == 'Yes' else 0

        pl_flag = st.selectbox('Has Personal Loan?', ['Yes', 'No'])
        pl_flag = 1 if pl_flag == 'Yes' else 0

    data = {
        'NETMONTHLYINCOME': net_monthly_income,
        'Time_With_Curr_Empr': time_with_employer,
        'Age_Oldest_TL': age_oldest_tl,
        'Age_Newest_TL': age_newest_tl,
        'MARITALSTATUS': marital_status,
        'EDUCATION': education,
        'GENDER': gender,
        'CC_TL': cc_tl,
        'PL_TL': pl_tl,
        'Home_TL': home_tl,
        'Tot_Missed_Pmnt': tot_missed_pmnt,
        'num_times_60p_dpd': num_times_60p_dpd,
        'pct_tl_open_L6M': pct_tl_open_L6M,
        'pct_tl_closed_L6M': pct_tl_closed_L6M,
        'last_prod_enq2': last_prod_enq2,
        'first_prod_enq2': first_prod_enq2,
        'HL_Flag': hl_flag,
        'GL_Flag': gl_flag,
        'CC_Flag': cc_flag,
        'PL_Flag': pl_flag,
        # Defaults
        'Tot_TL_closed_L12M': 0, 'pct_tl_open_L12M': 0.0, 'pct_tl_closed_L12M': 0.0,
        'Secured_TL': 0, 'Unsecured_TL': 0, 'Other_TL': 0,
        'time_since_recent_payment': 0, 'max_recent_level_of_deliq': 0,
        'num_deliq_6_12mts': 0, 'num_std_12mts': 0, 'num_sub': 0,
        'num_sub_12mts': 0, 'num_dbt': 0, 'num_dbt_12mts': 0, 'num_lss': 0,
        'recent_level_of_deliq': 0, 'enq_L3m': 0,
        'pct_PL_enq_L6m_of_ever': 0.0, 'pct_CC_enq_L6m_of_ever': 0.0
    }
    return pd.DataFrame(data, index=[0])

# --- Prediction ---
if model is not None:
    input_df = get_user_input()

    if st.button("🔍 Predict Approval Status"):
        with st.spinner('Processing your application...'):
            edu_mapping = {'SSC': 1, '12TH': 2, 'GRADUATE': 3, 'UNDER GRADUATE': 3,
                           'POST-GRADUATE': 4, 'OTHERS': 1, 'PROFESSIONAL': 3}
            input_df['EDUCATION'] = input_df['EDUCATION'].map(edu_mapping)
            input_df_encoded = pd.get_dummies(input_df)
            final_df = input_df_encoded.reindex(columns=model_columns, fill_value=0)
            prediction_encoded = model.predict(final_df)
            prediction = label_encoder.inverse_transform(prediction_encoded)
            proba = model.predict_proba(final_df)[0]

        # --- Result Card ---
        color_map = {
            'P1': '#22c55e', 
            'P2': '#15803d',  
            'P3': '#ca8a04',
            'P4': '#b91c1c'
        }
        result_status = prediction[0]
        st.markdown(
            f'<div style="background-color:{color_map[result_status]};color:white;padding:15px;border-radius:10px;">'
            f'<h4>{result_status}</h4>'
            f'<p>{["Very High","High","Moderate","Low"][int(result_status[-1])-1]} probability category.</p>'
            f'</div>', unsafe_allow_html=True
        )

        # Add space before chart
        st.markdown("<br>", unsafe_allow_html=True)

        # --- Probability Chart ---
        proba_df = pd.DataFrame({'class': label_encoder.classes_, 'probability': proba})
        chart = alt.Chart(proba_df).mark_bar().encode(
            x=alt.X('class', sort=label_encoder.classes_),
            y='probability',
            color=alt.Color('class', scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values()))),
            tooltip=['class', 'probability']
        ).properties(width=500, height=300)
        st.altair_chart(chart, use_container_width=True)

        # --- PDF Download ---
        def create_pdf(pred_class, prob_data, user_input_df):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=14)
            pdf.cell(200, 10, txt="Credit Approval Prediction Report", ln=True, align='C')
            
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
            pdf.cell(200, 10, txt=f"Predicted Class: {pred_class}", ln=True)
            pdf.ln(5)
            
            pdf.cell(200, 10, txt="Class Probabilities:", ln=True)
            for cls, pr in zip(prob_data['class'], prob_data['probability']):
                pdf.cell(200, 10, txt=f"{cls}: {pr:.2%}", ln=True)
            
            pdf.ln(5)
            pdf.cell(200, 10, txt="Applicant Details:", ln=True)
            for col, val in user_input_df.iloc[0].items():
                pdf.multi_cell(0, 8, f"{col}: {val}")
            
            return pdf.output(dest='S').encode('latin1')

        # Create PDF and save locally
        pdf_data = create_pdf(result_status, proba_df, input_df)

        # Save local copy
        local_filename = f"prediction_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        with open(local_filename, "wb") as f:
            f.write(pdf_data)
        st.success(f"📂 Report saved locally as `{local_filename}`")

        # Download button
        st.download_button(
            "📥 Download Prediction Report",
            data=pdf_data,
            file_name="prediction_report.pdf",
            mime="application/pdf"
        )
