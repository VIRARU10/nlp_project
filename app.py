import streamlit as st
import os
import pandas as pd
import numpy as np
from forensic import ForensicAnalyzer, generate_pdf_report, generate_json_report, get_plagiarism_level
from ngram_baseline import get_similarity_score

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Plagiarism Detection System", page_icon="🕵️", layout="wide")
st.title("Intelligent Plagiarism Detection System")
st.markdown("Analyze documents for plagiarism using AI-driven Semantic Analysis and Document Forensics.")

# ---- LOAD ANALYZER (CACHED) ----
@st.cache_resource
def load_analyzer():
    return ForensicAnalyzer()

try:
    analyzer = load_analyzer()
except Exception as e:
    st.error(f"Failed to load the model or index. Please run `python build_index.py` first. Error: {e}")
    st.stop()

# ---- SIDEBAR ----
st.sidebar.header("Configuration")
model_choice = st.sidebar.radio("Select Model", ["Advanced (Transformers/SBERT)", "Baseline (TF-IDF + Cosine)", "Pure N-Gram (Jaccard)"])
use_baseline = model_choice == "Baseline (TF-IDF + Cosine)"
use_ngram = model_choice == "Pure N-Gram (Jaccard)"

ngram_n = 3
ngram_mode = 'cleaned'
if use_ngram:
    st.sidebar.markdown("---")
    st.sidebar.subheader("N-Gram Config")
    ngram_n = st.sidebar.slider("N-Gram Size (N)", 1, 4, 3)
    ngram_mode = st.sidebar.selectbox("Preprocessing", ["raw", "lowercase", "cleaned"], index=2)

input_method = st.sidebar.radio("Input Method", ["Paste Text", "Upload File (PDF/DOCX/TXT)"])

# ---- MAIN AREA ----
report_map, sent_lengths, burst_score = None, None, None
file_path_for_report = "Streamlit_Input"

if input_method == "Paste Text":
    user_text = st.text_area("Enter text to verify:", height=200)
    if st.button("Analyze Text", type="primary"):
        if not user_text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing text..."):
                if use_ngram:
                    st.info(f"Running Pure N-Gram Matching (N={ngram_n}, Form={ngram_mode}) from UI.")
                    report_map, sent_lengths, burst_score = analyzer.analyze_document_text(user_text, use_baseline=False, use_ngram=True, ngram_n=ngram_n, ngram_mode=ngram_mode)
                else:
                    report_map, sent_lengths, burst_score = analyzer.analyze_document_text(user_text, use_baseline=use_baseline)
                file_path_for_report = "(pasted_text)"

elif input_method == "Upload File (PDF/DOCX/TXT)":
    uploaded_file = st.file_uploader("Upload your document", type=['pdf', 'docx', 'txt'])
    if st.button("Analyze File", type="primary"):
        if uploaded_file is None:
            st.warning("Please upload a file.")
        else:
            with st.spinner("Processing & Analyzing Document..."):
                temp_filename = f"temp_{uploaded_file.name}"
                with open(temp_filename, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                if use_ngram:
                    st.info(f"Running Pure N-Gram Matching (N={ngram_n}, Form={ngram_mode}) from UI.")
                    report_map, sent_lengths, burst_score = analyzer.analyze_document(temp_filename, use_baseline=False, use_ngram=True, ngram_n=ngram_n, ngram_mode=ngram_mode)
                else:
                    report_map, sent_lengths, burst_score = analyzer.analyze_document(temp_filename, use_baseline=use_baseline)
                file_path_for_report = uploaded_file.name
                
                # Clean up temporary file
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)

# ---- RESULTS DISPLAY ----
if report_map is not None:
    st.divider()
    st.header("Analysis Results")
    
    # Calculate key metrics
    avg_sim = np.mean([m['match']['score'] for m in report_map]) if report_map else 0
    plag_text, plag_color = get_plagiarism_level(avg_sim)
    burst_val = float(burst_score) if burst_score is not None else 0.0
    ai_status = "⚠️ LIKELY AI" if burst_val < 2.0 else "👤 LIKELY HUMAN"
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Similarity", f"{avg_sim:.2f}%")
    col2.metric("Plagiarism Assessment", plag_text)
    col3.metric("Writing Style", ai_status, delta=f"Burstiness: {burst_val:.2f}", delta_color="off")
    
    st.markdown("---")
    
    col_chart, col_matches = st.columns([1, 1])
    
    with col_chart:
        st.subheader("Author Rhythm (Burstiness)")
        st.markdown("*Sentence length variance across the text.*")
        chart_data = pd.DataFrame({"Word Count": sent_lengths})
        st.line_chart(chart_data)
        
    with col_matches:
        st.subheader("High-Risk Segments")
        # Filter matches greater than 30% for highlighting
        high_risk = list([m for m in report_map if m.get('match', {}).get('score', 0) > 30])
        if high_risk:
            for m in high_risk[:10]: # show top 10
                score = m['match']['score']
                match_file = m['match']['file']
                text = m['text']
                # Color code
                color = "red" if score >= 70 else "orange"
                st.markdown(f"**<span style='color:{color}'>[{score}% Match in {match_file}]</span>**: {text}", unsafe_allow_html=True)
            if len(high_risk) > 10:
                st.caption(f"... and {len(high_risk) - 10} more matches.")
        else:
            st.success("No significant overlapping segments found.")

    st.markdown("---")
    st.subheader("Download Detailed Reports")
    
    with st.spinner("Generating export files..."):
        # Generate the files
        generate_pdf_report(file_path_for_report, report_map, sent_lengths, burst_score)
        generate_json_report(file_path_for_report, report_map, sent_lengths, burst_score)
        
        pdf_file = "Forensic_Analysis_Report.pdf"
        json_file = "Forensic_Analysis_Report.json"
        
        col_dl1, col_dl2 = st.columns(2)
        
        if os.path.exists(pdf_file):
            with open(pdf_file, "rb") as pdf_data:
                col_dl1.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_data,
                    file_name=f"Forensic_Report_{file_path_for_report}.pdf",
                    mime="application/pdf"
                )
                
        if os.path.exists(json_file):
            with open(json_file, "r") as json_data:
                col_dl2.download_button(
                    label="{ } Download JSON Report",
                    data=json_data.read(),
                    file_name=f"Forensic_Report_{file_path_for_report}.json",
                    mime="application/json"
                )
