import os
import re
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from docx import Document
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import argparse

# ================= 1. THE BRAIN & ANALYZER =================
class ForensicAnalyzer:
    def __init__(self):
        print("🚀 Loading Models & Forensic Index...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.source_index = joblib.load('forensic_index.pkl')
        self.styles = getSampleStyleSheet()
        try:
            self.tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
            self.has_baseline = True
        except:
            self.has_baseline = False

    def extract_text(self, file_path):
        """Standardizes input from PDF, DOCX, or TXT."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            with pdfplumber.open(file_path) as pdf:
                return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        elif ext == '.docx':
            doc = Document(file_path)
            return "\n".join(p.text for p in doc.paragraphs)
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()

    def get_stylometrics(self, text):
        """Innovation: Measures Author Rhythm (Burstiness)."""
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.split()) > 1]
        lengths = [len(s.split()) for s in sentences]
        return lengths, np.std(lengths) if len(lengths) > 1 else 0

    def analyze_document(self, file_path, use_baseline=False, use_ngram=False, ngram_n=3, ngram_mode='cleaned'):
        raw_text = self.extract_text(file_path)
        return self._analyze(raw_text, use_baseline, use_ngram, ngram_n, ngram_mode)

    def analyze_document_text(self, raw_text, use_baseline=False, use_ngram=False, ngram_n=3, ngram_mode='cleaned'):
        return self._analyze(raw_text, use_baseline, use_ngram, ngram_n, ngram_mode)

    def _analyze(self, raw_text, use_baseline, use_ngram=False, ngram_n=3, ngram_mode='cleaned'):
        from ngram_baseline import get_similarity_score, preprocess_text, generate_ngrams, overlap_coefficient, STOP_WORDS
        import string
        import scipy.sparse as sp
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', raw_text) if len(s.split()) > 3]
        
        sent_lengths, burst_score = self.get_stylometrics(raw_text)
        report_map = []

        print(f"🔬 Mapping {len(sentences)} sentences for exact plagiarism... (Baseline={use_baseline}, NGram={use_ngram})")
        
        sim_matrix = None
        if use_baseline and self.has_baseline and not use_ngram:
            sent_tfidfs = self.tfidf_vectorizer.transform(sentences)
            all_tfidfs = sp.vstack([item['tfidf_vector'] for item in self.source_index if 'tfidf_vector' in item])
            sim_matrix = cosine_similarity(sent_tfidfs, all_tfidfs)
            
        elif not use_ngram:
            sent_vecs = self.model.encode(sentences, show_progress_bar=False, convert_to_numpy=True)
            # Vectorize SBERT search
            all_vecs = np.array([item['vector'] for item in self.source_index])
            sent_norms = np.linalg.norm(sent_vecs, axis=1, keepdims=True)
            all_norms = np.linalg.norm(all_vecs, axis=1, keepdims=True)
            sent_vecs_norm = sent_vecs / np.where(sent_norms == 0, 1, sent_norms)
            all_vecs_norm = all_vecs / np.where(all_norms == 0, 1, all_norms)
            sim_matrix = np.dot(sent_vecs_norm, all_vecs_norm.T)
            
        else:
            # Pre-compute N-grams for sentences to avoid repetitive tokenization
            sent_data = []
            for sent in sentences:
                tokens = preprocess_text(sent, ngram_mode)
                ngrams = generate_ngrams(tokens, ngram_n)
                clean_set = set(t for t in sent.lower().split() if t not in STOP_WORDS and t not in string.punctuation and t.isalnum())
                sent_data.append({"ngrams": ngrams, "set": clean_set})
                
        for i, sent in enumerate(sentences):
            best_match = {"file": "None", "score": 0.0}
            
            if not use_ngram:
                if len(self.source_index) > 0:
                    best_idx = np.argmax(sim_matrix[i])
                    best_sim = float(sim_matrix[i][best_idx])
                    if best_sim > best_match["score"]:
                        best_match = {"file": self.source_index[best_idx]['filename'], "score": float(f"{best_sim * 100:.2f}")}
            else:
                s_ngrams = sent_data[i]["ngrams"]
                s_set = sent_data[i]["set"]
                
                if s_ngrams:
                    highest_sim = 0.0
                    best_file = "None"
                    for item in self.source_index:
                        cc_text = item['text']
                        cc_set = set(t for t in cc_text.lower().split() if t not in STOP_WORDS and t not in string.punctuation and t.isalnum())
                        
                        # Aggressive stopword heuristic: sentence vs 200-word chunk
                        req_matches = max(1, min(3, len(s_set) // 2))
                        if len(s_set.intersection(cc_set)) < req_matches:
                            continue
                            
                        # N-gram overlap
                        tokens_b = preprocess_text(item['text'], ngram_mode)
                        ngrams_b = generate_ngrams(tokens_b, ngram_n)
                        sim = overlap_coefficient(s_ngrams, ngrams_b)
                        
                        if sim > highest_sim:
                            highest_sim = sim
                            best_file = item['filename']
                            
                    best_match = {"file": best_file, "score": float(f"{highest_sim * 100:.2f}")} if highest_sim > 0 else best_match

            report_map.append({"text": sent, "match": best_match})

        return report_map, sent_lengths, burst_score

# ================= 2. REPORT GENERATOR =================
def get_plagiarism_level(avg_sim):
    """Maps average similarity score to standard threshold brackets."""
    if avg_sim < 30:
        return f"LOW ({float(avg_sim):.1f}%)", colors.green
    elif avg_sim < 70:
        return f"MEDIUM ({float(avg_sim):.1f}%)", colors.orange
    else:
        return f"HIGH ({float(avg_sim):.1f}%)", colors.red

def generate_pdf_report(file_path, report_map, sent_lengths, burst_score):
    report_name = "Forensic_Analysis_Report.pdf"
    doc = SimpleDocTemplate(report_name)
    elements = []
    styles = getSampleStyleSheet()

    # Title
    elements.append(Paragraph(f"Forensic Plagiarism Report: {os.path.basename(file_path)}", styles['Title']))
    elements.append(Spacer(1, 12))

    # Summary Table
    avg_sim = np.mean([m['match']['score'] for m in report_map]) if report_map else 0
    plag_text, plag_color = get_plagiarism_level(avg_sim)
    ai_status = f"⚠️ LIKELY AI (Variance {float(burst_score):.2f})" if burst_score < 2.0 else f"👤 LIKELY HUMAN (Variance {float(burst_score):.2f})"

    data = [['Metric', 'Value'],
            ['Average Similarity', f"{float(avg_sim):.2f}%"],
            ['Author Rhythm (Burstiness)', f"{float(burst_score):.2f}"],
            ['Plagiarism Level', plag_text],
            ['Writing Style', ai_status]]
    
    t = Table(data, colWidths=[200, 200])
    t.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey), ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                           ('GRID', (0,0), (-1,-1), 1, colors.black), ('FONTSIZE', (0,0), (-1,-1), 12),
                           ('TEXTCOLOR', (1, 3), (1, 3), plag_color)]))
    elements.append(t)
    elements.append(Spacer(1, 24))

    # --- CHART 1: Author Rhythm (Burstiness) ---
    plt.figure(figsize=(6, 3))
    plt.plot(sent_lengths, marker='o', color='blue', linestyle='--')
    plt.title("Author DNA: Sentence Length Variance")
    plt.xlabel("Sentence Index")
    plt.ylabel("Word Count")
    plt.savefig("burst_chart.png")
    elements.append(Image("burst_chart.png", width=400, height=200))
    elements.append(Paragraph("<i>A flat line indicates robotic/AI writing; a jagged line indicates human rhythm.</i>", styles['Italic']))
    
    # --- EXACT MAPPING SECTION ---
    elements.append(Spacer(1, 24))
    elements.append(Paragraph("Detailed Paragraph Mapping (How & Where)", styles['Heading2']))
    
    for item in report_map:
        _, p_color = get_plagiarism_level(item['match']['score'])
        color_style = ParagraphStyle('Risk', parent=styles['Normal'], textColor=p_color)
        p_text = f"<b>[{item['match']['score']}% Match - {item['match']['file']}]</b>: {item['text']}"
        elements.append(Paragraph(p_text, color_style))
        elements.append(Spacer(1, 6))

    doc.build(elements)
    print(f"✅ Full Forensic Document Report Generated: {report_name}")

def generate_json_report(file_path, report_map, sent_lengths, burst_score):
    report_name = "Forensic_Analysis_Report.json"
    avg_sim = np.mean([m['match']['score'] for m in report_map]) if report_map else 0
    plag_text, _ = get_plagiarism_level(avg_sim)
    ai_status = f"LIKELY AI (Variance {float(burst_score):.2f})" if burst_score < 2.0 else f"LIKELY HUMAN (Variance {float(burst_score):.2f})"

    report_data = {
        "file_name": os.path.basename(file_path),
        "overall_metrics": {
            "average_similarity": float(f"{float(avg_sim):.2f}"),
            "plagiarism_level": plag_text,
            "burstiness_score": float(f"{float(burst_score):.2f}"),
            "writing_style": ai_status
        },
        "paragraph_mapping": report_map
    }

    with open(report_name, "w") as f:
        json.dump(report_data, f, indent=4)
    print(f"✅ Full Forensic JSON Report Generated: {report_name}")

# ================= 3. EXECUTION =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Forensic plagiarism analysis')
    parser.add_argument('--file', '-f', help='Path to input file (PDF/DOCX/TXT)')
    parser.add_argument('--text', '-t', help='Raw text to analyze (wrap in quotes)')
    parser.add_argument('--baseline', action='store_true', help='Use TF-IDF baseline model instead of Transformer')
    args = parser.parse_args()

    analyzer = ForensicAnalyzer()

    # Priority: --file, then --text, otherwise interactive prompt
    if args.file:
        if not os.path.exists(args.file):
            print('ERROR: file not found:', args.file)
            raise SystemExit(2)
        mapping, lengths, burst = analyzer.analyze_document(args.file, use_baseline=args.baseline)
        generate_pdf_report(args.file, mapping, lengths, burst)
        generate_json_report(args.file, mapping, lengths, burst)
    elif args.text:
        mapping, lengths, burst = analyzer.analyze_document_text(args.text, use_baseline=args.baseline)
        generate_pdf_report("(cmdline_text)", mapping, lengths, burst)
        generate_json_report("(cmdline_text)", mapping, lengths, burst)
    else:
        target_input = input("📁 Drag and drop your file here (PDF/DOCX/TXT) or paste raw text: ").strip('"')
        use_baseline_input = input("Use baseline model TF-IDF? (y/n) [n]: ").strip().lower()
        use_baseline = use_baseline_input == 'y'

        # If the input is an existing file path, analyze it. Otherwise, treat input as raw text.
        if os.path.exists(target_input):
            mapping, lengths, burst = analyzer.analyze_document(target_input, use_baseline=use_baseline)
            generate_pdf_report(target_input, mapping, lengths, burst)
            generate_json_report(target_input, mapping, lengths, burst)
        else:
            mapping, lengths, burst = analyzer.analyze_document_text(target_input, use_baseline=use_baseline)
            generate_pdf_report("(pasted_text)", mapping, lengths, burst)
            generate_json_report("(pasted_text)", mapping, lengths, burst)