#!/usr/bin/env python3
import os
import sys
import argparse
import joblib
from tqdm import tqdm

from chunking import chunk_text

# Text extraction helpers
import re
try:
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    from docx import Document
except Exception:
    Document = None
import xml.etree.ElementTree as ET

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


def find_source_documents(root):
    for dirpath, dirs, files in os.walk(root):
        if os.path.basename(dirpath) == 'source-documents':
            return dirpath
    return None


def extract_text_from_file(path):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == '.txt':
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        elif ext == '.xml':
            try:
                tree = ET.parse(path)
                return "\n".join(tree.getroot().itertext())
            except Exception:
                # fallback: strip tags
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    txt = f.read()
                return re.sub(r'<[^>]+>', '', txt)
        elif ext == '.pdf' and pdfplumber is not None:
            try:
                with pdfplumber.open(path) as pdf:
                    pages = [p.extract_text() for p in pdf.pages if p.extract_text()]
                    return "\n".join(pages)
            except Exception:
                return ''
        elif ext == '.docx' and Document is not None:
            try:
                doc = Document(path)
                return "\n".join(p.text for p in doc.paragraphs)
            except Exception:
                return ''
        else:
            # unknown type - try reading as text
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    except Exception:
        return ''


def main():
    parser = argparse.ArgumentParser(description='Build forensic_index.pkl from PAN corpus source-documents using Chunking')
    parser.add_argument('--root', default='.', help='Workspace root to search for source-documents')
    parser.add_argument('--out', default='forensic_index.pkl', help='Output pkl file')
    parser.add_argument('--max-files', default=200, type=int, help='Maximum number of files to index (0 for all)')
    parser.add_argument('--model', default='all-MiniLM-L6-v2', help='SentenceTransformer model name')
    parser.add_argument('--chunk-size', default=200, type=int, help='Word count per chunk')
    parser.add_argument('--chunk-overlap', default=50, type=int, help='Overlap word count between chunks')
    args = parser.parse_args()

    src_dir = find_source_documents(args.root)
    if not src_dir:
        print('ERROR: Could not find a `source-documents` directory under', os.path.abspath(args.root))
        sys.exit(2)

    print('Found source-documents at:', src_dir)

    # gather files
    filepaths: list[str] = []
    for dirpath, dirs, files in os.walk(src_dir):
        for f in files:
            if isinstance(f, str) and f.lower().endswith(('.txt', '.xml', '.pdf', '.docx')):
                filepaths.append(os.path.join(str(dirpath), str(f)))
    filepaths.sort()

    if args.max_files > 0:
        filepaths = list(filepaths)[:args.max_files]

    print(f'Indexing {len(filepaths)} files into sliding window chunks...')

    # Read and chunk
    chunk_fnames = []
    chunk_texts = []
    chunk_ids = []
    
    for p in tqdm(filepaths, desc='Reading & Chunking'):
        txt = extract_text_from_file(p)
        if not txt or len(txt.strip()) < 20:
            continue
            
        chunks = chunk_text(txt, chunk_size=args.chunk_size, overlap=args.chunk_overlap)
        fname = os.path.relpath(p)
        
        for i, c_txt in enumerate(chunks):
            # Only keep chunks of reasonable length
            if len(c_txt.strip()) > 10:
                chunk_fnames.append(fname)
                chunk_texts.append(c_txt)
                chunk_ids.append(i)

    print(f'Generated {len(chunk_texts)} chunks.')

    # load model
    print('Loading SentenceTransformer model:', args.model)
    model = SentenceTransformer(args.model)

    print(f'Computing TF-IDF vectors for {len(chunk_texts)} chunks...')
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(chunk_texts)
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

    print(f'Computing embeddings for {len(chunk_texts)} chunks...')
    vectors = model.encode(chunk_texts, show_progress_bar=True, convert_to_numpy=True)

    index = []
    for i, (fn, c_id, txt, vec) in enumerate(zip(chunk_fnames, chunk_ids, chunk_texts, vectors)):
        index.append({
            'filename': fn, 
            'chunk_id': c_id,
            'text': txt, 
            'vector': vec,
            'tfidf_vector': tfidf_matrix[i]
        })

    print('Saving chunked index to', args.out)
    joblib.dump(index, args.out)
    print('Done. Saved', len(index), 'chunk entries.')


if __name__ == '__main__':
    main()

