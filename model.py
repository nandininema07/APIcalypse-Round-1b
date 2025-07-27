#!/usr/bin/env python3
"""
Adobe Hackathon Round 1B - PDF Document Intelligence System
Context-aware extraction and ranking of document sections using LayoutLMv3 for heading detection
"""

import json
import os
import time
import re
import logging
from datetime import datetime
from typing import List, Dict, Tuple

import pdfplumber
import nltk
import torch
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, BertModel, LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from PIL import Image
import PyPDF2
import pytesseract
import platform

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

class DocumentAnalyzer:
    def __init__(self):
        """Initialize the document analyzer with BERT-Tiny and LayoutLMv3 models"""
        try:
            self.bert_tokenizer = BertTokenizer.from_pretrained('./pretrained_models_bert_tiny')
            self.bert_model = BertModel.from_pretrained('./pretrained_models_bert_tiny')
            self.bert_model.eval()
            logger.info("BERT-Tiny model loaded successfully")

            self.layoutlm_processor = LayoutLMv3Processor.from_pretrained("./models/layoutlmv3", apply_ocr=False)
            self.layoutlm_model = LayoutLMv3ForTokenClassification.from_pretrained("./models/layoutlmv3", ignore_mismatched_sizes=True)
            
            self.layoutlm_model.eval()
            logger.info("LayoutLMv3 model and processor loaded successfully")

            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                logger.info("Downloading NLTK punkt data")
                nltk.download('punkt', quiet=True)

            try:
                nltk.data.find('taggers/averaged_perceptron_tagger_eng')
            except LookupError:
                logger.info("Downloading NLTK averaged_perceptron_tagger_eng data")
                nltk.download('averaged_perceptron_tagger_eng', quiet=True)

        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    def extract_domain_terms(self, persona: str, job_to_be_done: str) -> List[str]:
        """Extract domain-specific terms using semantic similarity"""
        combined_text = f"{persona} {job_to_be_done}".lower()
        stop_words = set(['a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text)
        valid_words = [word for word in words if word not in stop_words]

        # Use embeddings to compute semantic relevance
        inputs = self.bert_tokenizer(valid_words, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            word_embeddings = outputs.last_hidden_state.mean(dim=1).numpy()

        query_inputs = self.bert_tokenizer(combined_text, return_tensors='pt')
        with torch.no_grad():
            query_outputs = self.bert_model(**query_inputs)
            query_embedding = query_outputs.last_hidden_state.mean(dim=1).numpy()

        similarities = cosine_similarity(query_embedding, word_embeddings)
        sorted_indices = np.argsort(similarities[0])[::-1]
        domain_terms = [valid_words[i] for i in sorted_indices[:8]]  # Top 8 terms
        logger.info(f"Extracted domain terms: {domain_terms}")
        return domain_terms

    def extract_context_keywords(self, persona: str, job_to_be_done: str) -> List[str]:
        """Extract context keywords dynamically based on semantic similarity"""
        combined_text = f"{persona} {job_to_be_done}".lower()
        domain_terms = self.extract_domain_terms(persona, job_to_be_done)

        # Extract words from the query text
        words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text)
        stop_words = set(['a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        valid_words = [word for word in words if word not in stop_words]

        # Filter keywords based on their presence in domain terms
        keywords = [word for word in valid_words if any(term in word for term in domain_terms)]
        keywords = list(dict.fromkeys(keywords))[:10]  # Limit to top 10 keywords
        logger.info(f"Extracted context keywords: {keywords}")
        return keywords
    
    def classify_headings_with_layoutlm(self, page_image: Image.Image) -> List[str]:
        """Classify headings using LayoutLMv3"""
        try:
            encoding = self.layoutlm_processor(page_image, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                outputs = self.layoutlm_model(**encoding)
            predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
            tokens = self.layoutlm_processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze().tolist())
            headings = [token for token, prediction in zip(tokens, predictions) if prediction == 1]  # Label 1 for headings
            return headings
        except Exception as e:
            logger.error(f"Error classifying headings with LayoutLMv3: {str(e)}")
            return []
        
    def is_bullet_point(self, text: str) -> bool:
        """Detect bullet points and list-like structures"""
        bullet_patterns = [
            r'^\s*[•▪▫‣⁃\u2022]\s+', r'^\s*[-*+]\s+', r'^\s*\d+[\.\)]\s+', r'^\s*[a-zA-Z][\.\)]\s+',
            r'^\s*(tip|note|warning|caution|important):?\s+'
        ]
        return any(re.match(pattern, text, re.IGNORECASE) for pattern in bullet_patterns)

    def is_valid_heading(self, text: str) -> bool:
        """Validate heading to reject fragments and bullet points"""
        text = text.strip()
        if len(text) < 3 or len(text) > 80 or len(text.split()) < 2 or len(text.split()) > 15:
            return False
        if re.match(r'.*[:,]$', text) or text.lower().startswith(('see ', 'refer ', 'check ', 'visit ')) or self.is_bullet_point(text):
            return False
        return True

    def validate_pdf(self, pdf_path: str) -> bool:
        """Validate PDF file integrity"""
        try:
            with open(pdf_path, 'rb') as f:
                PyPDF2.PdfReader(f, strict=False)
            return True
        except Exception as e:
            logger.error(f"PDF validation failed for {pdf_path}: {str(e)}")
            return False

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Optimized text extraction with OCR fallback"""
        sections = []
        logger.info(f"Processing PDF: {pdf_path}")

        if not self.validate_pdf(pdf_path):
            logger.warning(f"Skipping {pdf_path} due to validation failure")
            return []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 0):
                    text = page.extract_text()
                    if not text:
                        # OCR fallback
                        logger.warning(f"No text extracted from page {page_num}. Using OCR fallback.")
                        page_image = page.to_image(resolution=150).original.convert("RGB")
                        text = pytesseract.image_to_string(page_image)

                    if not text.strip():
                        logger.warning(f"OCR failed to extract text from page {page_num}. Skipping.")
                        continue

                    # Split text into paragraphs
                    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 20]
                    for para in paragraphs:
                        first_line = para.split('\n')[0].strip()
                        if len(first_line.split()) >= 2 and len(para.split()) >= 10 and self.is_valid_heading(first_line):
                            sections.append({
                                'document': os.path.basename(pdf_path),
                                'page': page_num,
                                'section_title': first_line[:80],
                                'text': para,
                                'level': 2
                            })

        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")

        logger.info(f"Extracted {len(sections)} sections from {pdf_path}")
        return sections[:15]

    def encode_text_bert(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Encode texts with mean pooling"""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = [re.sub(r'\s+', ' ', t.strip())[:500] for t in texts[i:i + batch_size]]
            inputs = self.bert_tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings.extend(batch_embeddings.numpy())
        return np.array(embeddings)

    def compute_relevance_scores(self, sections: List[Dict], persona: str, job_to_be_done: str, context_keywords: List[str]) -> List[Tuple[Dict, float]]:
        """Compute relevance scores using semantic similarity and contextual alignment"""
        if not sections:
            return []

        logger.info(f"Computing relevance scores for {len(sections)} sections")
        query = f"As a {persona}, I need to {job_to_be_done}. Focus on: {', '.join(context_keywords)}"
        query_inputs = self.bert_tokenizer(query, return_tensors='pt')
        with torch.no_grad():
            query_outputs = self.bert_model(**query_inputs)
            query_embedding = query_outputs.last_hidden_state.mean(dim=1).numpy()

        adjusted_scores = []
        for section in sections:
            section_text = f"{section['section_title']} {section['text']}"
            section_inputs = self.bert_tokenizer(section_text, return_tensors='pt')
            with torch.no_grad():
                section_outputs = self.bert_model(**section_inputs)
                section_embedding = section_outputs.last_hidden_state.mean(dim=1).numpy()

            # Compute semantic similarity
            similarity = cosine_similarity(query_embedding, section_embedding)[0][0]

            # Adjust scores based on keyword matches and section quality
            keyword_matches = sum(1 for keyword in context_keywords if keyword.lower() in section_text.lower())
            keyword_boost = 1 + (0.2 * keyword_matches)  # Boost score based on keyword matches
            adjusted_score = similarity * keyword_boost
            adjusted_scores.append((section, adjusted_score))

        # Sort sections by relevance
        scored_sections = sorted(adjusted_scores, key=lambda x: x[1], reverse=True)
        return scored_sections[:min(20, len(scored_sections))]

    def filter_sections(self, sections: List[Dict], context_keywords: List[str]) -> List[Dict]:
        """Filter sections based on query context"""
        filtered_sections = []
        for section in sections:
            section_text = f"{section['section_title']} {section['text']}".lower()
            if any(keyword.lower() in section_text for keyword in context_keywords):
                filtered_sections.append(section)
        return filtered_sections

    def extract_key_sentences(self, text: str, persona: str, job_to_be_done: str, context_keywords: List[str], top_k: int = 5) -> str:
        """Extract key sentences ensuring contextual alignment"""
        sentences = [s.strip() for s in nltk.sent_tokenize(text) if len(s.split()) >= 15 and len(s) >= 50]
        if len(sentences) < 2:
            return text[:400].replace('\n', ' ')  # Fallback for short text

        query = f"As a {persona}, I need to {job_to_be_done}. Key aspects: {', '.join(context_keywords)}"
        all_texts = [query] + sentences
        embeddings = self.encode_text_bert(all_texts, batch_size=16)
        query_embedding = embeddings[0:1]
        sentence_embeddings = embeddings[1:]
        similarities = cosine_similarity(query_embedding, sentence_embeddings)[0]

        # Rank sentences by relevance
        ranked_sentences = sorted(zip(sentences, similarities), key=lambda x: x[1], reverse=True)
        selected_sentences = [sentence for sentence, _ in ranked_sentences[:top_k]]
        return ' '.join(selected_sentences).replace('\n', ' ')

def load_input_config(input_dir: str) -> Tuple[str, str]:
    """Load input configuration"""
    config_path = os.path.join(input_dir, "input.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"input.json not found at {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    persona = config.get('persona')
    job_to_be_done = config.get('job_to_be_done')
    if isinstance(persona, dict):
        persona = persona.get('role') or persona.get('name')
    if isinstance(job_to_be_done, dict):
        job_to_be_done = job_to_be_done.get('task') or job_to_be_done.get('description')
    if not persona or not job_to_be_done:
        raise ValueError("Both 'persona' and 'job_to_be_done' must be specified in input.json")
    return str(persona), str(job_to_be_done)

def find_pdf_files(pdfs_dir: str) -> List[str]:
    """Find and validate PDF files"""
    if not os.path.exists(pdfs_dir):
        raise FileNotFoundError(f"PDFs directory not found: {pdfs_dir}")
    pdf_files = [f for f in os.listdir(pdfs_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {pdfs_dir}")
    return [os.path.join(pdfs_dir, f) for f in pdf_files]

def main():
    """Main function to process documents and generate output"""
    start_time = time.time()
    try:
        input_dir = "./input"
        pdfs_dir = os.path.join(input_dir, "PDFs")
        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Loading input configuration")
        persona, job_to_be_done = load_input_config(input_dir)
        logger.info(f"Persona: {persona}")
        logger.info(f"Job to be done: {job_to_be_done}")

        pdf_paths = find_pdf_files(pdfs_dir)
        pdf_files = [os.path.basename(path) for path in pdf_paths]
        logger.info(f"Processing {len(pdf_files)} documents: {pdf_files}")

        logger.info("Initializing document analyzer")
        analyzer = DocumentAnalyzer()

        context_keywords = analyzer.extract_context_keywords(persona, job_to_be_done)
        all_sections = []

        for pdf_path in pdf_paths:
            try:
                # Pass only the required argument
                sections = analyzer.extract_text_from_pdf(pdf_path)
                if sections:
                    all_sections.extend(sections)
                else:
                    logger.warning("")
            except Exception as e:
                logger.error("")

        if not all_sections:
            logger.warning("No sections extracted from any document. Skipping output generation.")
            return

        logger.info(f"Total sections extracted: {len(all_sections)}")
        scored_sections = analyzer.compute_relevance_scores(all_sections, persona, job_to_be_done, context_keywords)
        top_sections = scored_sections

        output_data = {
            "metadata": {
                "input_documents": pdf_files,
                "persona": persona,
                "job_to_be_done": job_to_be_done,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [
                {
                    "document": section['document'],
                    "section_title": section['section_title'],
                    "importance_rank": rank,
                    "page_number": section['page']
                } for rank, (section, _) in enumerate(top_sections[:min(20, len(top_sections))], 1)
            ],
            "sub_section_analysis": [
                {
                    "document": section['document'],
                    "refined_text": analyzer.extract_key_sentences(
                        section['text'], persona, job_to_be_done, context_keywords
                    ),
                    "page_number": section['page']
                } for section, _ in top_sections[:min(20, len(top_sections))]
            ]
        }

        output_path = os.path.join(output_dir, "output.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        processing_time = time.time() - start_time
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        logger.info(f"Output saved to {output_path}")
        for rank, (section, score) in enumerate(top_sections, 1):
            logger.info(f"{rank}. {section['document']} - {section['section_title']} (Level: {section['level']}, Score: {score:.3f})")

        return output_data

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()

#THIS IS GOOD