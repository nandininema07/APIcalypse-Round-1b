# Context-Aware PDF Document Intelligence System (Round 1B)

## Overview

This project builds upon the PDF Outline Extractor from [Round 1A](https://github.com/nandininema07/APIcalypse-Round-1a), enhancing document intelligence by focusing on **context-aware extraction and ranking of relevant document sections**. It leverages advanced NLP techniques and a specialized LayoutLMv3 model to not only identify headings but also understand their relevance to a user's specific persona and job-to-be-done. The system provides prioritized sections and refined key sentences, turning any PDF into an intelligent, actionable resource.

## Demo Video and Detailed Documentation

* [Click here for a **short video demonstration** of our execution.](https://youtu.be/jevhGHqSgsQ)

* [Click here for a **detailed documentation** of our execution.](documentation%20and%20demo/APIcalypse%20Documentation_Adobe%20Round%201b.pdf)

## Our Approach: Intelligent Section Extraction and Ranking
System Architecture:
![System Architecture](documentation%20and%20demo/APIcalypse%20Round%201b_System%20Architecture.png)

Project Pipeline:
![Pipeline](documentation%20and%20demo/APIcalypse%20round%201b_Process%20Flow.png)

Our solution is designed to understand your specific needs (persona and job-to-be-done) and highlighting the most relevant parts of a PDF. This is achieved through a multi-stage process:

1.  **Domain and Context Understanding:**
    * We first extract **domain-specific terms** based on the provided persona and job-to-be-done. This involves converting text into numerical tokens and using **K-Means clustering** to identify up to 8 core terms that best explain the context.
    * **Context keywords** are then identified, similar to domain terms but with a slightly softer approach, providing additional relevant terms.

2.  **Layout-Aware Section Extraction:**
    * Unlike Round 1A's rule-based approach, this system primarily uses **LayoutLMv3ForTokenClassification** for heading detection, processing PDF pages as images to understand visual layout in conjunction with text. This is crucial for accurately identifying section titles.
    * Text is extracted using `pdfplumber`, and each line is analyzed.
    * A robust validation system ensures extracted headings are meaningful, rejecting fragments, bullet points, or overly short/long lines.
    * Sections are merged if they share the same title and page, consolidating content.

3.  **Relevance Scoring and Ranking:**
    * All extracted sections, along with the user's query (persona, job-to-be-done, and extracted keywords), are converted into numerical representations (embeddings) using a **BERT-Tiny model** with mean pooling.
    * **Cosine similarity** is used to measure how relevant each section's content is to the user's query.
    * **Domain Boost**: Based on persona and job-to-be-done, we extract more relevant words related to the same under the name of **"Domain Terms"**.
    * Relevance scores are **boosted** based on several factors:
        * **Keyword Matches:** Sections containing more context keywords and domain terms receive higher boosts.
        * **Title Quality:** Boosts sections with titles related to common action words (e.g., 'form', 'manage').
        * **Contextual Domain Boost:** Additional boost for sections containing specific domain terms, thus improving the importance rank for the given context.
    * The sections are then ranked in descending order of their boosted scores.

4.  **Key Sentence Extraction:**
    * For the top-ranked sections, the system extracts the most relevant sentences.
    * Sentences are tokenized using NLTK, and irrelevant sentences (too short, warnings, etc.) are filtered.
    * Similar to section ranking, sentences are converted to embeddings, and their similarity to the query (with keywords and domain terms) is calculated.
    * **Sentence similarities are boosted** if they contain context keywords or action-oriented words, ensuring the most actionable and relevant sentences are selected.
    * The top sentences are combined to provide a concise, refined summary of the section.

## Folder Structure

```
APIcalypse-Round-1b
├── documentation and demo/           #You will find the documentation, demo video and pipeline of our solution here
├── input/
│   ├── PDFs/                         # Place your PDF documents here
│   └── input.json                    # User persona and job-to-be-done configuration
├── output/                           # Generated JSON output will be saved here
├── sample cases and outputs/ 
├── output/ 
├── src/                              # All the brains of the operation live here!
|   ├── pretrained_models_bert_tiny/  # Pre-trained BERT-Tiny model files
│   ├── models/
│   │   └── layoutlmv3/               # Pre-trained LayoutLMv3 model files/
│   └── model.py                      # Main script to run the analysis        
├── Dockerfile                        # Docker setup file
├── requirements.txt                  # Python dependencies
└── README.md                         # This README file
```

## Setup & Running the Extractor (Docker)

We leverage Docker to make setup and execution incredibly simple and portable.

### Build Docker Image

First, build the Docker image from the project's root directory:

```bash
docker build -t apicalypse-1b:latest .
```

### Prepare Input

1.  Inside `input`, place your `.pdf` documents in the `PDFs` directory. (sample input has been provided)

2.  Also inside `input`, update the `input.json` file with your `persona` and `job_to_be_done` details.

    **Example `input.json`:**

    ```json
    {
      "persona": "HR Manager",
      "job_to_be_done": "find information on employee onboarding process and compliance documents"
    }
    ```

### Run Docker Container

Execute the Docker container. This command mounts your local `input` and `output` directories to the container, allowing it to read your PDFs and save the results.

```bash
docker run --rm -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" --network none apicalypse-1b:latest
```

The processed output (a JSON file) will appear in the `output/` folder.

## Output Format

The system generates a `output.json` file in the `output` directory with the following structure:

```json
{
  "metadata": {
    "input_documents": [
      "Learn Acrobat - Fill and Sign.pdf",
      "Learn Acrobat - Request e-signatures_1.pdf"
    ],
    "persona": "HR professional",
    "job_to_be_done": "Create and manage fillable forms for onboarding and compliance.",
    "processing_timestamp": "2025-07-28T16:59:31.123456",
  },
  "extracted_sections": [
    {
      "document": "Learn Acrobat - Fill and Sign.pdf",
      "section_title": "Change flat forms to fillable (Acrobat Pro)",
      "importance_rank": 1,
      "page_number": 11,
    },
    {
      "document": "Learn Acrobat - Request e-signatures_1.pdf",
      "section_title": "Set privileges and permissions for others",
      "importance_rank": 2,
      "page_number": 20,
    },
   ],
  "sub_section_analysis": [
    {
      "document": "Learn Acrobat - Fill and Sign.pdf",
      "section_title": "Change flat forms to fillable (Acrobat Pro)",
      "refined_text": "When you create a different form, redo this task to enable Acrobat Reader users to use the tools. You can change a flat form to a fillable form by either using the Prepare Form tool or enabling the Fill & Sign tools. To enable the Fill & Sign tools, from the hamburger menu choose Save As Other > Acrobat Reader Extended PDF > Enable More Tools.",
      "page_number": 11
    },
    {
      "document": "Learn Acrobat - Request e-signatures_1.pdf",
      "section_title": "Set privileges and permissions for others",
      "refined_text": "Consider signing up for Adobe Acrobat Sign online. Set privileges and permissions for others: Certify a document while leaving portions of it available for form filling, signatures, or comments. Use Acrobat Pro to enable users of Reader 9 or later to sign with certificate IDs. To sign a document with a certificate-based signature, you must obtain a digital ID.",
      "page_number": 20
    },
   ]
}
```
## Performance

Our system is optimized for efficiency and effectiveness:

| Metric | Result |
| :------------------- | :------------------------------------------ |
| **Execution Time** | **Exceptional performance:** Optimized for speed, typically processing documents within minutes depending on complexity and length. |
| **Model Size** | BERT-Tiny (~18MB), LayoutLMv3 (974MB for base model) |
| **Network Dependency** | **None** (completely offline after initial model download) |
| **Architecture** | `linux/amd64` (Docker compatible) |

## PDFs Handled (Types/Structure)

This system is designed for broad applicability and can effectively process a wide range of PDF documents, including those with:

  * Standard text-based layouts
  * Complex, multi-column designs
  * Documents containing a mix of text and images (LayoutLMv3 excels here)
  * Scanned documents (utilizing OCR fallback where necessary, though direct image processing via LayoutLMv3 is preferred for visual understanding)
  * Business reports, academic papers, manuals, legal documents, etc.

## Models and Libraries/Dataset

Our solution is powered by a robust stack of modern technologies:

  * **Core NLP Models:**
      * **LayoutLMv3ForTokenClassification:** Used for robust heading detection by understanding both text and visual layout.
      * **BERT-Tiny:** A compact BERT model used for generating embeddings for text and queries, enabling semantic similarity calculations.
      * **BertTokenizer:** Tokenizes text for BERT models.
  * **Text Extraction:** `pdfplumber` for structured text and character data extraction.
  * **Image Processing:** `PIL (Pillow)` for handling page images for LayoutLMv3.
  * **Scientific Computing:** `numpy` for numerical operations.
  * **Machine Learning Utilities:** `scikit-learn` for `cosine_similarity` and `KMeans` clustering.
  * **Natural Language Processing:** `nltk` for sentence tokenization and POS tagging to refine keyword extraction.
  * **Deep Learning Framework:** `PyTorch` for model inference.
  * **Hugging Face Transformers:** Provides the interface for BERT-Tiny and LayoutLMv3 models.

## Testing

The system is tested to ensure accurate:

  * **Domain Term Extraction:** Verifying that relevant terms are identified from the persona and job-to-be-done.
  * **Context Keyword Generation:** Ensuring meaningful keywords are derived for boosting relevance.
  * **Heading Detection:** Confirming LayoutLMv3 correctly identifies section titles.
  * **Section Extraction:** Validating that text is correctly associated with its respective section.
  * **Relevance Ranking:** Checking that sections most pertinent to the query are ranked highest.
  * **Key Sentence Extraction:** Ensuring that the most informative sentences from top sections are summarized effectively.

## Team

  * [Nandini Nema](https://www.linkedin.com/in/nandininema/)
  * [Soham Chandane](https://www.linkedin.com/in/sohamchandane/)
  * [Parv Siria](https://www.linkedin.com/in/parv-siria/)

## Conclusion

This **Context-Aware PDF Document Intelligence System (Round 1B)** represents a significant leap in extracting actionable insights from PDFs. By deeply understanding user intent through persona and job-to-be-done, combined with state-of-the-art layout-aware AI for section identification and a sophisticated ranking mechanism, we transform static documents into dynamic, queryable knowledge bases. This system is a powerful module poised to revolutionize applications requiring intelligent document navigation, personalized information retrieval, and automated content summarization.