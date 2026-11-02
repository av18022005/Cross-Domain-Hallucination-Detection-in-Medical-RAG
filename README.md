# Cross-Domain Hallucination Detection in Medical RAG

## 1. Project Overview

This project focuses on detecting hallucinations in Medical Retrieval-Augmented Generation (RAG) systems.

Medical RAG systems retrieve information from medical documents and use that information to generate answers to medical questions. However, generated answers may contain incorrect, unsupported, or misleading medical information.

This project develops a medical hallucination detection pipeline that identifies unsupported statements in generated answers by comparing them against retrieved medical evidence.

The project also explores whether adapting a pretrained hallucination detection model to the medical domain improves its performance.

## 2. Objectives

The main objectives are:

- Build a medical information retrieval pipeline.
- Retrieve relevant medical evidence for medical questions.
- Generate answers using a Retrieval-Augmented Generation approach.
- Detect hallucinations in generated medical answers.
- Evaluate the original pretrained LettuceDetect model.
- Create human-reviewed medical hallucination annotations.
- Fine-tune LettuceDetect using medical-domain data.
- Compare the original and medical-adapted models using evaluation metrics.

## 3. Technologies and Tools Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Google Colab / Kaggle | Notebook execution environment |
| Pandas | Dataset processing |
| NumPy | Numerical operations |
| PyTorch | Deep learning |
| Hugging Face Transformers | Pretrained language models and fine-tuning |
| Sentence Transformers | Text embeddings |
| FAISS | Vector similarity search |
| BM25 | Keyword-based information retrieval |
| MedCPT | Biomedical information retrieval |
| ModernBERT | Backbone for LettuceDetect |
| LettuceDetect | Hallucination detection |
| Scikit-learn | Model evaluation |
| Matplotlib | Data visualization |

## 4. Dataset

The project uses PubMedQA-based medical question-answering data.

The first notebook works with 1,000 examples and develops the medical retrieval and RAG pipeline.

The second notebook processes 200 RAG examples for hallucination detection and annotation.

For the final medical-domain fine-tuning experiment, the dataset is divided into:

- Training: 51 examples
- Validation: 13 examples
- Final testing: 40 examples

The final test set contains 40 human-reviewed examples, including 17 examples with hallucinated spans and 26 annotated hallucination spans.

## 5. Project Architecture

The project consists of two major stages.

### Stage 1: Medical RAG Pipeline

1. Load PubMedQA dataset.
2. Prepare the medical retrieval corpus.
3. Generate text embeddings.
4. Build a FAISS retrieval index.
5. Implement BM25 retrieval.
6. Implement MedCPT retrieval.
7. Combine BM25 and MedCPT using hybrid retrieval.
8. Select relevant medical evidence.
9. Generate answers using the RAG pipeline.
10. Save the generated RAG results.

### Stage 2: Medical Hallucination Detection

1. Load the generated medical RAG dataset.
2. Prepare the input for LettuceDetect.
3. Load the original pretrained LettuceDetect model.
4. Generate baseline hallucination predictions.
5. Create human-reviewed annotations.
6. Prepare training, validation and test datasets.
7. Fine-tune LettuceDetect using medical-domain annotations.
8. Evaluate the adapted model.
9. Compare baseline and fine-tuned model performance.
10. Perform error analysis.

## 6. Methodology

### 6.1 Medical Information Retrieval

Different retrieval techniques are explored:

- FAISS with sentence embeddings.
- BM25 keyword-based retrieval.
- MedCPT biomedical retrieval.
- Hybrid BM25 + MedCPT retrieval.

The hybrid retrieval approach combines keyword matching and biomedical semantic retrieval to identify relevant medical evidence.

### 6.2 Retrieval-Augmented Generation

The retrieved medical evidence is provided as context for answer generation.

The system generates answers based on the medical question and retrieved evidence.

### 6.3 Hallucination Detection

LettuceDetect is used to identify potentially hallucinated portions of generated answers.

The model performs token-level classification to identify text that may be unsupported by the retrieved evidence.

### 6.4 Human Annotation

Human-reviewed annotations are used to identify hallucinated spans.

The annotations distinguish between:
- Hallucinated content.
- Partially supported answers.
- Appropriate abstentions.

### 6.5 Medical Domain Fine-Tuning

The pretrained LettuceDetect model is fine-tuned using the medically annotated training dataset.

The training process uses:
- 51 training examples.
- 13 validation examples.
- 40 held-out test examples.

A prediction threshold of 0.50 is selected using validation data and frozen for final evaluation.

## 7. Model Evaluation

The models are evaluated using:

- Precision
- Recall
- F1-score
- Accuracy

The final comparison uses the same 40 held-out test examples and the same human-reviewed ground truth.

### Final Results

| Metric | Original LettuceDetect | Medical Fine-Tuned |
|---|---:|---:|
| Span Precision | 0.0000 | 0.2267 |
| Span Recall | 0.0000 | 0.6538 |
| Span F1-score | 0.0000 | 0.3366 |
| Example F1-score | 0.0000 | 0.8108 |
| Accuracy | 0.5250 | 0.8250 |

The medical fine-tuned model demonstrates improvement over the original pretrained model on the final held-out test set.

However, span-level precision remains relatively low, indicating that the model still produces false-positive hallucination predictions.

## 8. Error Analysis

The project also investigates errors made by the medical-adapted model.

The analysis considers:

- Clinical negation.
- Medical terminology.
- Contradictory statements.
- Unsupported clinical claims.
- Numerical and dosage-related information.
- Partially supported medical statements.

This helps identify areas where hallucination detection requires further improvement.

## 9. How to Run the Project

### Requirements

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib
pip install torch transformers
pip install sentence-transformers
pip install faiss-cpu rank-bm25
pip install scikit-learn
```

### Execution

1. Open the first notebook.
2. Install the required dependencies.
3. Load the PubMedQA dataset.
4. Execute the medical retrieval pipeline.
5. Generate and save the RAG results.
6. Open the continuation notebook.
7. Load the generated RAG dataset.
8. Run the original LettuceDetect model.
9. Prepare and verify human annotations.
10. Fine-tune the model.
11. Run the final evaluation and comparison.

Note: The continuation notebook depends on the dataset generated by the first notebook. Model checkpoints and datasets may also need to be restored or regenerated in the execution environment.

## 10. Limitations

- The human-reviewed training dataset is relatively small.
- Medical terminology and clinical negation can be difficult to classify.
- The model may identify supported statements as hallucinations.
- Span-level precision requires further improvement.
- The system should not be used as a replacement for professional medical verification.

## 11. Future Enhancements

- Increase the size of the human-annotated medical dataset.
- Improve biomedical retrieval accuracy.
- Experiment with additional medical language models.
- Improve hallucination span identification.
- Reduce false-positive predictions.
- Evaluate the system on larger independent medical datasets.

## 12. Conclusion

This project demonstrates a complete pipeline for medical hallucination detection using Retrieval-Augmented Generation and domain-adapted hallucination detection.

It investigates the limitations of pretrained hallucination detection models in medical contexts and demonstrates how medical-domain fine-tuning can improve example-level detection performance.

The results highlight the potential of domain adaptation while identifying the need for further improvements in precise hallucination span detection.
