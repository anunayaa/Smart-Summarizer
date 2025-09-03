# Smart-Summarizer
A BART-based text summarization system that preprocesses, analyzes, and condenses large text datasets into clear, concise summaries. Includes data analysis, evaluation metrics, and examples with scope for future enhancements.

Project Overview

SmartSummarizer is a deep learning project that leverages BART (Bidirectional and Auto-Regressive Transformers) for abstractive text summarization.
The model generates concise, coherent summaries while retaining the original text's meaning, making it useful for tasks like news summarization, research paper condensation, and information retrieval automation.

Dataset

Source: Custom dataset of 44,000+ article–summary pairs.

Due to computational limitations, a sample of 5,000 records was used for fine-tuning.

Preprocessing steps:

Tokenization with BART tokenizer.

Truncation/padding of inputs (512 tokens) and summaries (128 tokens).

Removal of null/duplicate records.

⚙️ Methodology

Data Preprocessing

Tokenization, cleaning, and splitting into train/validation sets.

Model Training

Fine-tuned BART model using Hugging Face transformers.

Training executed on GPU-enabled Colab with reduced dataset size.


🖥️ How to Run
Prerequisites

Install dependencies:

pip install transformers datasets torch

Training
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments


tokenizer = BartTokenizer.from_pretrained("./bart_summary")
model = BartForConditionalGeneration.from_pretrained("./bart_summary")

text = "Your input text goes here..."
inputs = tokenizer([text], max_length=512, truncation=True, return_tensors="pt")
summary_ids = model.generate(inputs["input_ids"], max_length=128, min_length=30, length_penalty=2.0)
print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))

📊 Results & Achievements

Successfully implemented and fine-tuned BART for abstractive summarization.

Achieved high summarization performance

Built a full pipeline: preprocessing → training → evaluation → inference.

Demonstrated summarization on real-world articles.

🔮 Future Enhancements

Train on the full 44k+ dataset for better generalization.

Experiment with PEGASUS, T5, and Longformer for performance comparison.

Deploy as a web application or REST API.

Add ROUGE evaluation and human feedback mechanisms.

- Author

- Anunaya R Pillai

