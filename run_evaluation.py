import os
import sys
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

def evaluate_new_dataset(test_file_path="test_sa.csv", output_file_path="submission.csv"):
    """
    Classroom Evaluation Engine:
    Loads a newly provided Sanskrit test file, routes to GPU if available,
    and exports a structurally valid, UTF-8 encoded evaluation submission file.
    """
    if not os.path.exists(test_file_path):
        print(f"Error: Evaluation file '{test_file_path}' not found.")
        return

    # 1. Setup hardware target acceleration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Routing evaluation execution space to: {device}")
    
    # 2. Read evaluation target records
    df = pd.read_csv(test_file_path)
    source_ids = df["Source_id"].tolist()
    sanskrit_sentences = df["Sentence_sa"].tolist()
    translations = []
    
    # 3. Initialize Model Framework from disclosed checkpoints
    MODEL_NAME = "facebook/mbart-large-50"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.src_lang = "sa_IN"
    tokenizer.tgt_lang = "en_XX"
    
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    
    # 4. Run High-Speed Mini-Batch Generation
    batch_size = 32
    print(f"Processing {len(sanskrit_sentences)} lines in parallel mini-batches...")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sanskrit_sentences), batch_size)):
            batch_texts = sanskrit_sentences[i : i + batch_size]
            
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=128
            ).to(device)
            
            generated_ids = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=128, 
                num_beams=4,
                early_stopping=True
            )
            
            batch_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            translations.extend(batch_outputs)
            
    # 5. Formulate final submission structure strictly matching grading schema
    submission_df = pd.DataFrame({
        "Source_id": source_ids,
        "Sentence_en": translations
    })
    
    submission_df.to_csv(output_file_path, index=False, encoding="utf-8")
    print(f"Evaluation Complete! Output file written to: {output_file_path}")

if __name__ == "__main__":
    # If a file is passed via terminal arguments, use it; otherwise default to test_sa.csv
    target_file = sys.argv[1] if len(sys.argv) > 1 else "test_sa.csv"
    evaluate_new_dataset(target_file)
