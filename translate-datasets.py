import sys
import torch
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer, pipeline
from datasets import Dataset
from tqdm import tqdm

def translate_batch(examples, translator):
    texts = examples["search_term"]
    translations = translator(texts, max_length=128, batch_size=len(texts))
    translated_texts = [t["translation_text"] for t in translations]
    return {"translated_search_term": translated_texts}

def main():
    if len(sys.argv) < 4:
        print("Usage: python translate_dataset_filter.py <input_csv> <output_csv> <target_language_code>")
        sys.exit(1)
        
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    target_lang = sys.argv[3]  # e.g., 'de' for German
    
    # We assume rows with locale_language starting with "en" are English.
    model_name = f"Helsinki-NLP/opus-mt-en-{target_lang}"
    
    print("Loading model and tokenizer...")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    if torch.cuda.is_available():
        model.to("cuda")
        device = 0
        print("Using GPU for translation.")
    else:
        device = -1
        print("Using CPU for translation.")
    
    translator = pipeline("translation", model=model, tokenizer=tokenizer, device=device)
    
    print("Loading CSV with pandas (skipping bad lines)...")
    # Use pandas to read the CSV and skip lines that don't parse properly.
    df = pd.read_csv(input_csv, sep=";", on_bad_lines="skip")
    
    # Filter out rows that are not English.
    df = df[df["locale_language"].str.lower().str.startswith("en")].copy()
    total_rows = len(df)
    print(f"Total English rows to translate: {total_rows}")
    
    # Convert the DataFrame to a Hugging Face Dataset.
    ds = Dataset.from_pandas(df)
    
    print("Translating dataset in batches...")
    # Use map() with batched processing. (This may take a while on a large dataset.)
    ds = ds.map(lambda examples: translate_batch(examples, translator),
                batched=True, batch_size=32, load_from_cache_file=False)
    
    # Add a new column for the target locale.
    ds = ds.add_column("target_locale", [target_lang] * len(ds))
    
    # Create a pandas DataFrame with the desired columns.
    out_df = ds.to_pandas()[["translated_search_term", "product", "search_frequency", "target_locale"]]
    
    print("Saving output CSV...")
    out_df.to_csv(output_csv, sep=";", index=False)
    print("Translation complete. Output written to", output_csv)

if __name__ == "__main__":
    main()
