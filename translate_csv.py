import pandas as pd
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer, pipeline
import torch
import sys
import os

def main():
    if len(sys.argv) < 4:
        print("Usage: python translate_csv.py <input_csv> <output_csv> <target_language_code>")
        sys.exit(1)
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    target_lang = sys.argv[3]  # e.g., 'de' for German

    # For this example, we assume the input language is English.
    model_name = f"Helsinki-NLP/opus-mt-en-{target_lang}"
    
    print("Loading model and tokenizer...")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Use GPU if available
    if torch.cuda.is_available():
        model.to("cuda")
        device = 0
        print("Using GPU for translation.")
    else:
        device = -1
        print("Using CPU for translation.")
    
    # Create a translation pipeline
    translator = pipeline("translation", model=model, tokenizer=tokenizer, device=device)

    # Determine total lines (minus header) for progress bar
    with open(input_csv, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f) - 1

    # Prepare to read CSV in chunks (500 lines per chunk)
    chunk_size = 500
    reader = pd.read_csv(input_csv, sep=';', chunksize=chunk_size)

    # If output file exists, remove it
    if os.path.exists(output_csv):
        os.remove(output_csv)

    # Write output header once
    header_written = False

    print("Translating...")
    with tqdm(total=total_lines, desc="Translating", unit="line") as pbar:
        for chunk in reader:
            # Expecting columns: search_term, product, search_frequency, locale_language
            # Skip rows that already are in the target language (optional)
            mask = chunk["locale_language"] != target_lang
            to_translate = chunk.loc[mask, "search_term"].tolist()
            # For rows that don't need translation, keep them as-is
            unchanged = chunk.loc[~mask, "search_term"].tolist()

            # Translate only rows that need it.
            if to_translate:
                results = translator(to_translate, max_length=128)
                translated = [res['translation_text'] for res in results]
            else:
                translated = []

            # Combine: if a row did not need translation, keep original search_term
            final_search_terms = []
            i, j = 0, 0
            for idx in chunk.index:
                if chunk.loc[idx, "locale_language"] != target_lang:
                    final_search_terms.append(translated[i])
                    i += 1
                else:
                    final_search_terms.append(chunk.loc[idx, "search_term"])
                    j += 1

            # Add the translated search term and target locale to the chunk.
            chunk["translated_search_term"] = final_search_terms
            chunk["target_locale"] = target_lang

            # Select desired columns for output.
            out_chunk = chunk[["translated_search_term", "product", "search_frequency", "target_locale"]]

            # Append to output CSV.
            if not header_written:
                out_chunk.to_csv(output_csv, sep=';', index=False, mode='w')
                header_written = True
            else:
                out_chunk.to_csv(output_csv, sep=';', index=False, mode='a', header=False)

            pbar.update(len(chunk))

    print("Translation complete. Output written to", output_csv)

if __name__ == "__main__":
    main()
