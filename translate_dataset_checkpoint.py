import sys
import os
import torch
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer, pipeline
from datasets import load_dataset
from tqdm import tqdm

def translate_batch(examples, translator):
		texts = examples["search_term"]
		# Translate the batch (adjust max_length as needed)
		translations = translator(texts, max_length=128, batch_size=len(texts))
		translated_texts = [t["translation_text"] for t in translations]
		return {"translated_search_term": translated_texts}

def main():
		if len(sys.argv) < 4:
				print("Usage: python translate_dataset_checkpoint.py <input_csv> <output_csv> <target_language_code>")
				sys.exit(1)
				
		input_csv = sys.argv[1]
		output_csv = sys.argv[2]
		target_lang = sys.argv[3]  # e.g., 'de' for German

		# For this example, we assume that rows with locale_language starting with "en" are English.
		# We'll use a MarianMT model for English-to-target translation.
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
		
		print("Loading dataset from CSV...")
		# Load CSV as a dataset (delimiter is semicolon). Use pandas engine with on_bad_lines skip if needed.
		try:
				dataset = load_dataset("csv", data_files=input_csv, delimiter=";")
		except Exception as e:
				print("Error loading dataset with load_dataset:", e)
				sys.exit(1)
		
		ds = dataset["train"]
		# Filter: Keep only rows with locale_language starting with "en"
		ds = ds.filter(lambda example: example["locale_language"].lower().startswith("en"))
		
		total_rows = ds.num_rows
		print(f"Total English rows to translate: {total_rows}")
		
		# Set up checkpointing: we want to save every 10% of progress.
		checkpoint_interval = total_rows / 10
		next_checkpoint = checkpoint_interval
		processed = 0
		results_list = []
		chunk_size = 32  # Adjust as needed
		
		# Remove output file if it exists.
		if os.path.exists(output_csv):
				os.remove(output_csv)
		
		# Process the dataset in chunks.
		print("Translating dataset in batches with checkpointing...")
		for start in tqdm(range(0, total_rows, chunk_size), desc="Processing batches"):
				end = min(start + chunk_size, total_rows)
				batch = ds.select(range(start, end))
				processed_batch = batch.map(lambda examples: translate_batch(examples, translator),
																		batched=True, batch_size=chunk_size, load_from_cache_file=False)
				df_chunk = processed_batch.to_pandas()[["translated_search_term", "product", "search_frequency"]]
				df_chunk["target_locale"] = target_lang
				results_list.append(df_chunk)
				
				processed += (end - start)
				
				# When we reach a checkpoint, save accumulated results.
				if processed >= next_checkpoint:
						df_save = pd.concat(results_list, ignore_index=True)
						# Write header only if file does not exist yet.
						header = not os.path.exists(output_csv)
						df_save.to_csv(output_csv, sep=";", mode="a", index=False, header=header)
						results_list = []  # Clear the accumulator.
						print(f"Checkpoint saved: {processed/total_rows*100:.1f}% completed.")
						next_checkpoint += checkpoint_interval
		
		# Save any remaining results.
		if results_list:
				df_save = pd.concat(results_list, ignore_index=True)
				header = not os.path.exists(output_csv)
				df_save.to_csv(output_csv, sep=";", mode="a", index=False, header=header)
		
		print("Translation complete. Output written to", output_csv)

if __name__ == "__main__":
		main()