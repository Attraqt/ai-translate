import sys
import torch
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer, pipeline
from datasets import load_dataset
from tqdm import tqdm

def translate_batch(examples, translator):
		# This function assumes that all examples in the batch are English.
		texts = examples["search_term"]
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

		# Use a MarianMT model for English-to-target translation.
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
		
		print("Loading dataset...")
		# Load CSV as a dataset; note delimiter is a semicolon.
		dataset = load_dataset("csv", data_files=input_csv, delimiter=";")
		ds = dataset["train"]
		
		# Filter to keep only rows with locale_language starting with "en"
		ds = ds.filter(lambda example: example["locale_language"].lower().startswith("en"))
		
		total_rows = ds.num_rows
		print(f"Total English rows to translate: {total_rows}")
		
		batch_size = 32  # adjust as needed
		checkpoint_interval = total_rows / 10  # save every 10%
		next_checkpoint = checkpoint_interval
		processed = 0
		results_list = []
		
		# Prepare output file by removing it if it exists.
		try:
				open(output_csv, "w").close()
		except Exception as e:
				print(f"Error preparing output file: {e}")
				sys.exit(1)
		
		# Iterate over the dataset in batches.
		print("Translating dataset in batches and saving every 10% progress...")
		for start in tqdm(range(0, total_rows, batch_size), desc="Processing batches"):
				end = min(start + batch_size, total_rows)
				batch = ds.select(range(start, end))
				# Process translation on this batch.
				processed_batch = batch.map(lambda examples: translate_batch(examples, translator),
																		batched=True, batch_size=batch_size, load_from_cache_file=False)
				# Convert batch to pandas DataFrame.
				df_chunk = processed_batch.to_pandas()[["translated_search_term", "product", "search_frequency"]]
				df_chunk["target_locale"] = target_lang
				results_list.append(df_chunk)
				
				processed += (end - start)
				# Check if we've passed the next checkpoint.
				if processed >= next_checkpoint:
						# Concatenate results and append to CSV.
						df_save = pd.concat(results_list, ignore_index=True)
						# Write header only if file is empty.
						header = (processed - (end - start)) == 0  # if first checkpoint
						df_save.to_csv(output_csv, sep=";", index=False, mode="a", header=header)
						results_list = []  # clear stored chunks
						print(f"Progress saved: {processed/total_rows*100:.1f}%")
						next_checkpoint += checkpoint_interval
		
		# After finishing, if any remaining results, append them.
		if results_list:
				df_save = pd.concat(results_list, ignore_index=True)
				df_save.to_csv(output_csv, sep=";", index=False, mode="a", header=False)
		
		print("Translation complete. Output written to", output_csv)

if __name__ == "__main__":
		main()