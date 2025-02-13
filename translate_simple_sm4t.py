import sys
import torch
import pandas as pd
from transformers import pipeline

def main():
		if len(sys.argv) < 4:
				print("Usage: python translate_simple_sm4t.py <input_csv> <output_csv> <target_language_code>")
				sys.exit(1)
				
		input_csv = sys.argv[1]
		output_csv = sys.argv[2]
		target_lang = sys.argv[3]  # e.g., 'de' for German
		
		# For SeamlessM4T v2, we assume the model identifier is "facebook/seamless_m4t_v2".
		# This model is a unified multilingual model that supports text translation.
		model_name = "facebook/seamless_m4t_v2"
		
		print("Loading SeamlessM4T v2 model and tokenizer...")
		# When using SM4T v2, we pass extra parameters to specify the source and target languages.
		# (Ensure that your Transformers version supports these parameters.)
		if torch.cuda.is_available():
				device = 0
				print("Using GPU for translation.")
		else:
				device = -1
				print("Using CPU for translation.")
		
		translator = pipeline(
				"translation",
				model=model_name,
				tokenizer=model_name,
				device=device,
				# Specify the source and target languages for the translation.
				src_lang="en",
				tgt_lang=target_lang,
		)
		
		print("Reading input CSV...")
		df = pd.read_csv(input_csv, sep=';')
		
		# Filter only rows that are English (i.e. locale_language starts with "en")
		df_en = df[df["locale_language"].str.lower().str.startswith("en")].copy()
		if df_en.empty:
				print("No English rows found in the input CSV.")
				sys.exit(1)
				
		print(f"Translating {len(df_en)} English rows...")
		texts = df_en["search_term"].tolist()
		# Process texts in batches; adjust max_length and batch_size as needed.
		translations = translator(texts, max_length=128, batch_size=32)
		translated_texts = [t["translation_text"] for t in translations]
		df_en["translated_search_term"] = translated_texts

		# Prepare output DataFrame with the desired columns.
		output_df = df_en[["translated_search_term", "product", "search_frequency"]].copy()
		output_df["target_locale"] = target_lang

		print("Saving output CSV...")
		output_df.to_csv(output_csv, sep=";", index=False)
		print("Translation complete. Output written to", output_csv)

if __name__ == "__main__":
		main()