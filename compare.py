import pandas as pd
import nltk
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt')

# Initialize CodeBERT
model = SentenceTransformer('microsoft/codebert-base')

# Load Excel
df = pd.read_excel('responses_comparison_three_with_code_modified2.xlsx', sheet_name=None, header=0)  # all sheets, header=0 for first row as column names

# BLEU
def calculate_bleu(reference, hypothesis):
    reference = str(reference)
    hypothesis = str(hypothesis)
    reference_tokens = nltk.word_tokenize(reference)
    hypothesis_tokens = nltk.word_tokenize(hypothesis)
    return nltk.translate.bleu_score.sentence_bleu([reference_tokens], hypothesis_tokens)

# ROUGE
def calculate_rouge(reference, hypothesis):
    reference = str(reference)
    hypothesis = str(hypothesis)
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores['rouge1'].fmeasure

# CodeBERT Similarity
def calculate_codebert_similarity(reference, hypothesis):
    reference = str(reference)
    hypothesis = str(hypothesis)
    ref_embedding = model.encode([reference])
    hyp_embedding = model.encode([hypothesis])
    similarity = cosine_similarity(ref_embedding, hyp_embedding)
    return similarity[0][0]

# Compute metrics per sheet
def compute_metrics_for_task(task_df):
    results = {
        "task": [],
        # Inter-model comparisons
        "bleu_zero_shot": [],
        "rouge_zero_shot": [],
        "codebert_zero_shot": [],
        "bleu_cot": [],
        "rouge_cot": [],
        "codebert_cot": [],
        # Intra-model comparisons
        "bleu_openai_zerovsCot": [],
        "rouge_openai_zerovsCot": [],
        "codebert_openai_zerovsCot": [],
        "bleu_gemini_zerovsCot": [],
        "rouge_gemini_zerovsCot": [],
        "codebert_gemini_zerovsCot": []
    }

    # Strip any extra spaces in column names
    task_df.columns = task_df.columns.str.strip()

    # Debugging: Print the column names to check
    print(f"Columns in the current sheet: {task_df.columns.tolist()}")

    # Locate the rows corresponding to Zero Shot and Chain of Thought Prompts
    zero_shot_prompt_row = task_df[task_df["Unnamed: 0"] == "Zero Shot Prompt"]
    cot_prompt_row = task_df[task_df["Unnamed: 0"] == "Chain of Thought Prompt"]

    if zero_shot_prompt_row.empty or cot_prompt_row.empty:
        print("Missing data for Zero Shot Prompt or Chain of Thought Prompt.")
        return results

    # Access the columns with the code outputs
    openai_code_output_col = "OpenAI  Code Output"
    gemini_code_output_col = "Gemini  Code Output"

    for i, row in zero_shot_prompt_row.iterrows():
        try:
            # Fetch outputs for Zero Shot Prompt from columns
            openai_zero = row[openai_code_output_col]
            gemini_zero = row[gemini_code_output_col]
            
            # Now get the outputs for the Chain of Thought Prompt (from cot_prompt_row)
            cot_row = cot_prompt_row.iloc[0]  # we only have one row for the chain of thought
            openai_cot = cot_row[openai_code_output_col]
            gemini_cot = cot_row[gemini_code_output_col]

            # Check for missing values
            if any(pd.isna(val) for val in [openai_zero, gemini_zero, openai_cot, gemini_cot]):
                print(f"Skipping row {i} due to missing code output.")
                continue

            # Convert all to strings
            openai_zero = str(openai_zero)
            gemini_zero = str(gemini_zero)
            openai_cot = str(openai_cot) if openai_cot else ""
            gemini_cot = str(gemini_cot) if gemini_cot else ""

            # Inter-model comparisons
            bleu_zero = calculate_bleu(openai_zero, gemini_zero)
            rouge_zero = calculate_rouge(openai_zero, gemini_zero)
            codebert_zero = calculate_codebert_similarity(openai_zero, gemini_zero)

            bleu_cot = calculate_bleu(openai_cot, gemini_cot)
            rouge_cot = calculate_rouge(openai_cot, gemini_cot)
            codebert_cot = calculate_codebert_similarity(openai_cot, gemini_cot)

            # Intra-model (OpenAI)
            bleu_openai_internal = calculate_bleu(openai_zero, openai_cot)
            rouge_openai_internal = calculate_rouge(openai_zero, openai_cot)
            codebert_openai_internal = calculate_codebert_similarity(openai_zero, openai_cot)

            # Intra-model (Gemini)
            bleu_gemini_internal = calculate_bleu(gemini_zero, gemini_cot)
            rouge_gemini_internal = calculate_rouge(gemini_zero, gemini_cot)
            codebert_gemini_internal = calculate_codebert_similarity(gemini_zero, gemini_cot)

            # Store results
            results["task"].append(row["Unnamed: 0"])

            results["bleu_zero_shot"].append(bleu_zero)
            results["rouge_zero_shot"].append(rouge_zero)
            results["codebert_zero_shot"].append(codebert_zero)

            results["bleu_cot"].append(bleu_cot)
            results["rouge_cot"].append(rouge_cot)
            results["codebert_cot"].append(codebert_cot)

            results["bleu_openai_zerovsCot"].append(bleu_openai_internal)
            results["rouge_openai_zerovsCot"].append(rouge_openai_internal)
            results["codebert_openai_zerovsCot"].append(codebert_openai_internal)

            results["bleu_gemini_zerovsCot"].append(bleu_gemini_internal)
            results["rouge_gemini_zerovsCot"].append(rouge_gemini_internal)
            results["codebert_gemini_zerovsCot"].append(codebert_gemini_internal)

        except KeyError as e:
            print(f"Missing column in row {i}: {e}")
            continue

    return results

# Aggregate across sheets
metrics_results = {
    "task": [],
    "bleu_zero_shot": [],
    "rouge_zero_shot": [],
    "codebert_zero_shot": [],
    "bleu_cot": [],
    "rouge_cot": [],
    "codebert_cot": [],
    "bleu_openai_zerovsCot": [],
    "rouge_openai_zerovsCot": [],
    "codebert_openai_zerovsCot": [],
    "bleu_gemini_zerovsCot": [],
    "rouge_gemini_zerovsCot": [],
    "codebert_gemini_zerovsCot": []
}

for sheet_name, sheet_data in df.items():
    print(f"Processing: {sheet_name}")
    task_results = compute_metrics_for_task(sheet_data)
    for key in metrics_results:
        metrics_results[key].extend(task_results[key])

# Save results
metrics_df = pd.DataFrame(metrics_results)
metrics_df.to_excel("model_comparison_full_metrics.xlsx", index=False)
print("Done: Results saved to 'model_comparison_full_metrics.xlsx'")
