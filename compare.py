import pandas as pd
import nltk
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources (BLEU requires them)
nltk.download('punkt')
nltk.download('punkt_tab')

# Initialize SentenceTransformer for CodeBERT embeddings
model = SentenceTransformer('microsoft/codebert-base')

# Load the responses from the file
df = pd.read_excel('responses_comparison_two.xlsx', sheet_name=None)  # Load all sheets into a dict

# Debugging: Print the sheet names to ensure the correct file is loaded
print("Loaded sheets:", df.keys())

# Define BLEU Score Calculation (for standard text BLEU)
def calculate_bleu(reference, hypothesis):
    reference_tokens = nltk.word_tokenize(reference)  # Tokenize by word
    hypothesis_tokens = nltk.word_tokenize(hypothesis)
    return nltk.translate.bleu_score.sentence_bleu([reference_tokens], hypothesis_tokens)

# Define ROUGE Score Calculation (using rouge-score package)
def calculate_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure

# Define CodeBERT Embedding Similarity Calculation
def calculate_codebert_similarity(reference, hypothesis):
    ref_embedding = model.encode([reference])
    hyp_embedding = model.encode([hypothesis])
    similarity = cosine_similarity(ref_embedding, hyp_embedding)
    return similarity[0][0]

# Define a function to calculate metrics for each task
def compute_metrics_for_task(task_df):
    results = {
        "task": [],
        "openai_zero_shot_bleu": [],
        "gemini_zero_shot_bleu": [],
        "openai_chain_of_thought_bleu": [],
        "gemini_chain_of_thought_bleu": [],
        "openai_zero_shot_rouge1": [],
        "gemini_zero_shot_rouge1": [],
        "openai_chain_of_thought_rouge1": [],
        "gemini_chain_of_thought_rouge1": [],
        "openai_zero_shot_codebert": [],
        "gemini_zero_shot_codebert": [],
        "openai_chain_of_thought_codebert": [],
        "gemini_chain_of_thought_codebert": []
    }

    # Debugging: Check and clean column names
    print("Original Columns:", task_df.columns)  # Print original column names
    task_df.columns = task_df.columns.str.strip()  # Strip leading/trailing spaces from column names
    print("Cleaned Columns:", task_df.columns)  # Print cleaned column names
    
    for index, row in task_df.iterrows():
        try:
            # Use 'OpenAI Entire Output' and 'Gemini Entire Output' for comparison
            openai_zero_shot_response = row['OpenAI Entire Output']
            gemini_zero_shot_response = row['Gemini Entire Output']
        except KeyError as e:
            print(f"Column missing: {e}")
            continue  # Skip this iteration if required columns are missing
        
        # Calculate BLEU for Zero-Shot and Chain-of-Thought
        openai_zero_shot_bleu = calculate_bleu(openai_zero_shot_response, gemini_zero_shot_response)
        gemini_zero_shot_bleu = calculate_bleu(gemini_zero_shot_response, openai_zero_shot_response)
        
        # Calculate ROUGE for Zero-Shot and Chain-of-Thought
        openai_zero_shot_rouge1, _, _ = calculate_rouge(openai_zero_shot_response, gemini_zero_shot_response)
        gemini_zero_shot_rouge1, _, _ = calculate_rouge(gemini_zero_shot_response, openai_zero_shot_response)
        
        # Calculate CodeBERT similarity for Zero-Shot and Chain-of-Thought
        openai_zero_shot_codebert = calculate_codebert_similarity(openai_zero_shot_response, gemini_zero_shot_response)
        gemini_zero_shot_codebert = calculate_codebert_similarity(gemini_zero_shot_response, openai_zero_shot_response)

        # Store results
        results["task"].append(row["Prompt"])  # Use 'Prompt' as the task name
        results["openai_zero_shot_bleu"].append(openai_zero_shot_bleu)
        results["gemini_zero_shot_bleu"].append(gemini_zero_shot_bleu)
        results["openai_chain_of_thought_bleu"].append(openai_zero_shot_bleu)  # Replace with chain-of-thought calculation if needed
        results["gemini_chain_of_thought_bleu"].append(gemini_zero_shot_bleu)  # Replace with chain-of-thought calculation if needed
        
        results["openai_zero_shot_rouge1"].append(openai_zero_shot_rouge1)
        results["gemini_zero_shot_rouge1"].append(gemini_zero_shot_rouge1)
        results["openai_chain_of_thought_rouge1"].append(openai_zero_shot_rouge1)  # Replace with chain-of-thought calculation if needed
        results["gemini_chain_of_thought_rouge1"].append(gemini_zero_shot_rouge1)  # Replace with chain-of-thought calculation if needed
        
        results["openai_zero_shot_codebert"].append(openai_zero_shot_codebert)
        results["gemini_zero_shot_codebert"].append(gemini_zero_shot_codebert)
        results["openai_chain_of_thought_codebert"].append(openai_zero_shot_codebert)  # Replace with chain-of-thought calculation if needed
        results["gemini_chain_of_thought_codebert"].append(gemini_zero_shot_codebert)  # Replace with chain-of-thought calculation if needed
    
    return results

# Compute metrics for all tasks and generate a DataFrame
metrics_results = {
    "task": [],
    "openai_zero_shot_bleu": [],
    "gemini_zero_shot_bleu": [],
    "openai_chain_of_thought_bleu": [],
    "gemini_chain_of_thought_bleu": [],
    "openai_zero_shot_rouge1": [],
    "gemini_zero_shot_rouge1": [],
    "openai_chain_of_thought_rouge1": [],
    "gemini_chain_of_thought_rouge1": [],
    "openai_zero_shot_codebert": [],
    "gemini_zero_shot_codebert": [],
    "openai_chain_of_thought_codebert": [],
    "gemini_chain_of_thought_codebert": []
}

for sheet_name, sheet_data in df.items():
    print(f"Processing sheet: {sheet_name}")
    task_results = compute_metrics_for_task(sheet_data)
    
    for key, value in task_results.items():
        metrics_results[key].extend(value)

# Convert metrics into a DataFrame
metrics_df = pd.DataFrame(metrics_results)

# Write metrics to a new Excel file
metrics_df.to_excel("model_comparison_metrics.xlsx", index=False)

print("Metrics calculation complete. Results saved to 'model_comparison_metrics.xlsx'.")
