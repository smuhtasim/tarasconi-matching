# %% Cell 1: Imports & Setup
import pandas as pd
import re
import ast
from rapidfuzz import fuzz, process
from tqdm import tqdm
import numpy as np

# %% Cell 2: Load Data
print("Loading datasets...")
# Added low_memory=False to fix the DtypeWarning
company_docs = pd.read_csv('../data/companies.csv', low_memory=False)
people = pd.read_csv('../data/people.csv', low_memory=False)

applicants = pd.read_parquet('../data/applicants.parquet')
inventors = pd.read_parquet('../data/inventors.parquet')
print(f"Companies: {company_docs.shape}, Applicants: {applicants.shape}")# %% Cell 3: Define Cleaning Functions

# Unpack lists like "['USA']" -> "USA"
def unpack_list_string(text):
    if isinstance(text, str) and text.startswith('[') and text.endswith(']'):
        try:
            val_list = ast.literal_eval(text)
            if isinstance(val_list, list) and len(val_list) > 0:
                return str(val_list[0])
        except (ValueError, SyntaxError):
            pass
    return text

# Harmonize Names (Remove 'Inc', 'Ltd', punctuation, etc.)
def clean_name(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    # Legal forms to strip (Tarasconi Preprocessing)
    legal_forms = [
        r'\binc\b', r'\binc\.', r'\bincorporated\b', 
        r'\bllc\b', r'\bltd\b', r'\blimited\b', 
        r'\bcorp\b', r'\bcorporation\b', r'\bgmbh\b', 
        r'\bco\b', r'\bcompany\b', r'\bs\.a\.\b', r'\bsa\b'
    ]
    regex_legal = '|'.join(legal_forms)
    text = re.sub(regex_legal, '', text)
    
    # Remove special chars
    text = re.sub(r'[^\w\s]', '', text) 
    
    # Remove stopwords
    stopwords = [r'\bthe\b', r'\bgroup\b', r'\bholdings\b', r'\binternational\b']
    regex_stop = '|'.join(stopwords)
    text = re.sub(regex_stop, '', text)
    
    # Whitespace cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Create "Fingerprint" (Alphanumeric only)
def create_fingerprint(text):
    return re.sub(r'\s+', '', text)

print("Functions defined.")

# %% Cell 4: Execute Preprocessing & Harmonization
print("Unpacking list strings in Crunchbase...")
cols_to_unpack = ['name', 'legal_name', 'country', 'region', 'city']
for col in cols_to_unpack:
    if col in company_docs.columns:
        company_docs[col] = company_docs[col].apply(unpack_list_string)

print("Harmonizing names...")
company_docs['clean_name'] = company_docs['name'].apply(clean_name)
applicants['clean_name'] = applicants['person_name'].apply(clean_name)

print("Standardizing Country Codes...")
# Map Crunchbase (3-letter) to PATSTAT (2-letter)
country_map = {
    'USA': 'US', 'GBR': 'GB', 'DEU': 'DE', 'FRA': 'FR', 
    'CAN': 'CA', 'CHN': 'CN', 'JPN': 'JP', 'KOR': 'KR',
    'ITA': 'IT', 'ESP': 'ES', 'IND': 'IN', 'ISR': 'IL'
}
company_docs['std_country'] = company_docs['country'].map(country_map).fillna(company_docs['country'])

# Handle Missing Countries
company_docs['std_country'] = company_docs['std_country'].fillna('UNKNOWN')
applicants['person_ctry_code'] = applicants['person_ctry_code'].fillna('UNKNOWN')

# Prepare pool for matching
unmatched_companies = company_docs.copy()
unmatched_applicants = applicants.copy() # Can filter this down if needed
all_matches = []

applicants.to_csv("../data/preprocessed_applicants.csv", index=False)
company_docs.to_csv("../data/preprocessed_companies.csv", index=False)

print("Preprocessing complete.")

# %% Cell 5: Pass 1 - Exact String Match
print("Running Pass 1: Exact Match...")

exact_matches = pd.merge(
    unmatched_companies,
    unmatched_applicants,
    left_on=['clean_name', 'std_country'],
    right_on=['clean_name', 'person_ctry_code'],
    how='inner'
)

if not exact_matches.empty:
    exact_matches['match_type'] = 'Exact'
    exact_matches['score'] = 1.0
    # Keep only relevant IDs
    all_matches.append(exact_matches[['company_id', 'person_id', 'match_type', 'score']])
    
    # Remove matched companies from the pool
    matched_cb_ids = exact_matches['company_id'].unique()
    unmatched_companies = unmatched_companies[~unmatched_companies['company_id'].isin(matched_cb_ids)]

print(f"Exact matches found: {len(exact_matches)}")
print(f"Remaining companies to match: {len(unmatched_companies)}")

# %% Cell 6: Pass 2 - Alphanumeric Match
print("Running Pass 2: Alphanumeric Match...")

# Generate fingerprints
unmatched_companies['fingerprint'] = unmatched_companies['clean_name'].apply(create_fingerprint)
unmatched_applicants['fingerprint'] = unmatched_applicants['clean_name'].apply(create_fingerprint)

alpha_matches = pd.merge(
    unmatched_companies,
    unmatched_applicants,
    left_on=['fingerprint', 'std_country'],
    right_on=['fingerprint', 'person_ctry_code'],
    how='inner'
)

if not alpha_matches.empty:
    alpha_matches['match_type'] = 'Alphanumeric'
    alpha_matches['score'] = 1.0
    all_matches.append(alpha_matches[['company_id', 'person_id', 'match_type', 'score']])

    matched_cb_ids = alpha_matches['company_id'].unique()
    unmatched_companies = unmatched_companies[~unmatched_companies['company_id'].isin(matched_cb_ids)]

print(f"Alphanumeric matches found: {len(alpha_matches)}")
print(f"Remaining companies to match: {len(unmatched_companies)}")
# PASS 3: Fuzzy Matching (Blocked)
# ---------------------------------------------------------
# %% Cell 7: Pass 3 - Fuzzy Matching
print("Running Pass 3: Fuzzy Match...")

# Type casting for safety
unmatched_companies['clean_name'] = unmatched_companies['clean_name'].astype(str)
unmatched_applicants['clean_name'] = unmatched_applicants['clean_name'].astype(str)

# Blocking Strategy (First 2 chars) to speed up search
unmatched_companies['block'] = unmatched_companies['clean_name'].str[:2]
unmatched_applicants['block'] = unmatched_applicants['clean_name'].str[:2]

fuzzy_results = []

# Create a dictionary of applicant blocks for O(1) lookup
app_blocks = dict(list(unmatched_applicants.groupby('block')))

# Iterate through remaining companies
for idx, cb_row in tqdm(unmatched_companies.iterrows(), total=unmatched_companies.shape[0]):
    block = cb_row['block']
    
    if block in app_blocks:
        app_group = app_blocks[block]
        
        # Tarasconi Rule: Filter by Country Code
        country_app_group = app_group[app_group['person_ctry_code'] == cb_row['std_country']]
        
        if country_app_group.empty:
            continue
            
        # Rapidfuzz extraction
        matches = process.extract(
            cb_row['clean_name'], 
            country_app_group['clean_name'], 
            scorer=fuzz.token_sort_ratio, 
            limit=1, 
            score_cutoff=90
        )
        
        for match in matches:
            # match = (match_string, score, match_index_label)
            matched_idx_label = match[2]
            
            # Use .loc because Rapidfuzz returns the dataframe Index Label
            matched_person_id = country_app_group.loc[matched_idx_label]['person_id']
            
            fuzzy_results.append({
                'company_id': cb_row['company_id'],
                'person_id': matched_person_id,
                'match_type': 'Fuzzy',
                'score': match[1] / 100.0
            })

if len(fuzzy_results) > 0:
    fuzzy_df = pd.DataFrame(fuzzy_results)
    all_matches.append(fuzzy_df)
    print(f"Fuzzy matches found: {len(fuzzy_df)}")
else:
    print("No Fuzzy matches found.")

# %% Cell 8: Save Final Results
if len(all_matches) > 0:
    final_org_matches = pd.concat(all_matches, ignore_index=True)
    final_org_matches.to_csv("data/step1_organization_matches.csv", index=False)
    print(f"Total Matches Found: {len(final_org_matches)}")
    print("Results saved to data/step1_organization_matches.csv")
else:
    print("No matches found.")

# %% Cell 9: Step 2 - People Validation Logic (Corrected)
import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

print("Starting Step 2: People Validation...")

# 1. Load Step 1 Results
# ---------------------------------------------------------
try:
    step1_matches = pd.read_csv('../data/step1_organization_matches.csv')
    print(f"Loaded {len(step1_matches)} potential organization matches.")
except FileNotFoundError:
    raise SystemExit("Error: Step 1 file not found. Please run previous cells.")

if step1_matches.empty:
    print("Step 1 found no matches. Validation skipped.")
else:
    # 2. Prepare People Data (Crunchbase)
    # ---------------------------------------------------------
    # exact_condition = (step1_matches['match_type'] == 'Exact')
    # step1_matches = step1_matches[exact_condition]
    # print(f"Total Exact Match rows found: {len(step1_matches)}")
    # print("\nSample of Exact Matches:")
    # print(step1_matches.head())
    print("Preparing Crunchbase People...")
    relevant_cb_ids = step1_matches['company_id'].unique()
    relevant_people = people[people['company_id'].isin(relevant_cb_ids)].copy()

    # Function to normalize names by sorting tokens (solves "John Doe" vs "Doe John")
    def sort_tokens(text):
        if not isinstance(text, str): return ""
        # 1. Clean (reuse your clean_name logic or simple version)
        text = str(text).lower().replace(',', ' ').replace('.', '')
        # 2. Sort words alphabetically
        return " ".join(sorted(text.split()))

    # Create Full Names
    relevant_people['full_name'] = relevant_people['first_name'].astype(str) + " " + relevant_people['last_name'].astype(str)
    # Apply Token Sort
    relevant_people['token_sorted_name'] = relevant_people['full_name'].apply(sort_tokens)

    # 3. Prepare Inventor Data (PATSTAT)
    # ---------------------------------------------------------
    print("Preparing PATSTAT Inventors...")
    relevant_pat_ids = step1_matches['person_id'].unique()
    
    # Get Apps
    relevant_apps = applicants[applicants['person_id'].isin(relevant_pat_ids)][['person_id', 'appln_id']]
    
    # Get Inventors
    relevant_inv = inventors[inventors['appln_id'].isin(relevant_apps['appln_id'])].copy()
    # Apply Token Sort
    relevant_inv['token_sorted_name'] = relevant_inv['person_name'].apply(sort_tokens)

    # 4. The Validation Loop
    # ---------------------------------------------------------
    print("Running Validation Cross-Check...")

    # A. Flatten Crunchbase People (Company ID -> Sorted Name)
    cb_lookup = relevant_people[['company_id', 'token_sorted_name']].drop_duplicates()
    cb_lookup.rename(columns={'token_sorted_name': 'cb_person'}, inplace=True)

    # B. Flatten PATSTAT Inventors (Applicant ID -> Sorted Name)
    # Merge Inventors -> Applicants
    pat_inv_lookup = pd.merge(relevant_inv[['appln_id', 'token_sorted_name']], 
                              relevant_apps[['appln_id', 'person_id']], 
                              on='appln_id')
    pat_inv_lookup = pat_inv_lookup[['person_id', 'token_sorted_name']].drop_duplicates()
    pat_inv_lookup.rename(columns={'token_sorted_name': 'pat_person'}, inplace=True)

    # C. Merge and Compare
    # Join Step 1 matches with CB People
    validation_table = pd.merge(step1_matches, cb_lookup, on='company_id', how='left')
    # Join with PATSTAT Inventors
    validation_table = pd.merge(validation_table, pat_inv_lookup, on='person_id', how='left')

    # D. Strict Match on Sorted Tokens
    # "elon musk" == "elon musk" (regardless of original order)
    validated_rows = validation_table[
        (validation_table['cb_person'].notna()) & 
        (validation_table['pat_person'].notna()) & 
        (validation_table['cb_person'] == validation_table['pat_person'])
    ]

    # Get valid IDs
    valid_pairs = validated_rows[['company_id', 'person_id']].drop_duplicates()
    valid_pairs['is_validated'] = True

    # 5. Integrate & Save
    # ---------------------------------------------------------
    final_results = pd.merge(step1_matches, valid_pairs, on=['company_id', 'person_id'], how='left')
    final_results['is_validated'] = final_results['is_validated'].fillna(False)

    # Output Stats
    total = len(final_results)
    validated = final_results['is_validated'].sum()
    print(f"Total Matches from Step 1: {total}")
    print(f"Matches Validated by People: {validated} ({validated/total:.2%})")

    if validated == 0:
        print("Still 0? Check 'relevant_people' and 'relevant_inv' manually to see if names look similar.")

    final_results.to_csv("../data/final_tarasconi_results.csv", index=False)
    print("Final results saved.")

# %% Cell 10: Check People Validation Logic
# Assuming 'companies' and 'people' DataFrames are loaded
# --- Define the specific match IDs ---
TARGET_COMPANY_ID = 161128
TARGET_APPLICANT_ID = 45172035

# --- Assuming DataFrames (companies, applicants, inventors, people) are loaded ---

# 1. Find the Company's Name and People (Crunchbase Side)
cb_org_details = company_docs[company_docs['company_id'] == TARGET_COMPANY_ID]
cb_people_list = people[people['company_id'] == TARGET_COMPANY_ID]

# 2. Find the Applicant's Name and Associated Patents (PATSTAT Side)
pat_applicant_info = applicants[applicants['person_id'] == TARGET_APPLICANT_ID]

# Extract all application IDs (appln_id) associated with this applicant
# These are the patents that prove the validation
associated_applns = pat_applicant_info['appln_id'].unique()

# 3. Find the Inventors and Patent Titles
# Filter inventors table to see who worked on these patents
pat_inventors_list = inventors[inventors['appln_id'].isin(associated_applns)]

# --------------------------------------------------------------------------------
# STEP 2: PRINT AND VERIFY
# --------------------------------------------------------------------------------

print("=====================================================================")
print(f"VERIFICATION FOR COMPANY ID: {TARGET_COMPANY_ID}")
print("=====================================================================")

# A. Display Organizational Match (Check Name and Country match validity)
print("\nA. ORGANIZATIONAL MATCH (Name/Country):")
print("-" * 50)
print(f"CRUNCHBASE COMPANY NAME: {cb_org_details['name'].iloc[0]} ({cb_org_details['country'].iloc[0]})")
print(f"PATSTAT APPLICANT NAME: {pat_applicant_info['person_name'].iloc[0]} ({pat_applicant_info['person_ctry_code'].iloc[0]})")

# B. Display The Triangle of Trust (Check the Human Overlap)
print("\nB. PEOPLE VALIDATION (Triangle of Trust):")
print("-" * 50)

# Extract cleaned names for easy comparison
cb_person_names = cb_people_list['first_name'] + " " + cb_people_list['last_name']
pat_inventor_names = pat_inventors_list['person_name'].unique()

print("\n--- CRUNCHBASE PEOPLE (Founders/Executives) ---")
print(cb_person_names.to_string(index=False))

print("\n--- PATSTAT INVENTORS (Patent Authors) ---")
# Show a list of all unique inventors on their patents
print("\n".join(pat_inventor_names))

print(f"\nTotal Patents Found: {len(associated_applns)}")

# C. Display Validation Proof
# Manually cross-check names from the two lists above.
# If a name from the Crunchbase list appears in the PATSTAT list, the 'is_validated=True' flag is confirmed.
# %% Cell 11: Setup
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import gc # Garbage collection to manage RAM

print("Loading Embedding Model...")
# Load the lightweight model
# If you have a GPU, this will automatically use it. If not, it uses CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

print(f"Model loaded on: {device}")


# %% Cell 12: Load Data
print("Loading Datasets...")

# 1. Main Entities
applicants = pd.read_parquet('../data/applicants.parquet')
company_docs = pd.read_csv('../data/companies.csv', low_memory=False)

# 2. Context Files (NACE & Patents)
patent_nace = pd.read_parquet('../data/patent_nace.parquet')
nace_codes = pd.read_parquet('../data/nace_codes.parquet')
# We need patents.parquet now for Titles/Abstracts
patents = pd.read_parquet('../data/patents.parquet')
# %% Cell 12b: Merge Publication Dates
usecols = ['appln_id', 'publn_date', 'ipc_class_symbol', 'cpc_class_symbol']
df_dates = pd.read_csv(
    '../data/cleantech_publn_date.csv', 
    usecols=usecols,              # optional: converts string dates to datetime objects
)

merged_df = pd.merge(
    applicants, 
    df_dates, 
    on='appln_id', 
    how='left'
)

print(f"Parquet rows: {len(applicants)}")
print(f"Merged rows: {len(merged_df)}")
print(merged_df.head())

# 5. Save the result (optional)
merged_df.to_parquet('../data/combined_data.parquet')

print("Data Loaded.")
df_parquet = pd.read_parquet('../data/combined_data.parquet')
# %% Cell 13 a: data merge and preprocessing for patent


# --- 3. Define a Helper Function for Cleaning ---
def aggregate_codes(series):
    all_codes = []
    
    for item in series:
        # 1. Handle actual List or Array objects
        if isinstance(item, (list, np.ndarray, pd.Series)):
            all_codes.extend([str(i) for i in item])
        
        # 2. Handle cases where the list is "hidden" inside a string like "['code1', 'code2']"
        elif isinstance(item, str) and item.startswith('[') and item.endswith(']'):
            # Strip brackets and split by comma, then clean quotes
            parts = item.strip('[]').split(',')
            all_codes.extend([p.strip().replace("'", "").replace('"', '') for p in parts])
            
        # 3. Handle single items
        else:
            all_codes.append(str(item))
    
    # Preprocessing individual codes
    cleaned = []
    for c in all_codes:
        # Skip nulls or the literal string 'nan'
        if pd.isna(c) or str(c).lower() == 'nan' or str(c) == 'None' or not c:
            continue
        
        # Strip whitespace and collapse multiple spaces (H01L  21 -> H01L 21)
        s = re.sub(r'\s+', ' ', str(c)).strip()
        
        if s:
            cleaned.append(s)
    
    # Deduplicate and Join with '*'
    unique_sorted = sorted(list(set(cleaned)))
    return "*".join(unique_sorted) if unique_sorted else np.nan

# --- 4. Aggregate the Parquet Data ---
# Group by ID to handle multiple entries per application
df_codes = df_parquet.groupby('appln_id').agg({
    'ipc_class_symbol': aggregate_codes,
    'cpc_class_symbol': aggregate_codes,
    # Add other columns here if you need to aggregate them (e.g., 'title': 'first')
}).reset_index()

# Rename columns to indicate they are now cleaned/aggregated
cols_to_use = df_dates.columns.difference(['ipc_class_symbol', 'cpc_class_symbol'])

# --- 5. Merge Dataframes ---
# Left join: We keep all IDs from our Codes dataset (or Dates, depending on priority).
# Here we stick to the Dates file as the base if that's your master list.
df_final = pd.merge(df_dates[cols_to_use], df_codes, on='appln_id', how='left')

# --- 6. Create the "Combined" Column with Special Separator ---
# We handle cases where one might be NaN by filling with empty strings
df_final['ipc_cleaned'] = df_final['ipc_class_symbol'].fillna('')
df_final['cpc_cleaned'] = df_final['cpc_class_symbol'].fillna('')

# Combine format: IPC * IPC * IPC | CPC * CPC
# The '|' acts as the special separator between the two systems
df_final['combined_classifications'] = (
    df_final['ipc_cleaned'] + " | " + df_final['cpc_cleaned']
)

# Clean up artifacts (e.g., if one side was empty, we might have " | code" or "code | ")
df_final['combined_classifications'] = df_final['combined_classifications'].str.strip(' |')

# --- 7. Inspection ---
print("Sample of processed data:")
print(df_final[['appln_id', 'publn_date', 'combined_classifications']].head())

df_final.to_parquet('../data/combined_codes_dates.parquet')

# %% Cell 13c: load all data for patent enrichment

print("Enriching PATSTAT Data...")



if len(patents) != len(augmented_codes):
    print(f"⚠️ Warning: Length mismatch! Patents: {len(patents)}, Augmented: {len(augmented_codes)}")

# 2. Check if the index values are identical
index_match = patents.index.equals(augmented_codes.index)

if not index_match:
    print("❌ Indexes do NOT match. Concatenation will create NaNs.")
    # Check if they at least have the same IDs so we can fix it
    missing_in_aug = patents.index.difference(augmented_codes.index)
    print(f"Items in Patents missing from Augmented: {len(missing_in_aug)}")
else:
    print("✅ Indexes match perfectly. Safe to combine.")


# %% Cell 13: Prepare PATSTAT Documents (Query Side)
patents = pd.read_parquet('../data/patents.parquet')
augmented_codes = pd.read_parquet('../data/combined_codes_dates.parquet')
applicants = pd.read_parquet('../data/applicants.parquet')
print("Data Loaded. Starting Smart Profiling...")



# --- STEP A: Create the "Rich Text" for each Patent ---
# We use the 'combined_classifications' you just perfected.
# We also extract a 'subclass' (first 4 chars) for the "Top Sectors" statistical calculation.
def extract_subclasses(text):
    if pd.isna(text) or text == '': return []
    # Find all codes, but only keep the first 4 chars (e.g., H01L)
    # We split by * and | to get individual codes
    codes = re.split(r'[*|]', text)
    return [c.strip()[:4] for c in codes if len(c.strip()) >= 4]



augmented_codes['subclasses'] = augmented_codes['combined_classifications'].apply(extract_subclasses)

# Create the text blob for the individual patent
# Note: We include the FULL classification string here for deep context
combined_df = pd.merge(
    patents[['appln_id', 'title', 'abstract']], 
    augmented_codes[['appln_id', 'combined_classifications', 'subclasses', 'publn_date']], 
    on='appln_id', 
    how='left'
)

# --- 2. Now create full_text safely within the same DataFrame ---
combined_df['full_text'] = (
    "Title: " + combined_df['title'].fillna('No Title').astype(str) + 
    ". Classes: " + combined_df['combined_classifications'].fillna('').astype(str) + 
    ". Abstract: " + combined_df['abstract'].fillna('').str.slice(0, 300).astype(str)
)

# Verify no NaNs were created in the string concatenation
nan_count = combined_df['full_text'].isna().sum()
print(f"Number of NaN strings in full_text: {nan_count}")

# --- STEP B: Link Patents to Applicants ---
app_pat_link = applicants[['person_id', 'appln_id', 'person_ctry_code']].merge(
    combined_df[['appln_id', 'full_text', 'publn_date', 'subclasses']],
    on='appln_id',
    how='inner'
)

# --- STEP C: The "Smart" Sort (Most Recent First) ---
print("Sorting patents by date...")
app_pat_link = app_pat_link.sort_values(by=['person_id', 'publn_date'], ascending=[True, False])

# --- STEP D: Aggregate to Applicant Level ---
def aggregate_profile(x):
    # 1. Recent Text: Take top 3 recent patents
    recent_text = " || ".join(x['full_text'].head(3))
    
    # 2. Dominant Tech: Flatten the list of subclasses and find the most frequent
    all_subs = [item for sublist in x['subclasses'] for item in sublist]
    if all_subs:
        top_subs = pd.Series(all_subs).value_counts().head(3).index.tolist()
        ipc_string = ", ".join(top_subs)
    else:
        ipc_string = "Unknown"
    
    return pd.Series({'recent_patents': recent_text, 'top_sectors': ipc_string})

print("Grouping by Applicant...")
applicant_docs = app_pat_link.groupby(['person_id', 'person_ctry_code']).apply(aggregate_profile).reset_index()

# --- STEP E: Create Final Embedding String ---
applicant_docs['embed_string'] = (
    "Applicant Country: " + applicant_docs['person_ctry_code'].fillna('UNK') + 
    ". Top Tech Sectors: " + applicant_docs['top_sectors'] + 
    ". Recent Patent History: " + applicant_docs['recent_patents']
)

print(f"Created {len(applicant_docs)} smart applicant profiles.")
print("\nSample Embedding String (first 500 chars):")
print(applicant_docs['embed_string'].iloc[0][:500])

# Save immediately
applicant_docs.to_parquet('../data/intermediate_pat_smart_enriched.parquet', index=False)

# %% Cell 13b: Re-Align Applicant Embeddings
from sentence_transformers import SentenceTransformer
import pandas as pd

# 1. Load the NEW, sorted/grouped text data
applicant_docs = pd.read_parquet('../data/intermediate_pat_smart_enriched.parquet')

# 2. Load the Model
model = SentenceTransformer('all-MiniLM-L6-v2') # Or your specific model

# 3. Re-Create Embeddings
print(f"Generating embeddings for {len(applicant_docs)} applicants...")
# This ensures Vector 0 = Person 0, Vector 1 = Person 1
applicant_embeddings = model.encode(
    applicant_docs['embed_string'].tolist(), 
    batch_size=32, 
    show_progress_bar=True, 
    convert_to_tensor=True
)

# 4. Save them immediately so they never get out of sync
import pickle
with open('../data/applicant_embeddings.pkl', 'wb') as f:
    pickle.dump(applicant_embeddings, f)
    
print("Embeddings re-aligned and saved.")

# %% Cell 14: Prepare Crunchbase Documents (Candidate Side)
print("Enriching Crunchbase Data...")

# Helper to clean lists "['Category']" -> "Category"
def clean_cb_text(text):
    if pd.isna(text) or text is None:
        return ""
    
    # If it's a list string (like "['Category']"), clean it up
    if isinstance(text, str) and text.startswith("['"):
        return text.replace("['", "").replace("']", "").replace("', '", ", ")
    return str(text)

company_docs['clean_desc'] = (
    company_docs['cb_short_description'].apply(clean_cb_text) + 
    " " + company_docs['pb_keywords'].apply(clean_cb_text)
).str.strip()
company_docs['clean_cats'] = company_docs['cb_category_list'].apply(clean_cb_text)

company_docs['final_desc'] = np.where(
    company_docs['clean_desc'].str.len() > 5, 
    company_docs['clean_desc'], 
    'No Description Provided. Name is key.' # Imputation Text
)

company_docs['final_cats'] = np.where(
    company_docs['clean_cats'].str.len() > 5,
    company_docs['clean_cats'],
    'No Categories Listed.' # Imputation Text
)

# Create Final Embedding String
# Structure: "Name. Description. Categories."
company_docs['embed_string'] = (
    "Startup: " + company_docs['name'].fillna('') + 
    ". Loc: " + company_docs['std_country'].fillna('UNK') + # Assuming you have std_country from Step 1
    ". Desc: " + company_docs['clean_desc'] + 
    ". Cats: " + company_docs['clean_cats']
)

# Filter out empty rows to avoid errors
company_docs = company_docs[company_docs['embed_string'].str.len() > 30].copy()

print(f"Created {len(company_docs)} enriched startup documents.")

print("Sample (First non-empty entry):")
# Find a non-trivial sample to demonstrate the fix
non_empty_sample = company_docs[company_docs['final_desc'].str.contains('No Description Provided') == False].head(1)['embed_string']
if not non_empty_sample.empty:
    print(non_empty_sample.iloc[0][:150])
else:
    # If all samples are empty, print the first one with the fallback text
    print(company_docs['embed_string'].iloc[0][:150])
# NEW STEP: Save enriched companies data
company_docs.to_parquet('../data/intermediate_cb_enriched.parquet', index=False)


# %%
valid_mask = company_docs['final_desc'].str.contains('No Description Provided') == True

# Get the index of the first True value
first_valid_index = valid_mask.idxmax()
# Get the specific index of the first match


print(f"Index of first non-empty sample: {first_valid_index}")

# Inspect that specific row
print(company_docs.loc[first_valid_index, 'embed_string'])
# %% Cell 15: Generate Embeddings
company_docs = pd.read_parquet('../data/intermediate_cb_enriched.parquet')
print("Encoding Crunchbase Candidates (This may take a few minutes)...")
# Encode all startups into a matrix
cb_embeddings = model.encode(
    company_docs['embed_string'].tolist(), 
    batch_size=64, 
    show_progress_bar=True, 
    convert_to_tensor=True
)

applicant_docs = pd.read_parquet('../data/intermediate_pat_smart_enriched.parquet')
print("Encoding PATSTAT Queries (This may take longer)...")
# For testing, let's just do the Ambiguous ones or a sample
# If running fully, remove the .head()
applicant_embeddings = model.encode(
    applicant_docs['embed_string'].tolist(), 
    batch_size=64, 
    show_progress_bar=True, 
    convert_to_tensor=True
)

print(f"CB Matrix: {cb_embeddings.shape}")
print(f"PATSTAT Matrix: {applicant_embeddings.shape}")
# %% Cell 16: Save Embeddings Aligned with IDs
# 1. Convert Tensor to a List of NumPy arrays
# We move it to CPU first, then to Numpy
embeddings_np = cb_embeddings.cpu().numpy()

# 2. Assign it back to your companies DataFrame
# Now every company row has its own embedding vector
company_docs['embedding'] = list(embeddings_np)

# 3. Save the specific columns you need for the LLM step
# You generally want the ID, the Text you used, and the Vector
columns_to_save = ['company_id', 'embed_string', 'embedding']
company_docs[columns_to_save].to_parquet('../data/modified_embedded_crunchbase.parquet', index=False)

print("Saved embeddings aligned with IDs to data/embedded_crunchbase.parquet")

# %% Cell 17: Save Applicant Embeddings Aligned with IDs
print("Saving Applicant Embeddings...")
# --- 1. Convert Tensor to NumPy ---
# Move the tensor from GPU (if applicable) to CPU memory, then convert to a NumPy array.
embeddings_np = applicant_embeddings.cpu().numpy()

# --- 2. Add the Vectors to the DataFrame ---
# Convert the NumPy array of embeddings into a list of vectors,
# where each element is a list of 384 numbers.
applicant_docs['embedding'] = list(embeddings_np)

# --- 3. Save the DataFrame as Parquet ---
# Select the columns necessary for the retrieval and LLM steps: the ID, the string, and the vector.
columns_to_save = ['person_id', 'embed_string', 'embedding']
applicant_docs[columns_to_save].to_parquet('../data/embedded_patstat.parquet', index=False)

# %% Check alignment between DataFrame and Embeddings
import pickle
import torch
import io

# Define a custom unpickler that redirects GPU data to CPU
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

# --- Updated Verification Code ---
import pandas as pd

# 1. Load Data
df_apps = pd.read_parquet('../data/intermediate_pat_smart_enriched.parquet')

# 2. Load Embeddings using the Custom Unpickler
print("Loading embeddings to CPU...")
with open('../data/applicant_embeddings.pkl', 'rb') as f:
    emb_apps = CPU_Unpickler(f).load()

# 3. Check Alignment
print(f"DataFrame Rows: {len(df_apps)}")
print(f"Embedding Rows: {emb_apps.shape[0]}")

if len(df_apps) != emb_apps.shape[0]:
    print("🚨 FATAL: Lengths do not match. You must re-generate embeddings.")
else:
    print("✅ Lengths match. Proceeding to semantic search.")

# 3. The "sanity check" isn't possible directly on vectors without the model, 
# BUT we can ensure we don't sort after this point.
# IF you sort here, you break the link.
# df_apps = df_apps.sort_values(...) # <--- NEVER DO THIS AFTER LOADING EMBEDDINGS

print("Saved Applicant embeddings aligned with IDs to data/embedded_patstat.parquet")

# %% Cell 18: Generate Crunchbase Embeddings pickles
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle

# 1. Load your Company Text Data
# Use the file you have: 'intermediate_cb_enriched' or 'embedded_crunchbase'
print("Loading Company Data...")
company_docs = pd.read_parquet('../data/embedded_crunchbase.parquet') 
# (Make sure this file has the 'embed_string' column)

# 2. Load the Model
# Important: Use the EXACT same model you used for Applicants
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Generate Embeddings
print(f"Encoding {len(company_docs)} companies... (This might take a minute)")
cb_embeddings = model.encode(
    company_docs['embed_string'].tolist(), 
    batch_size=32, 
    show_progress_bar=True, 
    convert_to_tensor=True
)

# 4. Save as Pickle (So your search code can find it)
print("Saving embeddings to pickle...")
with open('../data/crunchbase_embeddings.pkl', 'wb') as f:
    pickle.dump(cb_embeddings, f)

print("✅ Done! You can now run the Search Cell.")



# %% Cell 19: Load Embeddings for Retrieval
import pandas as pd
import torch

print("Loading enriched datasets from Parquet files...")

# 1. Load the Parquet files
applicant_docs = pd.read_parquet('../data/embedded_patstat.parquet')
companies_docs = pd.read_parquet('../data/embedded_crunchbase.parquet')


# 3. JOIN the countries back to your embedded dataframes
# print("Patching missing country columns...")
# applicant_docs = applicant_docs.merge(applicants, on='person_id', how='left')
# companies_docs = companies_docs.merge(company_docs, on='company_id', how='left')

# # 4. Convert to Tensors (as before)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# applicant_embeddings = torch.tensor(applicant_docs['embedding'].tolist()).to(device)
# cb_embeddings = torch.tensor(companies_docs['embedding'].tolist()).to(device)


# Clean up memory
# %% Cell 16: Retrieval with Country Blocking
from sentence_transformers import util
print("Running Semantic Search with Country Blocking...")

# We need a map to look up the original Company IDs
cb_idx_to_id = company_docs['company_id'].reset_index(drop=True)
# We also need the country for every row in the embedding matrix
cb_countries = company_docs['std_country'].reset_index(drop=True).values

results_list = []

# Iterate through Applicants (Queries)
# We do this in a loop because we need to apply the Country Filter *before* searching
# Optimizing: We can group queries by country to speed this up

unique_countries = applicant_docs['person_ctry_code'].unique()

for country in unique_countries:
    if country == 'UNKNOWN': continue
    
    # 1. Get Indices of Startups in this Country
    # This is the "Hard Filter"
    candidate_indices = [i for i, c in enumerate(cb_countries) if c == country]
    
    if not candidate_indices: continue # No startups in this country
    
    # Get the embedding subset for this country
    country_cb_embeddings = cb_embeddings[candidate_indices]
    
    # 2. Get Indices of Applicants in this Country
    # (In your df, you can just filter)
    app_mask = applicant_docs['person_ctry_code'] == country
    country_app_embeddings = applicant_embeddings[app_mask] # Be careful with tensor indexing
    # We need the original IDs
    current_app_ids = applicant_docs.loc[app_mask, 'person_id'].values
    
    # 3. Semantic Search (Cosine Similarity)
    # Search for top-10 matches
    hits = util.semantic_search(
        country_app_embeddings, 
        country_cb_embeddings, 
        top_k=10
    )
    
    # 4. Store Results
    for i, query_hits in enumerate(hits):
        app_id = current_app_ids[i]
        
        for hit in query_hits:
            # hit contains {'corpus_id', 'score'}
            # corpus_id is the index inside 'country_cb_embeddings'
            # We need to map it back to the global 'candidate_indices'
            local_idx = hit['corpus_id']
            global_idx = candidate_indices[local_idx]
            company_id = cb_idx_to_id[global_idx]
            
            results_list.append({
                'person_id': app_id,
                'company_id': company_id,
                'semantic_score': hit['score'],
                'rank': 1 # rank logic can be added
            })

print(f"Retrieval complete. Found {len(results_list)} candidate pairs.")
# %%
company_docs = pd.read_parquet('../data/embedded_crunchbase.parquet')
print("Columns in companies:", company_docs.columns.tolist())
# %% cell 19: Save Semantic Matches
# Create the DataFrame
matches_df = pd.DataFrame(results_list)

# 1. Save the full results in a compressed format
matches_df.to_parquet('../data/new_all_semantic_matches.parquet', index=False)


# 2. Save a smaller, manageable CSV for manual inspection (e.g., top 10,000)
matches_df.head(10000).to_csv('../data/new_sample_matches.csv', index=False)

# %% Cell 16: Retrieval with Country Blocking (New)
from sentence_transformers import util
import pandas as pd
import pickle
import torch
import io

# --- 1. Helper for CPU Loading (Since you are on CPU) ---
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

print("Loading Data for Search...")

# --- 2. Load APPLICANTS (Queries) ---
# CRITICAL: Do not sort this DataFrame after loading!
applicant_docs = pd.read_parquet('../data/intermediate_pat_smart_enriched.parquet')
applicant_docs = applicant_docs.reset_index(drop=True) # Ensure standard 0..N index

with open('../data/applicant_embeddings.pkl', 'rb') as f:
    applicant_embeddings = CPU_Unpickler(f).load()

# --- 3. Load COMPANIES (Targets) ---
company_docs = pd.read_parquet('../data/embedded_crunchbase.parquet')
# Ensure standard 0..N index for companies too
company_docs = company_docs.reset_index(drop=True)

# Load Company Embeddings
# Assuming you saved these similarly; if not, adjust filename
with open('../data/crunchbase_embeddings.pkl', 'rb') as f:
    cb_embeddings = CPU_Unpickler(f).load()

companies = pd.read_csv('../data/preprocessed_companies.csv')

# --- 4. Load Name Lookups for Better Readability ---
# Load applicants for person name mapping
applicants = pd.read_csv('../data/preprocessed_applicants.csv', low_memory=False)
person_name_map = applicants[['person_id', 'person_name']].drop_duplicates().set_index('person_id')['person_name'].to_dict()

# Create company name mapping from company_docs
company_name_map = company_docs.set_index('company_id')['embed_string'].to_dict()

# --- 5. Setup Lookups ---
# Map Index -> Company ID for fast lookup
cb_idx_to_id = companies['company_id'].to_dict()
# Get country list for filtering (aligned with cb_embeddings rows)
cb_countries = companies['std_country'].values 

results_list = []
unique_countries = applicant_docs['person_ctry_code'].unique()

print(f"Starting Search across {len(unique_countries)} countries...")

# --- 6. The Search Loop ---
for country in unique_countries:
    if pd.isna(country) or country == 'UNKNOWN': continue
    
    # A. Get TARGET (Startup) Indices for this Country
    # This gives us integers like [0, 5, 20...]
    startup_indices = [i for i, c in enumerate(cb_countries) if c == country]
    
    if not startup_indices: continue # No startups in this country
    
    # Slice the embedding matrix (these are the 'corpus')
    country_cb_embeddings = cb_embeddings[startup_indices]
    
    # B. Get QUERY (Applicant) Indices for this Country
    # We use .index to get the exact integer positions in the main DataFrame
    app_indices = applicant_docs.index[applicant_docs['person_ctry_code'] == country].tolist()
    
    if not app_indices: continue

    # Slice the query embeddings
    country_app_embeddings = applicant_embeddings[app_indices]
    
    # C. Perform Search (Cosine Similarity)
    # Returns 10 matches for each applicant in this batch
    hits = util.semantic_search(
        country_app_embeddings, 
        country_cb_embeddings, 
        top_k=10
    )
    
    # D. Map Results back to IDs
    for i, query_hits in enumerate(hits):
        # 'i' is the local index in the batch. 
        # We need the global index to get the ID.
        global_app_idx = app_indices[i]
        app_id = applicant_docs.at[global_app_idx, 'person_id']
        
        # Get applicant name
        person_name = person_name_map.get(app_id, "Unknown Applicant")
        
        for hit in query_hits:
            # hit['corpus_id'] is the local index in the startup subset
            local_startup_idx = hit['corpus_id']
            
            # Map to Global Startup Index
            global_startup_idx = startup_indices[local_startup_idx]
            
            # Retrieve Company ID
            company_id = cb_idx_to_id[global_startup_idx]
            
            # Extract company name from embed_string (format: "Startup: {name} ...")
            company_embed = company_name_map.get(company_id, "Unknown Company")
            company_name = company_embed.split("Loc:")[0].replace("Startup:", "").strip() if "Loc:" in company_embed else "Unknown Company"
            
            results_list.append({
                'person_id': app_id,
                'person_name': person_name,
                'company_id': company_id,
                'company_name': company_name,
                'semantic_score': hit['score'],
                'rank': i+1
            })

print(f"Search Complete. Found {len(results_list)} matches.")

# %% cell 20: Inspect Sample Matches
# Select a random sample of 10 matches to inspect
sample_to_check = matches_df.sample(10).copy()

# Join with applicant names/text
sample_to_check = sample_to_check.merge(
    applicant_docs[['person_id', 'embed_string']], 
    on='person_id', 
    how='left'
).rename(columns={'embed_string': 'applicant_info'})

# Join with company names/text
sample_to_check = sample_to_check.merge(
    company_docs[['company_id', 'embed_string']], 
    on='company_id', 
    how='left'
).rename(columns={'embed_string': 'company_info'})

# Display the results
for i, row in sample_to_check.iterrows():
    print(f"--- Match {i+1} (Score: {row['semantic_score']:.4f}) ---")
    print(f"APPLICANT: ({row['person_name']}) {row['applicant_info'][:400]}...")
    print(f"STARTUP:   {row['company_info'][:400]}...")
    print("\n")
# %% Score Distribution
import matplotlib.pyplot as plt
matches_df = pd.read_parquet('../data/new_all_semantic_matches.parquet')
# Plot the distribution of semantic scores
plt.hist(matches_df['semantic_score'], bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Semantic Scores')
plt.xlabel('Cosine Similarity Score')
plt.ylabel('Number of Matches')
plt.axvline(x=0.5, color='red', linestyle='--', label='High Quality Threshold')
plt.legend()
plt.show()

# %% Calculate counts for different thresholds
tiers = [0.4, 0.5, 0.6, 0.7]
for t in tiers:
    count = len(matches_df[matches_df['semantic_score'] >= t])
    print(f"Matches >= {t}: {count:,}")
# %%
matches_df = pd.read_parquet('../data/new_all_semantic_matches.parquet')
# 1. Filter for high-quality matches first
high_score_matches = matches_df[matches_df['semantic_score'] > 0.60].copy()

# 2. Select a sample (e.g., 10) from this high-quality group
high_quality_sample = high_score_matches.sample(min(10, len(high_score_matches)))

# 3. Join the text back (re-using the logic from before)
# (Assuming cb_text_lookup and pat_text_lookup are still in memory)
review_df = high_quality_sample.merge(applicant_docs, on='person_id', how='left') \
                               .merge(company_docs, on='company_id', how='left', suffixes=('_pat', '_cb'))

# 4. Print results
print(f"Showing {len(review_df)} High-Score Matches (> 0.70):\n")
for _, row in review_df.iterrows():
    print(f"SCORE: {row['semantic_score']:.4f}")
    print(f"PATENT APP: ({row['person_name']}) {row['embed_string_pat'][:200]}...")
    print(f"CB STARTUP: {row['embed_string_cb'][:200]}...")
    print("-" * 30)
# %% Generate the High-Confidence Report
matches_df = pd.read_parquet('../data/new_all_semantic_matches.parquet')
# 1. Filter for scores above 0.58 (High Confidence)
high_conf_matches = matches_df[matches_df['semantic_score'] >= 0.50].copy()

applicant_docs = pd.read_parquet('../data/intermediate_pat_enriched.parquet')
companies = pd.read_parquet('../data/intermediate_cb_enriched.parquet')

# Create lookup for Applicants
# Replace 'doc_std_name' with 'person_name' if that is what your file uses
applicant_names = applicant_docs[['person_id', 'person_name', 'person_ctry_code']].drop_duplicates()

# Create lookup for Companies
# 'companies' is the dataframe you loaded from modified_embedded_crunchbase.parquet
company_names = companies[['company_id', 'legal_name', 'domain', 'country']].drop_duplicates()

# 2. Add the actual names and countries for readability
# (Assuming you have a mapping of person_id to name and company_id to name)
final_report = high_conf_matches.merge(applicant_names, on='person_id').merge(company_names, on='company_id')

# 3. Save as Excel or CSV for the supervisor
final_report.to_csv('../data/top_semantic_matches_to_verify.csv', index=False)

# %%
