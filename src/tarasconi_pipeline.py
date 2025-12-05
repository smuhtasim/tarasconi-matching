# %% Cell 1: Imports & Setup
import pandas as pd
import re
import ast
from rapidfuzz import fuzz, process
from tqdm import tqdm

# %% Cell 2: Load Data
print("Loading datasets...")
# Added low_memory=False to fix the DtypeWarning
companies = pd.read_csv('../data/companies.csv', low_memory=False)
people = pd.read_csv('../data/people.csv', low_memory=False)

applicants = pd.read_parquet('../data/applicants.parquet')
inventors = pd.read_parquet('../data/inventors.parquet')
print(f"Companies: {companies.shape}, Applicants: {applicants.shape}")# %% Cell 3: Define Cleaning Functions

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
    if col in companies.columns:
        companies[col] = companies[col].apply(unpack_list_string)

print("Harmonizing names...")
companies['clean_name'] = companies['name'].apply(clean_name)
applicants['clean_name'] = applicants['person_name'].apply(clean_name)

print("Standardizing Country Codes...")
# Map Crunchbase (3-letter) to PATSTAT (2-letter)
country_map = {
    'USA': 'US', 'GBR': 'GB', 'DEU': 'DE', 'FRA': 'FR', 
    'CAN': 'CA', 'CHN': 'CN', 'JPN': 'JP', 'KOR': 'KR',
    'ITA': 'IT', 'ESP': 'ES', 'IND': 'IN', 'ISR': 'IL'
}
companies['std_country'] = companies['country'].map(country_map).fillna(companies['country'])

# Handle Missing Countries
companies['std_country'] = companies['std_country'].fillna('UNKNOWN')
applicants['person_ctry_code'] = applicants['person_ctry_code'].fillna('UNKNOWN')

# Prepare pool for matching
unmatched_companies = companies.copy()
unmatched_applicants = applicants.copy() # Can filter this down if needed
all_matches = []

print("Preprocessing complete.")# %% Cell 5: Pass 1 - Exact String Match
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

