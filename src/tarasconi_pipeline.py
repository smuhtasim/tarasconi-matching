import pandas as pd
import re
import ast
from rapidfuzz import fuzz, process  # ADDED 'process' HERE
from tqdm import tqdm

# 1. Load Data
# ---------------------------------------------------------
print("Loading datasets...")
# Added low_memory=False to fix the DtypeWarning
companies = pd.read_csv('data/companies.csv', low_memory=False)
people = pd.read_csv('data/people.csv', low_memory=False)

applicants = pd.read_parquet('data/applicants.parquet')
inventors = pd.read_parquet('data/inventors.parquet')

# 2. Data Cleaning & Unpacking
# ---------------------------------------------------------
print("Cleaning data formats...")

# FUNCTION: Remove brackets/quotes if data looks like "['Value']"
def unpack_list_string(text):
    if isinstance(text, str) and text.startswith('[') and text.endswith(']'):
        try:
            # Safely evaluate the string as a list and get the first element
            val_list = ast.literal_eval(text)
            if isinstance(val_list, list) and len(val_list) > 0:
                return str(val_list[0])
        except (ValueError, SyntaxError):
            pass
    return text

# Apply unpacking to critical columns in Crunchbase
# This turns "['USA']" -> "USA" and "['Zana']" -> "Zana"
cols_to_unpack = ['name', 'legal_name', 'country', 'region', 'city']
for col in cols_to_unpack:
    if col in companies.columns:
        companies[col] = companies[col].apply(unpack_list_string)

# 3. Name Harmonization Function
# ---------------------------------------------------------
def clean_name(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    # Remove Legal Forms
    legal_forms = [
        r'\binc\b', r'\binc\.', r'\bincorporated\b', 
        r'\bllc\b', r'\bltd\b', r'\blimited\b', 
        r'\bcorp\b', r'\bcorporation\b', r'\bgmbh\b', 
        r'\bco\b', r'\bcompany\b', r'\bs\.a\.\b', r'\bsa\b'
    ]
    regex_legal = '|'.join(legal_forms)
    text = re.sub(regex_legal, '', text)
    
    # Remove Special Characters & Punctuation
    text = re.sub(r'[^\w\s]', '', text) 
    
    # Remove Common Stopwords
    stopwords = [r'\bthe\b', r'\bgroup\b', r'\bholdings\b', r'\binternational\b']
    regex_stop = '|'.join(stopwords)
    text = re.sub(regex_stop, '', text)
    
    # Remove Extra Whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

print("Harmonizing names...")
companies['clean_name'] = companies['name'].apply(clean_name)
applicants['clean_name'] = applicants['person_name'].apply(clean_name)

# 4. Standardize Country Codes
# ---------------------------------------------------------
# PATSTAT uses 2-letter codes (US). Crunchbase often uses 3-letter (USA).
# We need to map CB -> PATSTAT format.
country_map = {
    'USA': 'US', 'GBR': 'GB', 'DEU': 'DE', 'FRA': 'FR', 
    'CAN': 'CA', 'CHN': 'CN', 'JPN': 'JP', 'KOR': 'KR',
    'ITA': 'IT', 'ESP': 'ES', 'IND': 'IN', 'ISR': 'IL'
}

companies['std_country'] = companies['country'].map(country_map).fillna(companies['country'])

# Fill NaNs to avoid matching errors
companies['std_country'] = companies['std_country'].fillna('UNKNOWN')
applicants['person_ctry_code'] = applicants['person_ctry_code'].fillna('UNKNOWN')

print("Preprocessing complete.")
print(f"Sample Company Country: {companies['std_country'].iloc[0]}") # Should print 'US' now, not '["USA"]'

# =========================================================
# MATCHING PIPELINE
# =========================================================

unmatched_companies = companies.copy()
# Optional: Filter PATSTAT to only companies (if you can identify them) to speed up
unmatched_applicants = applicants.copy()

all_matches = []

# PASS 1: Exact String Match
# ---------------------------------------------------------
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
    all_matches.append(exact_matches[['company_id', 'person_id', 'match_type', 'score']])
    
    # Remove matched
    matched_cb_ids = exact_matches['company_id'].unique()
    unmatched_companies = unmatched_companies[~unmatched_companies['company_id'].isin(matched_cb_ids)]

print(f"Exact matches found: {len(exact_matches)}")

# PASS 2: Alphanumeric Match
# ---------------------------------------------------------
print("Running Pass 2: Alphanumeric Match...")

def create_fingerprint(text):
    return re.sub(r'\s+', '', text)

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

    # Remove matched
    matched_cb_ids = alpha_matches['company_id'].unique()
    unmatched_companies = unmatched_companies[~unmatched_companies['company_id'].isin(matched_cb_ids)]

print(f"Alphanumeric matches found: {len(alpha_matches)}")

# PASS 3: Fuzzy Matching (Blocked)
# ---------------------------------------------------------
print("Running Pass 3: Fuzzy Match...")

# Ensure clean_name is string
unmatched_companies['clean_name'] = unmatched_companies['clean_name'].astype(str)
unmatched_applicants['clean_name'] = unmatched_applicants['clean_name'].astype(str)

# Blocking: First 2 chars
unmatched_companies['block'] = unmatched_companies['clean_name'].str[:2]
unmatched_applicants['block'] = unmatched_applicants['clean_name'].str[:2]

fuzzy_results = []

# Group applicants by block for faster lookup
# We convert to a dict of DataFrames
app_blocks = dict(list(unmatched_applicants.groupby('block')))

# Loop through companies
for idx, cb_row in tqdm(unmatched_companies.iterrows(), total=unmatched_companies.shape[0]):
    block = cb_row['block']
    
    if block in app_blocks:
        app_group = app_blocks[block]
        
        # Filter by country first (Strict Tarasconi Rule)
        country_app_group = app_group[app_group['person_ctry_code'] == cb_row['std_country']]
        
        if country_app_group.empty:
            continue
            
        # Extract best matches
        # process.extract returns: [(match_string, score, match_index_label), ...]
        matches = process.extract(
            cb_row['clean_name'], 
            country_app_group['clean_name'], 
            scorer=fuzz.token_sort_ratio, 
            limit=1, 
            score_cutoff=90
        )
        
        for match in matches:
            # match[2] is the INDEX LABEL (original ID) from the dataframe
            matched_idx_label = match[2]
            
            # --- FIX IS HERE: Changed .iloc to .loc ---
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

# Save Step 1 Results
if len(all_matches) > 0:
    final_org_matches = pd.concat(all_matches, ignore_index=True)
    final_org_matches.to_csv("data/step1_organization_matches.csv", index=False)
    print("Step 1 Results saved to data/step1_organization_matches.csv")
else:
    print("No matches found at all. Check data alignment.")