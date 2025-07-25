# web_classifier.py

# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import tensorflow as tf
import pandas as pd
import json
import re
import os
import sys

# ✅ Step 1: Load file
input_path = input("📥 Enter the path to your input CSV or JSON file: ").strip()
if not os.path.isfile(input_path):
    raise FileNotFoundError("❌ File does not exist.")

ext = os.path.splitext(input_path)[1].lower()
if ext == '.csv':
    df = pd.read_csv(input_path, encoding='utf-8',
                     on_bad_lines='skip', engine='python')
elif ext == '.json':
    df = pd.read_json(input_path, orient='records')
else:
    raise ValueError("❌ Unsupported file format. Use CSV or JSON.")

# ✅ Step 2: Prepare combined text
fields = ['Class', 'Content', 'ID', 'Name',
          'Tag', 'Xpath', 'label', 'object_name']
for f in fields:
    if f not in df.columns:
        df[f] = ''
df[fields] = df[fields].fillna('')
df['combined'] = df[fields].agg(' '.join, axis=1)

# ✅ Step 3: Semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['combined'].astype(
    str).tolist(), show_progress_bar=True)
anchor_embedding = model.encode(
    ["button input form label field search login submit register password xpath id"])[0]
df['semantic_similarity'] = cosine_similarity(
    [anchor_embedding], embeddings)[0]

# ✅ Step 4: Rule-based scoring


def xpath_depth(xpath):
    return xpath.count('/') if isinstance(xpath, str) else 0


def is_descriptive(text):
    return bool(re.search(r'[a-zA-Z]{3,}', str(text)))


def is_generic(text):
    return bool(re.fullmatch(r'(temp|div|span|input|id)?\d{1,4}', str(text).lower()))


def contains_action_words(text):
    action_words = ['click', 'submit', 'register',
                    'login', 'search', 'add', 'delete', 'save', 'next']
    return any(word in str(text).lower() for word in action_words)


def compute_similarity(t1, t2):
    if not str(t1).strip() or not str(t2).strip():
        return 0.0
    emb1 = model.encode([str(t1)])[0].reshape(1, -1)
    emb2 = model.encode([str(t2)])[0].reshape(1, -1)
    return cosine_similarity(emb1, emb2)[0][0]


def rule_based_score(row):
    score = 0
    tag = str(row['Tag']).lower()
    if tag in ['button', 'a', 'input', 'select', 'textarea', 'label']:
        score += 2
    depth = xpath_depth(row['Xpath'])
    score += (depth >= 5) + (depth >= 7)
    score += 1.5 if contains_action_words(
        row['Content']) or contains_action_words(row['label']) else 0
    score += is_descriptive(row['ID']) * 1
    score += is_descriptive(row['Name']) * 0.5
    score += is_descriptive(row['Class']) * 0.5
    score -= is_generic(row['ID']) or is_generic(row['Name'])
    score += is_descriptive(row['object_name']) * 0.5
    score += 1 if len(str(row['Content']).strip()) >= 5 else 0
    sim_avg = sum([
        compute_similarity(row['label'], row['Content']),
        compute_similarity(row['label'], row['ID']),
        compute_similarity(row['Content'], row['ID'])
    ]) / 3
    score += 1 if sim_avg > 0.5 else 0
    if row['Name'] and all(not row[f] for f in ['Class', 'Content', 'ID', 'Tag', 'Xpath', 'label']):
        if str(row['Name']).lower() in ['username', 'email', 'password', 'search']:
            score += 2
    if row['Content'] and all(not row[f] for f in ['Class', 'ID', 'Name', 'Tag', 'Xpath', 'label']):
        if contains_action_words(row['Content']) or len(str(row['Content']).strip()) >= 5:
            score += 2
    return score


# ✅ Step 5: Filter
df['rule_score'] = df.apply(rule_based_score, axis=1)
df['is_valid'] = (df['semantic_similarity'] >= 0.4) | (df['rule_score'] >= 4)
valid_df = df[df['is_valid']].drop(
    columns=['combined', 'semantic_similarity', 'rule_score', 'is_valid'])

# ✅ Step 6: Save results
out_dir = os.path.dirname(input_path)
folder_name = os.path.splitext(os.path.basename(input_path))[0]
output_dir = os.path.join(out_dir, folder_name)
os.makedirs(output_dir, exist_ok=True)

csv_out = os.path.join(output_dir, "filtered_web_elements.csv")
json_out = os.path.join(output_dir, "filtered_web_elements.json")
valid_df.to_csv(csv_out, index=False)
valid_df.to_json(json_out, orient='records', indent=2)

print(
    f"\n✅ Filtered web elements saved to:\n📄 CSV:  {csv_out}\n📄 JSON: {json_out}")
