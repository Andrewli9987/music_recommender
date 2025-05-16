from enum import global_enum_repr

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('tracks.csv')

# Preview basic info
print("Initial shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# Drop rows with missing values (optional: only drop critical columns if needed)
df_clean = df.dropna()

# Drop irrelevant columns (depending on what's in your file — adjust as needed)
columns_to_drop = ['id', 'uri', 'track_href', 'analysis_url', 'type', 'Unnamed: 0']
df_clean = df_clean.drop(columns=[col for col in columns_to_drop if col in df_clean.columns])

# Reset index
df_clean = df_clean.reset_index(drop=True)

# Show cleaned DataFrame info
print("\nCleaned shape:", df_clean.shape)
print(df_clean.head())

# Save cleaned data (optional)
df_clean.to_csv('tracks_cleaned.csv', index=False)

# LIMIT rows to reduce memory usage
df = pd.read_csv('tracks_cleaned.csv').dropna().reset_index(drop=True)

# SAMPLE subset of songs for similarity computation
df = df.sample(n=5000, random_state=42).reset_index(drop=True)

# -------- STEP 1: Select features --------
features = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo'
]

# Drop any rows missing these features
df = df.dropna(subset=features)

# -------- STEP 2: Normalize features --------
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

# -------- STEP 3: Compute cosine similarity --------
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(scaled_features)


# -------- STEP 4: Recommend function --------
def recommend(song_index, num_recommendations=5):
    print(f"\nSelected song: {df.iloc[song_index]['track_name']} by {df.iloc[song_index]['artists']}")
    input_genre = df.iloc[song_index]['track_genre']

    sim_scores = list(enumerate(similarity_matrix[song_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations + 1]

    print("\nRecommended songs:")
    for idx, score in sim_scores:
        name = df.iloc[idx]['track_name']
        artist = df.iloc[idx]['artists']
        genre = df.iloc[idx]['track_genre']
        print(f"- {name} by {artist} [{genre}] (score: {score:.2f})")

    return sim_scores, input_genre


def precision_at_k(recommended_indices, input_genre):
    relevant = 0
    for idx, _ in recommended_indices:
        if df.iloc[idx]['track_genre'] == input_genre:
            relevant += 1
    return relevant / len(recommended_indices)


if __name__ == '__main__':
    song_index = 0  # Try any valid index
    recommended, genre = recommend(song_index=song_index, num_recommendations=5)
    precision = precision_at_k(recommended, genre)
    print(f"\nPrecision@5: {precision:.2f}")


