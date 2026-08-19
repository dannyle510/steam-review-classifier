import pandas as pd
import time
from google import genai
from google.genai import errors

# Configure Gemini with new SDK
client = genai.Client(api_key='YOUR_GEMINI_API_KEY_HERE')

# Load labeled reviews
df = pd.read_csv('review_labels.csv')

# Filter to escalations only
escalations = df[df['final_label'] == 'escalation'].copy()
print(f"Total escalations: {len(escalations)}")
print(escalations['game_name'].value_counts())

def generate_brief(game_name):
    game_escalations = escalations[escalations['game_name'] == game_name]
    
    if len(game_escalations) == 0:
        print(f"No escalations found for {game_name}")
        return None
    
    # Limit to top 100 reviews to stay within token limits
    top_reviews = game_escalations.head(100)
    reviews_text = "\n---\n".join(top_reviews['cleaned_review'].tolist())
    
    prompt = f"""
    You are a senior product manager at a game studio.
    
    Below are {len(top_reviews)} player reviews flagged as technical escalations 
    for {game_name}. These are real bug reports and performance issues reported 
    by players.
    
    Analyze these reviews and produce a concise product brief with:
    1. Top 3 engineering priorities (most frequently reported issues)
    2. Severity assessment for each (Critical / High / Medium)
    3. One recommended immediate action per issue
    
    Keep the brief under 300 words. Be specific and actionable.
    
    REVIEWS:
    {reviews_text[:10000]}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=prompt
        )
        return response.text
    except errors.ClientError as e:
        if '429' in str(e):
            print(f"Rate limit hit for {game_name}, waiting 60 seconds...")
            time.sleep(60)
            # retry once
            response = client.models.generate_content(
                model='gemini-3.5-flash-lite',
                contents=prompt
            )
            return response.text
        raise

# Generate briefs for all three games
games = ['helldivers2', 'monster_hunter_wilds', 'borderlands4']
briefs = {}

for game in games:
    print(f"\nGenerating brief for {game}...")
    brief = generate_brief(game)
    briefs[game] = brief
    print(f"\n=== {game.upper()} PRODUCT BRIEF ===")
    print(brief)
    time.sleep(5)  # pause between games to avoid rate limits

# Save briefs locally
with open('product_briefs.txt', 'w', encoding='utf-8') as f:
    for game, brief in briefs.items():
        f.write(f"=== {game.upper()} ===\n\n")
        f.write(brief)
        f.write("\n\n" + "="*50 + "\n\n")

print("\nBriefs saved to product_briefs.txt")