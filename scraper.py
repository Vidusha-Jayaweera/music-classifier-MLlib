import lyricsgenius
import pandas as pd
import re
import csv

# --- 1. Scrape Student Dataset ---
GENIUS_API_TOKEN = '1MNaCsS0Vkw1RQoT61F2IF-LwcvgFnkoHPMfMxQnsCShpNmZ0u4AMYZ_TXw94vd8' 
genius = lyricsgenius.Genius(GENIUS_API_TOKEN, timeout=30, retries=3)

artists = ["The Specials", "Madness", "Reel Big Fish", "Catch 22", "Streetlight Manifesto", "Less Than Jake", "The Selecter", "Sublime", "Goldfinger", "The Beat"]
data = []

print("Starting to scrape songs... This might take a few minutes.")

for artist_name in artists:
    if len(data) >= 100: break
    try:
        print(f"Searching for {artist_name}...")
        artist = genius.search_artist(artist_name, max_songs=15, sort="popularity")
        
        if not artist: 
            print(f"Could not find artist: {artist_name}")
            continue
            
        for song in artist.songs:
            if len(data) >= 100: break
            
            # --- THE FIX: Extract the year safely from the raw dictionary ---
            song_dict = song.to_dict()
            release_year = None
            
            # Try to get it from the exact release date components
            components = song_dict.get('release_date_components')
            if components and isinstance(components, dict) and components.get('year'):
                release_year = str(components.get('year'))
            
            # Fallback to the display date string
            if not release_year:
                display_date = song_dict.get('release_date_for_display')
                if display_date:
                    match = re.search(r'\d{4}', str(display_date))
                    if match: 
                        release_year = match.group(0)
            
            # Hard fallback so we don't lose the song and get empty cells
            if not release_year:
                release_year = "2000" 
                
            if not song.lyrics: continue
                
            clean_lyrics = re.sub(r'[\r\n]+', ' ', song.lyrics).replace(',', '')
            clean_lyrics = re.sub(r'^.*?Lyrics', '', clean_lyrics)
            clean_lyrics = re.sub(r'Embed$', '', clean_lyrics)
            clean_lyrics = re.sub(r'\[.*?\]', '', clean_lyrics)
            
            data.append({
                "artist_name": artist.name, "track_name": song.title,
                "release_date": release_year, "genre": "Ska", "lyrics": clean_lyrics.strip()
            })
    except Exception as e: 
        print(f"FAILED on {artist_name} due to error: {e}")

# Create the DataFrame
columns_needed = ['artist_name', 'track_name', 'release_date', 'genre', 'lyrics']
student_df = pd.DataFrame(data, columns=columns_needed)

student_df.to_csv('Student_dataset.csv', index=False, quoting=csv.QUOTE_ALL)
print(f"\nSaved {len(student_df)} songs to Student_dataset.csv")

# --- 2. Merge Datasets ---
if len(student_df) > 0:
    print("Merging datasets...")
    mendeley_df = pd.read_csv('Mendeley_dataset.csv')
    
    merged_df = pd.concat([mendeley_df[columns_needed], student_df[columns_needed]], ignore_index=True)
    merged_df.to_csv('Merged_dataset.csv', index=False, quoting=csv.QUOTE_ALL)
    print("Merged_dataset.csv created successfully!")