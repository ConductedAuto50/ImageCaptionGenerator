import re
import csv
from pathlib import Path


def remove_emojis(text):
    """Remove emojis from text"""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FAFF"  # extended symbols
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)


def remove_mentions(text):
    """Remove @mentions from text"""
    return re.sub(r'@\w+', '', text)


def remove_urls(text):
    """Remove URLs from text"""
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    text = url_pattern.sub('', text)
    # Also remove www. URLs
    text = re.sub(r'www\.[^\s]+', '', text)
    return text


def remove_hashtags(text):
    """Remove #hashtags from text"""
    return re.sub(r'#\w+', '', text)


def clean_caption(caption):
    """Apply all cleaning operations to a caption"""
    if not caption:
        return ""
    
    # Apply all cleaning steps
    caption = remove_emojis(caption)
    caption = remove_mentions(caption)
    caption = remove_urls(caption)
    caption = remove_hashtags(caption)
    
    # Clean up extra whitespace
    caption = re.sub(r'\s+', ' ', caption)
    caption = caption.strip()
    
    return caption


def clean_csv_file(input_csv, output_csv):
    """Clean a CSV file by removing unwanted content and blank captions"""
    cleaned_rows = []
    total_rows = 0
    removed_rows = 0
    
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            total_rows += 1
            
            # Clean the caption
            if 'Caption' in row:
                original_caption = row['Caption']
                cleaned_caption = clean_caption(original_caption)
                
                # Skip if caption is blank after cleaning
                if not cleaned_caption:
                    removed_rows += 1
                    continue
                
                row['Caption'] = cleaned_caption
                cleaned_rows.append(row)
    
    # Write cleaned data to output CSV
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)
    
    print(f"Processed {input_csv}")
    print(f"  Total rows: {total_rows}")
    print(f"  Removed rows (blank captions): {removed_rows}")
    print(f"  Kept rows: {len(cleaned_rows)}")
    print(f"  Saved to: {output_csv}\n")


def clean_all_datasets(dataset_folder="datasets"):
    """Clean all CSV files in the datasets folder"""
    dataset_path = Path(dataset_folder)
    
    if not dataset_path.exists():
        print(f"Error: {dataset_folder} directory not found")
        return
    
    # Find all CSV files
    csv_files = list(dataset_path.rglob('*.csv'))
    
    if not csv_files:
        print("No CSV files found in the dataset folder")
        return
    
    print(f"Found {len(csv_files)} CSV file(s) to clean\n")
    
    for csv_file in csv_files:
        # Create output filename with _cleaned suffix
        output_file = csv_file.parent / f"{csv_file.stem}_cleaned{csv_file.suffix}"
        clean_csv_file(csv_file, output_file)
    
    print("All datasets cleaned successfully!")


if __name__ == "__main__":
    clean_all_datasets()
