import pandas as pd
import re
import argparse
from pathlib import Path

def create_abbreviation_mapping():
    """
    Create a comprehensive mapping of Latin abbreviations to modern equivalents.
    """
    # Dictionary mapping Latin abbreviations to modern forms
    abbreviations = {
        # Macron substitutions (vowel + macron represents missing nasal)
        'ā': 'an',
        'ē': 'en', 
        'ī': 'in',
        'ō': 'on',
        'ū': 'un',
        
        # Tilde substitutions (similar to macrons)
        'ã': 'an',
        'ẽ': 'en',
        'ĩ': 'in',
        'õ': 'on',
        'ũ': 'un',
        
        # Special Latin abbreviation characters
        'ꝑ': 'per',
        'ꝓ': 'pro', 
        'ꝗ': 'que',
        'ꝙ': 'que',
        'ꝯ': 'con',
        'ꝰ': 'us',
        
        # Abbreviation marks with letters
        'q̄': 'que',
        'q̃': 'que',
        'ꝗ̄': 'que',
        'ꝗ̃': 'que',
        'm̄': 'men',
        'ñ': 'nn',
        
        # Common word abbreviations
        'xp̄o': 'cristo',
        'xpo': 'cristo', 
        'chr̄o': 'cristo',
        'chr̃o': 'cristo',
        'iē': 'iesu',
        'iēs': 'iesus',
        'iē̄': 'iesu',
        'dñs': 'dominus',
        'dē': 'deus',
        'sp̄s': 'spiritus',
        'sč': 'sanctus',
        'scō': 'sancto',
        'scā': 'sancta',
        'scī': 'sancti',
        'scē': 'sancte',
        
        # Contraction marks
        'cō': 'con',
        'cõ': 'con',
        'nō': 'non',
        'nõ': 'non',
        'pē': 'per',
        'prō': 'pro',
        
        # Other common abbreviations
        'ꝺ': 'd',
        'ſ': 's',  # long s
        'ꝭ': 'is',
        'ꝼ': 'f',
        'ł': 'l',  # l with stroke
        
        # Suspension marks
        'et̄': 'etc',
        '&c': 'etc',
        '&c.': 'etc.',
        
        # Common endings with abbreviation marks
        'ꝫ': 'et',
        'z̄': 'et',
        
        # Numbers with abbreviation marks  
        'ꝯ': 'con',
        'xiī': 'xii',
        'xiii̊': 'xiii',
        'xiiij': 'xiiii',
        
        # Prepositions and particles
        'p̄': 'per',
        'ꝑ̃': 'pren',
        'p̃': 'pre',
    }
    
    return abbreviations

def expand_word_patterns():
    """
    Create patterns for word-level abbreviations that need context.
    """
    word_patterns = [
        # Common word contractions
        (r'\bchriſ?to\b', 'cristo'),
        (r'\bieſ?us?\b', 'iesus'),
        (r'\bſ([aeiou])', r's\1'),  # long s at beginning of words
        (r'([aeiou])ſ\b', r'\1s'),  # long s at end of words
        (r'ſ', 's'),  # remaining long s
        
        # Common medieval contractions
        (r'\bq(ue|́|̃|̄)\b', 'que'),
        (r'\bq([uo])(ue|́|̃|̄)', r'qu\1que'),
        
        # Nasal contractions
        (r'([aeiou])(m|n)̃', r'\1\2\2'),  # vowel + nasal + tilde = double nasal
        (r'([aeiou])̃', r'\1n'),  # vowel + tilde = vowel + n
        
        # Specific religious abbreviations
        (r'\bxp̄?o\b', 'cristo'),
        (r'\biē̄?s?\b', 'iesus'),
        (r'\bdom̄?i?n[eu]s?\b', 'dominus'),
        (r'\bdē̄?[iou]s?\b', 'deus'),
        (r'\bsp̄?irit[ūu]s?\b', 'spiritus'),
    ]
    
    return word_patterns

def convert_abbreviations(text, abbreviation_map, word_patterns):
    """
    Convert Latin abbreviations in text to modern equivalents.
    
    Args:
        text (str): Input text with abbreviations
        abbreviation_map (dict): Character-level abbreviation mappings
        word_patterns (list): Word-level pattern replacements
        
    Returns:
        str: Text with abbreviations expanded
    """
    if pd.isna(text) or not isinstance(text, str):
        return text
    
    # Make a copy to work with
    result = text
    
    # Apply character-level substitutions
    for abbrev, expansion in abbreviation_map.items():
        result = result.replace(abbrev, expansion)
    
    # Apply word-level pattern substitutions
    for pattern, replacement in word_patterns:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    # Clean up any remaining combination marks or strange characters
    result = re.sub(r'[̄̃́̊̌̂̆̇̋̀̈̌̋̇]', '', result)  # Remove combining diacriticals
    
    return result

def process_dataset(input_file, output_file=None):
    """
    Process the dataset to convert Latin abbreviations.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file (optional)
    """
    # Read the dataset
    print(f"Reading dataset from {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Create abbreviation mappings
    abbreviation_map = create_abbreviation_mapping()
    word_patterns = expand_word_patterns()
    
    # Process the text column
    print("Converting Latin abbreviations...")
    if 'text' in df.columns:
        df['text_converted'] = df['text'].apply(
            lambda x: convert_abbreviations(x, abbreviation_map, word_patterns)
        )
        
        # Show some examples of conversions
        print("\nExample conversions:")
        for i in range(min(10, len(df))):
            original = df.iloc[i]['text']
            converted = df.iloc[i]['text_converted']
            if pd.notna(original) and original != converted:
                print(f"Original:  {original}")
                print(f"Converted: {converted}")
                print("-" * 50)
    else:
        print("Error: 'text' column not found in dataset!")
        return
    
    # Set output filename if not provided
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_converted{input_path.suffix}"
    
    # Save the converted dataset
    print(f"Saving converted dataset to {output_file}...")
    df.to_csv(output_file, index=False)
    
    print(f"Conversion complete! Converted dataset saved as: {output_file}")
    
    # Show statistics
    if 'text' in df.columns and 'text_converted' in df.columns:
        total_rows = len(df)
        changed_rows = sum(df['text'] != df['text_converted'])
        print(f"\nStatistics:")
        print(f"Total rows: {total_rows}")
        print(f"Rows with changes: {changed_rows}")
        print(f"Percentage changed: {changed_rows/total_rows*100:.1f}%")

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert Latin abbreviations in OCR dataset to modern language"
    )
    parser.add_argument(
        "input_file", 
        help="Path to input CSV dataset file"
    )
    parser.add_argument(
        "-o", "--output", 
        help="Path to output CSV file (default: adds '_converted' to input filename)"
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' does not exist!")
        return 1
    
    try:
        process_dataset(args.input_file, args.output)
        return 0
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 