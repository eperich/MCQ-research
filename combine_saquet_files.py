import pandas as pd
from pathlib import Path
import re

base_path = Path('.')
folders = ['DataSciencePython', 'ProgrammingInJava', 'ProgrammingInPython3', 'DiscreteMath']

for folder_name in folders:
    folder_path = base_path / folder_name
    if not folder_path.exists():
        continue
    
    print(f'\n=== Processing {folder_name} ===')
    
    # Get all SAQUET files
    saquet_files = [f for f in folder_path.glob('SAQUET_results_*.csv') if 'SME' not in f.name]
    
    # Group files by base topic name (removing numbers at the end)
    topic_groups = {}
    for file in saquet_files:
        topic = file.stem.replace('SAQUET_results_', '')
        # Remove trailing numbers (1, 2, etc.)
        base_topic = re.sub(r'\d+$', '', topic)
        
        if base_topic not in topic_groups:
            topic_groups[base_topic] = []
        topic_groups[base_topic].append(file)
    
    # Combine files that have multiple parts
    for base_topic, files in topic_groups.items():
        if len(files) > 1:
            print(f'Combining {len(files)} files for topic: {base_topic}')
            
            # Read and combine all dataframes
            dfs = []
            for file in sorted(files):
                df = pd.read_csv(file)
                print(f'  - {file.name}: {len(df)} questions')
                dfs.append(df)
            
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Save combined file
            output_name = f'SAQUET_results_{base_topic.rstrip("_")}.csv'
            output_path = folder_path / output_name
            combined_df.to_csv(output_path, index=False)
            print(f'  ✓ Saved combined file: {output_name} ({len(combined_df)} questions)')
            
            # Optionally delete the original split files
            # Uncomment the following lines if you want to delete the originals
            # for file in files:
            #     file.unlink()
            #     print(f'  Deleted: {file.name}')

print('\n✓ Done combining SAQUET files')
