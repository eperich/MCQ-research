"""
Combine split SAQUET files that end with 1 or 2.
"""
import pandas as pd
from pathlib import Path
import re

base_path = Path('.')
disciplines = ['DataSciencePython', 'ProgrammingInJava', 'ProgrammingInPython3', 'DiscreteMath']

for discipline in disciplines:
    discipline_path = base_path / discipline
    if not discipline_path.exists():
        continue
    
    print(f"\n{discipline}:")
    
    # Find all SAQUET files (AI and SME)
    for pattern in ['SAQUET_results_', 'SME_SAQUET_results_']:
        files = list(discipline_path.glob(f'{pattern}*.csv'))
        
        # Group files by base name (without the trailing 1 or 2)
        file_groups = {}
        for file in files:
            # Check if filename ends with 1 or 2
            if re.search(r'[12]\.csv$', file.name):
                # Get base name without the number
                base_name = re.sub(r'[12]\.csv$', '', file.name)
                if base_name not in file_groups:
                    file_groups[base_name] = []
                file_groups[base_name].append(file)
        
        # Combine files in each group
        for base_name, group_files in file_groups.items():
            if len(group_files) > 1:
                # Sort to ensure 1 comes before 2
                group_files.sort()
                
                print(f"  Combining: {[f.name for f in group_files]}")
                
                # Read and combine all parts
                dfs = []
                for file in group_files:
                    df = pd.read_csv(file)
                    dfs.append(df)
                
                combined_df = pd.concat(dfs, ignore_index=True)
                
                # Save combined file
                output_name = base_name + '.csv'
                output_path = discipline_path / output_name
                combined_df.to_csv(output_path, index=False)
                print(f"    → Created {output_name} ({len(combined_df)} questions)")
                
                # Delete the split files
                for file in group_files:
                    file.unlink()
                    print(f"    → Deleted {file.name}")

print("\n✓ All split files combined!")
