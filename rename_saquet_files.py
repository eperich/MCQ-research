from pathlib import Path

base_path = Path('.')

# Define renaming mappings for each folder
renamings = {
    'ProgrammingInJava': {
        'SAQUET_results_Arithmetic_expressions_int.csv': 'SAQUET_results_Arithmetic expressions (int).csv',
        'SAQUET_results_Floating point numbers (double).csv': 'SAQUET_results_Floating-point (double).csv',
        'SAQUET_results_Using math methods.csv': 'SAQUET_results_Math methods.csv'
    },
    'ProgrammingInPython3': {
        'SAQUET_results_Floating point.csv': 'SAQUET_results_Floating-point.csv',
        'SAQUET_results_Programming in Python.csv': 'SAQUET_results_Programming using Python.csv',
        'SAQUET_results_User defined function basics.csv': 'SAQUET_results_Function basics.csv'
    },
    'DiscreteMath': {
        'SAQUET_results_Laws of propositional logic.csv': 'SAQUET_results_Laws of propositional  logic.csv',
        'SAQUET_results_Set of sets.csv': 'SAQUET_results_Sets of sets.csv'
    }
}

for folder_name, rename_map in renamings.items():
    folder_path = base_path / folder_name
    if not folder_path.exists():
        print(f'Folder not found: {folder_name}')
        continue
    
    print(f'\n=== {folder_name} ===')
    for old_name, new_name in rename_map.items():
        old_path = folder_path / old_name
        new_path = folder_path / new_name
        
        if old_path.exists():
            old_path.rename(new_path)
            print(f'✓ Renamed: {old_name} -> {new_name}')
        else:
            print(f'✗ File not found: {old_name}')

print('\n✓ Done renaming files')
