import pandas as pd
from pathlib import Path
import re
from io import StringIO

base_path = Path('.')
disciplines = {
    'DataSciencePython': 'Data Science',
    'ProgrammingInJava': 'Computer Science',
    'ProgrammingInPython3': 'Computer Science',
    'DiscreteMath': 'Discrete Mathematics'
}

lo_data = []

for folder_name, discipline in disciplines.items():
    folder_path = base_path / folder_name
    if not folder_path.exists():
        continue
    
    analysis_files = list(folder_path.glob('other_analysis_*.csv'))
    
    for file in analysis_files:
        try:
            topic = file.stem.replace('other_analysis_', '')
            topic = re.sub(r'_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}.*$', '', topic)
            
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'Question Alignment Analysis' not in content:
                continue
            
            start_marker = 'Question Alignment Analysis'
            section_start = content.find(start_marker)
            if section_start == -1:
                continue
            
            section_content = content[section_start:]
            lines = section_content.split('\n')
            
            csv_lines = []
            found_header = False
            for line in lines:
                if 'Question ID,Question Text' in line:
                    found_header = True
                    continue
                if found_header:
                    if line.strip() == '' or '===' in line or 'Summary' in line:
                        break
                    csv_lines.append(line)
            
            if csv_lines:
                csv_string = '\n'.join(csv_lines)
                temp_df = pd.read_csv(StringIO(csv_string), 
                                    names=['question_id', 'question_text', 'run_number', 'aligned', 'explanation'])
                
                for _, row in temp_df.iterrows():
                    lo_data.append({
                        'book': folder_name,
                        'discipline': discipline,
                        'topic': topic,
                        'run_number': row['run_number'],
                        'question_text': row['question_text'].strip(),
                        'lo_aligned': str(row['aligned']).strip().lower() == 'yes'
                    })
        
        except Exception as e:
            print(f'Error in {file.name}: {e}')

df = pd.DataFrame(lo_data)
print(f'Total loaded: {len(df)}')
print(f'Unique by book+topic+text: {len(df.drop_duplicates(subset=["book", "topic", "question_text"]))}')
print(f'Unique by book+topic+run+text: {len(df.drop_duplicates(subset=["book", "topic", "run_number", "question_text"]))}')

# Check run numbers
print(f'\nRun numbers in data: {sorted(df["run_number"].unique())}')

# Check how many questions have multiple run numbers
print('\nQuestions appearing in multiple runs:')
for (book, topic, text), group in df.groupby(['book', 'topic', 'question_text']):
    if len(group['run_number'].unique()) > 1:
        print(f'{book}/{topic}: "{text[:60]}..." appears in runs {sorted(group["run_number"].unique())}')
