"""
Composite Efficacy Analysis: Combines SAQUET Quality + LO Alignment Relevance

This script merges quality evaluation (SAQUET) with relevance evaluation (LO alignment)
to create a comprehensive efficacy metric for AI-generated MCQs.

Efficacy Categories:
1. Classroom-Ready: High quality (≤1 SAQUET failure) AND LO-aligned
2. Quality Issues: LO-aligned but poor quality (≥2 SAQUET failures)
3. Relevance Issues: High quality but NOT LO-aligned
4. Major Revision: Poor quality AND NOT LO-aligned
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Publication-quality styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 100,
    'savefig.dpi': 600,
    'axes.linewidth': 0.8
})

# WCAG-compliant colors
COLORS = {
    'classroom_ready': '#06A77D',    # Teal - ideal
    'refine_quality': '#F77F00',     # Orange - needs quality work
    'realign_content': '#118AB2',    # Blue - needs relevance work
    'major_revision': '#E63946'      # Red - needs major work
}

CRITERIA_COLUMNS = [
    'implausible_distractors', 'none_of_the_above', 'all_of_the_above',
    'true_or_false', 'absolute_terms', 'longest_answer_correct',
    'negative_worded_stem', 'word_repeats_in_stem_and_correct_answer',
    'avoid_logical_cues', 'lost_sequence', 'more_than_one_correct',
    'complex_k_type', 'ambiguous_unclear_information',
    'gratuitous_information_in_stem', 'avoid_convergence_cues',
    'grammatical_cues_in_stem', 'vague_terms', 'unfocused_stem'
]


class CompositeEfficacyAnalyzer:
    """Analyzes combined quality (SAQUET) and relevance (LO alignment)."""
    
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.disciplines = {
            'DataSciencePython': 'Data Science',
            'ProgrammingInJava': 'Computer Science',
            'ProgrammingInPython3': 'Computer Science',
            'DiscreteMath': 'Discrete Mathematics'
        }
        self.merged_data = None
    
    def load_and_merge_data(self):
        """Load quality and LO alignment data, then merge."""
        print("\n=== Loading Quality and Relevance Data ===")
        
        quality_data = self._load_quality_data()
        lo_data = self._load_lo_alignment_data()
        
        print(f"✓ Loaded {len(quality_data)} questions with quality data")
        print(f"✓ Loaded {len(lo_data)} questions with LO alignment data")
        
        # Remove duplicates from both datasets (keep first occurrence)
        quality_data = quality_data.drop_duplicates(subset=['book', 'discipline', 'topic', 'question_text'], keep='first')
        lo_data = lo_data.drop_duplicates(subset=['book', 'discipline', 'topic', 'run_number', 'question_text'], keep='first')
        print(f"✓ After deduplication: {len(quality_data)} unique quality, {len(lo_data)} unique LO alignment questions")
        
        # Merge - keep all LO alignment questions, join quality where available
        merged = pd.merge(
            lo_data,
            quality_data,
            on=['book', 'discipline', 'topic', 'question_text'],
            how='left'
        )
        
        # For questions without quality data, mark as failing quality
        merged['quality_failures'].fillna(99, inplace=True)  # High number = missing data treated as fail
        merged['quality_acceptable'].fillna(False, inplace=True)
        
        print(f"✓ Successfully merged {len(merged)} questions")
        
        # Calculate composite efficacy category
        merged['efficacy_category'] = merged.apply(self._categorize_efficacy, axis=1)
        
        self.merged_data = merged
        return merged
    
    def _load_quality_data(self):
        """Load SAQUET quality evaluation results."""
        quality_data = []
        
        for folder_name, discipline in self.disciplines.items():
            folder_path = self.base_path / folder_name
            if not folder_path.exists():
                continue
            
            # Find SAQUET result files (AI-generated only)
            saquet_files = [f for f in folder_path.glob('SAQUET_results_*.csv')
                          if 'SME' not in f.name]
            
            for file in saquet_files:
                try:
                    df = pd.read_csv(file)
                    
                    # Extract topic from filename
                    topic = file.stem.replace('SAQUET_results_', '')
                    
                    # Count failures per question
                    for idx, row in df.iterrows():
                        failures = sum(1 for col in CRITERIA_COLUMNS 
                                     if col in df.columns and row[col] == 'Fail')
                        
                        quality_data.append({
                            'book': folder_name,
                            'discipline': discipline,
                            'topic': topic,
                            'question_text': row['text'],
                            'quality_failures': failures,
                            'quality_acceptable': failures <= 1
                        })
                
                except Exception as e:
                    print(f"Warning: Error loading {file.name}: {e}")
        
        return pd.DataFrame(quality_data)
    
    def _load_lo_alignment_data(self):
        """Load LO alignment evaluation results."""
        lo_data = []
        
        for folder_name, discipline in self.disciplines.items():
            folder_path = self.base_path / folder_name
            if not folder_path.exists():
                continue
            
            # Find other_analysis files
            analysis_files = list(folder_path.glob('other_analysis_*.csv'))
            
            for file in analysis_files:
                try:
                    # Extract topic from filename - remove timestamp
                    topic = file.stem.replace('other_analysis_', '')
                    # Remove timestamp pattern (e.g., _2026-01-16T14-46-16)
                    import re
                    topic = re.sub(r'_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}.*$', '', topic)
                    
                    # Read entire file to find the section
                    with open(file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Find Question Alignment Analysis section
                    if 'Question Alignment Analysis' not in content:
                        continue
                    
                    # Find start and end of section
                    start_marker = 'Question Alignment Analysis'
                    section_start = content.find(start_marker)
                    if section_start == -1:
                        continue
                    
                    # Extract just the alignment section
                    section_content = content[section_start:]
                    lines = section_content.split('\n')
                    
                    # Find the CSV data (skip header)
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
                    
                    # Parse CSV lines properly using pandas
                    if csv_lines:
                        from io import StringIO
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
                    print(f"Warning: Error loading {file.name}: {e}")
        
        return pd.DataFrame(lo_data)
    
    def _categorize_efficacy(self, row):
        """Categorize question efficacy based on quality and relevance."""
        if row['quality_acceptable'] and row['lo_aligned']:
            return 'Classroom-Ready'
        elif not row['quality_acceptable'] and row['lo_aligned']:
            return 'Refine Quality'
        elif row['quality_acceptable'] and not row['lo_aligned']:
            return 'Realign Content'
        else:
            return 'Major Revision'
    
    def calculate_efficacy_metrics(self):
        """Calculate efficacy metrics overall and by discipline."""
        print("\n=== Calculating Efficacy Metrics ===")
        
        # Overall metrics
        total = len(self.merged_data)
        overall_counts = self.merged_data['efficacy_category'].value_counts()
        
        overall_df = pd.DataFrame({
            'Category': ['Classroom-Ready', 'Refine Quality', 'Realign Content', 'Major Revision'],
            'Count': [overall_counts.get(cat, 0) for cat in 
                     ['Classroom-Ready', 'Refine Quality', 'Realign Content', 'Major Revision']],
            'Percentage': [overall_counts.get(cat, 0) / total * 100 for cat in 
                          ['Classroom-Ready', 'Refine Quality', 'Realign Content', 'Major Revision']]
        })
        
        # By discipline
        discipline_data = []
        for discipline in ['Computer Science', 'Data Science', 'Discrete Mathematics']:
            disc_data = self.merged_data[self.merged_data['discipline'] == discipline]
            disc_total = len(disc_data)
            
            if disc_total == 0:
                continue
            
            counts = disc_data['efficacy_category'].value_counts()
            
            discipline_data.append({
                'Discipline': discipline,
                'Total_Questions': disc_total,
                'Classroom_Ready': counts.get('Classroom-Ready', 0),
                'Classroom_Ready_%': round(counts.get('Classroom-Ready', 0) / disc_total * 100, 2),
                'Refine_Quality': counts.get('Refine Quality', 0),
                'Refine_Quality_%': round(counts.get('Refine Quality', 0) / disc_total * 100, 2),
                'Realign_Content': counts.get('Realign Content', 0),
                'Realign_Content_%': round(counts.get('Realign Content', 0) / disc_total * 100, 2),
                'Major_Revision': counts.get('Major Revision', 0),
                'Major_Revision_%': round(counts.get('Major Revision', 0) / disc_total * 100, 2)
            })
        
        discipline_df = pd.DataFrame(discipline_data)
        
        # Save to CSV
        overall_df.to_csv(self.base_path / 'table_composite_efficacy_overall.csv', index=False)
        discipline_df.to_csv(self.base_path / 'table_composite_efficacy_by_discipline.csv', index=False)
        
        print("✓ Saved table_composite_efficacy_overall.csv")
        print("✓ Saved table_composite_efficacy_by_discipline.csv")
        
        return overall_df, discipline_df
    
    def create_visualizations(self):
        """Create composite efficacy visualizations."""
        print("\n=== Creating Composite Efficacy Visualizations ===")
        
        # 1. Overall efficacy pie chart
        self._create_overall_pie_chart()
        
        # 2. Efficacy by discipline stacked bar
        self._create_discipline_stacked_bar()
        
        # 3. Efficacy funnel chart
        self._create_efficacy_funnel()
        
        # 4. Quality vs Relevance scatter
        self._create_quality_relevance_matrix()
        
        # 5. Classroom-ready rate comparison
        self._create_classroom_ready_comparison()
        
        print("✓ All composite efficacy visualizations complete")
    
    def _create_overall_pie_chart(self):
        """Create pie chart of overall efficacy distribution."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        counts = self.merged_data['efficacy_category'].value_counts()
        categories = ['Classroom-Ready', 'Refine Quality', 'Realign Content', 'Major Revision']
        values = [counts.get(cat, 0) for cat in categories]
        colors = [COLORS['classroom_ready'], COLORS['refine_quality'], 
                 COLORS['realign_content'], COLORS['major_revision']]
        
        wedges, texts, autotexts = ax.pie(
            values,
            labels=categories,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            explode=(0.05, 0, 0, 0),
            wedgeprops={'edgecolor': 'black', 'linewidth': 1}
        )
        
        # Style the text
        for text in texts:
            text.set_fontsize(11)
            text.set_fontweight('bold')
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        
        ax.set_title('Overall Question Efficacy Distribution\n(Quality + Relevance Combined)',
                    fontsize=13, fontweight='bold', pad=20)
        
        plt.tight_layout()
        filename = 'composite_efficacy_overall.png'
        plt.savefig(self.base_path / filename, dpi=600, bbox_inches='tight', facecolor='white')
        plt.savefig(self.base_path / filename.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.savefig(self.base_path / filename.replace('.png', '.eps'), bbox_inches='tight', facecolor='white', format='eps')
        plt.close()
        print(f"✓ Saved {filename} (PNG, PDF, EPS)")
    
    def _create_discipline_stacked_bar(self):
        """Create stacked bar chart by discipline."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        
        disciplines = ['Computer Science', 'Data Science', 'Discrete Mathematics']
        categories = ['Classroom-Ready', 'Refine Quality', 'Realign Content', 'Major Revision']
        
        # Prepare data
        data_by_discipline = {}
        for discipline in disciplines:
            disc_data = self.merged_data[self.merged_data['discipline'] == discipline]
            total = len(disc_data)
            counts = disc_data['efficacy_category'].value_counts()
            data_by_discipline[discipline] = [counts.get(cat, 0) / total * 100 for cat in categories]
        
        x = np.arange(len(disciplines))
        width = 0.6
        
        # Create stacked bars
        bottom = np.zeros(len(disciplines))
        bars = []
        
        color_map = {
            'Classroom-Ready': 'classroom_ready',
            'Refine Quality': 'refine_quality',
            'Realign Content': 'realign_content',
            'Major Revision': 'major_revision'
        }
        
        for i, category in enumerate(categories):
            values = [data_by_discipline[disc][i] for disc in disciplines]
            color_key = color_map[category]
            bar = ax.bar(x, values, width, label=category, bottom=bottom,
                        color=COLORS[color_key], edgecolor='black', linewidth=1)
            bars.append(bar)
            
            # Add percentage labels
            for j, (rect, val) in enumerate(zip(bar, values)):
                if val > 3:  # Only show label if segment is large enough
                    height = rect.get_height()
                    ax.text(rect.get_x() + rect.get_width()/2., bottom[j] + height/2.,
                           f'{val:.1f}%', ha='center', va='center',
                           fontsize=9, fontweight='bold', color='white')
            
            bottom += values
        
        ax.set_ylabel('Percentage of Questions (%)', fontsize=12)
        ax.set_xlabel('Discipline', fontsize=12)
        ax.set_title('Question Efficacy by Discipline\n(Combined Quality and Relevance Assessment)',
                    fontsize=13, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(disciplines, fontsize=11)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=True, fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        filename = 'composite_efficacy_by_discipline.png'
        plt.savefig(self.base_path / filename, dpi=600, bbox_inches='tight', facecolor='white')
        plt.savefig(self.base_path / filename.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.savefig(self.base_path / filename.replace('.png', '.eps'), bbox_inches='tight', facecolor='white', format='eps')
        plt.close()
        print(f"✓ Saved {filename} (PNG, PDF, EPS)")
    
    def _create_efficacy_funnel(self):
        """Create funnel chart showing filtering effect."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        total = len(self.merged_data)
        lo_aligned = self.merged_data['lo_aligned'].sum()
        quality_acceptable = self.merged_data['quality_acceptable'].sum()
        classroom_ready = len(self.merged_data[self.merged_data['efficacy_category'] == 'Classroom-Ready'])
        
        stages = ['Total Generated', 'LO Aligned', 'Quality Acceptable', 'Classroom-Ready']
        values = [total, lo_aligned, quality_acceptable, classroom_ready]
        percentages = [100, lo_aligned/total*100, quality_acceptable/total*100, classroom_ready/total*100]
        
        # Create horizontal funnel
        y = np.arange(len(stages))
        colors_grad = ['#E63946', '#F77F00', '#118AB2', '#06A77D']
        
        bars = ax.barh(y, values, color=colors_grad, edgecolor='black', linewidth=1)
        
        ax.set_yticks(y)
        ax.set_yticklabels(stages, fontsize=11)
        ax.set_xlabel('Number of Questions', fontsize=12)
        ax.set_title('Question Efficacy Funnel\n(Showing filtering effect of quality and relevance criteria)',
                    fontsize=13, fontweight='bold', pad=20)
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Add value and percentage labels
        for i, (bar, val, pct) in enumerate(zip(bars, values, percentages)):
            width = bar.get_width()
            ax.text(width + 10, bar.get_y() + bar.get_height()/2.,
                   f'{val} ({pct:.1f}%)', ha='left', va='center',
                   fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        filename = 'composite_efficacy_funnel.png'
        plt.savefig(self.base_path / filename, dpi=600, bbox_inches='tight', facecolor='white')
        plt.savefig(self.base_path / filename.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.savefig(self.base_path / filename.replace('.png', '.eps'), bbox_inches='tight', facecolor='white', format='eps')
        plt.close()
        print(f"✓ Saved {filename} (PNG, PDF, EPS)")
    
    def _create_quality_relevance_matrix(self):
        """Create bar chart showing composite efficacy categories."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)
        
        # Count questions in each category
        category_counts = self.merged_data['efficacy_category'].value_counts()
        
        # Define category order (ascending) and labels
        category_mapping = {
            'Major Revision': 'Rewrite',
            'Realign Content': 'Realign Content',
            'Refine Quality': 'Refine Quality',
            'Classroom-Ready': 'Classroom-Ready'
        }
        categories_internal = ['Major Revision', 'Realign Content', 'Refine Quality', 'Classroom-Ready']
        categories_display = [category_mapping[cat] for cat in categories_internal]
        counts = [category_counts.get(cat, 0) for cat in categories_internal]
        colors = [COLORS['major_revision'], COLORS['realign_content'], 
                 COLORS['refine_quality'], COLORS['classroom_ready']]
        
        # Create bar chart
        bars = ax.bar(categories_display, counts, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # Add count labels on top of bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            percentage = (count / len(self.merged_data)) * 100
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}\n({percentage:.1f}%)',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Set up axes
        ax.set_ylabel('Number of Questions', fontsize=13, fontweight='bold', labelpad=10)
        ax.set_xlabel('Composite Efficacy Category', fontsize=13, fontweight='bold', labelpad=10)
        ax.set_title('Composite Efficacy Distribution\n(Quality + Relevance Combined)',
                    fontsize=14, fontweight='bold', pad=15)
        
        # Rotate x-axis labels for better readability
        ax.set_xticklabels(categories_display, rotation=0, ha='center', fontsize=11)
        
        # Add gridlines for easier reading
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        
        # Set y-axis to start at 0 with some headroom
        ax.set_ylim(0, max(counts) * 1.15)
        
        filename = 'composite_efficacy_matrix.png'
        plt.savefig(self.base_path / filename, dpi=600, bbox_inches='tight', facecolor='white')
        plt.savefig(self.base_path / filename.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.savefig(self.base_path / filename.replace('.png', '.eps'), bbox_inches='tight', facecolor='white', format='eps')
        plt.close()
        print(f"✓ Saved {filename} (PNG, PDF, EPS)")
    
    def _create_classroom_ready_comparison(self):
        """Create comparison of classroom-ready rates by discipline."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        
        disciplines = ['Computer Science', 'Data Science', 'Discrete Mathematics']
        
        # Calculate rates
        classroom_ready_rates = []
        lo_aligned_rates = []
        quality_acceptable_rates = []
        
        for discipline in disciplines:
            disc_data = self.merged_data[self.merged_data['discipline'] == discipline]
            total = len(disc_data)
            
            classroom_ready_rates.append(
                len(disc_data[disc_data['efficacy_category'] == 'Classroom-Ready']) / total * 100
            )
            lo_aligned_rates.append(disc_data['lo_aligned'].sum() / total * 100)
            quality_acceptable_rates.append(disc_data['quality_acceptable'].sum() / total * 100)
        
        x = np.arange(len(disciplines))
        width = 0.25
        
        bars1 = ax.bar(x - width, lo_aligned_rates, width, label='LO Aligned',
                      color='#118AB2', edgecolor='black', linewidth=1)
        bars2 = ax.bar(x, quality_acceptable_rates, width, label='Quality Acceptable',
                      color='#F77F00', edgecolor='black', linewidth=1)
        bars3 = ax.bar(x + width, classroom_ready_rates, width, label='Classroom-Ready (Both)',
                      color='#06A77D', edgecolor='black', linewidth=1)
        
        ax.set_ylabel('Percentage of Questions (%)', fontsize=12)
        ax.set_xlabel('Discipline', fontsize=12)
        ax.set_title('Classroom-Ready Rate vs Individual Criteria\n(Combined efficacy compared to individual components)',
                    fontsize=13, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(disciplines, fontsize=11)
        ax.set_ylim(0, 105)
        ax.legend(frameon=True, fontsize=10, loc='lower right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        filename = 'composite_efficacy_classroom_ready.png'
        plt.savefig(self.base_path / filename, dpi=600, bbox_inches='tight', facecolor='white')
        plt.savefig(self.base_path / filename.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.savefig(self.base_path / filename.replace('.png', '.eps'), bbox_inches='tight', facecolor='white', format='eps')
        plt.close()
        print(f"✓ Saved {filename} (PNG, PDF, EPS)")


def main():
    """Main execution."""
    print("=" * 70)
    print("Composite Efficacy Analysis - Quality + Relevance")
    print("=" * 70)
    
    base_path = Path(__file__).parent
    analyzer = CompositeEfficacyAnalyzer(base_path)
    
    # Load and merge data
    print("\n[1/3] Loading and merging quality and relevance data...")
    merged_data = analyzer.load_and_merge_data()
    
    # Calculate metrics
    print("\n[2/3] Calculating composite efficacy metrics...")
    overall_df, discipline_df = analyzer.calculate_efficacy_metrics()
    
    # Create visualizations
    print("\n[3/3] Creating composite efficacy visualizations...")
    analyzer.create_visualizations()
    
    # Print summary
    print("\n" + "=" * 70)
    print("COMPOSITE EFFICACY SUMMARY")
    print("=" * 70)
    
    print("\n1. OVERALL EFFICACY DISTRIBUTION")
    print(overall_df.to_string(index=False))
    
    print("\n\n2. EFFICACY BY DISCIPLINE")
    print(discipline_df.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("✓ Analysis complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  Visualizations:")
    print("    - composite_efficacy_overall.png/pdf/eps")
    print("    - composite_efficacy_by_discipline.png/pdf/eps")
    print("    - composite_efficacy_funnel.png/pdf/eps")
    print("    - composite_efficacy_matrix.png/pdf/eps")
    print("    - composite_efficacy_classroom_ready.png/pdf/eps")
    print("  Data Tables:")
    print("    - table_composite_efficacy_overall.csv")
    print("    - table_composite_efficacy_by_discipline.csv")
    print(f"\nAll files saved to: {base_path}")


if __name__ == '__main__':
    main()
