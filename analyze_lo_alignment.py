"""
Analyze Learning Outcome (LO) alignment for AI-generated MCQs.

Examines:
1. Overall LO alignment rates
2. Alignment by discipline
3. Correlations with other quality metrics
4. Detailed list of non-aligned questions with justifications
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Publication-quality settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.edgecolor'] = '#000000'
plt.rcParams['axes.labelcolor'] = '#000000'
plt.rcParams['text.color'] = '#000000'
plt.rcParams['xtick.color'] = '#000000'
plt.rcParams['ytick.color'] = '#000000'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.edgecolor'] = '#000000'

# Modern color palette
COLORS = {
    'aligned': '#06A77D',      # Teal - good alignment
    'not_aligned': '#E63946',  # Coral red - not aligned
    'discipline1': '#118AB2',  # Ocean blue - Data Science
    'discipline2': '#6A4C93',  # Royal purple - Computer Science
    'discipline3': '#D62828',  # Deep red - Discrete Math
}

class LOAlignmentAnalyzer:
    """Analyze LO alignment from other_analysis files."""
    
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.alignment_data = None
        
    def load_all_files(self):
        """Load LO alignment data from all other_analysis files."""
        print("\n[Loading LO Alignment Data]")
        
        all_alignment = []
        
        disciplines = {
            'DataSciencePython': 'Data Science',
            'ProgrammingInJava': 'Computer Science',
            'ProgrammingInPython3': 'Computer Science',
            'DiscreteMath': 'Discrete Mathematics'
        }
        
        for folder, discipline in disciplines.items():
            folder_path = self.base_path / folder
            if not folder_path.exists():
                continue
            
            files = list(folder_path.glob('other_analysis_*.csv'))
            print(f"  {folder}: {len(files)} files")
            
            for file in files:
                try:
                    # Extract topic name
                    topic = file.stem.replace('other_analysis_', '').split('_202')[0]
                    
                    # Read the file as text to find the alignment section
                    with open(file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Find the "Question Alignment Analysis" section
                    alignment_start_idx = None
                    for idx, line in enumerate(lines):
                        if 'Question Alignment Analysis' in line or 'Question ID,Question Text,Run Number,Aligned' in line:
                            # Found the section header or the column headers
                            if 'Question ID' in line:
                                alignment_start_idx = idx
                            else:
                                # Next line should have headers
                                alignment_start_idx = idx + 1
                            break
                    
                    if alignment_start_idx is None:
                        continue
                    
                    # Read from this point forward
                    alignment_lines = lines[alignment_start_idx:]
                    
                    # Create a temporary file-like object
                    from io import StringIO
                    alignment_text = ''.join(alignment_lines)
                    alignment_df = pd.read_csv(StringIO(alignment_text))
                    
                    # Clean up the dataframe
                    alignment_df = alignment_df.dropna(subset=['Question ID'])
                    
                    # Add metadata
                    alignment_df['discipline'] = discipline
                    alignment_df['topic'] = topic
                    alignment_df['source_folder'] = folder
                    
                    # Normalize alignment column
                    alignment_df['is_aligned'] = alignment_df['Aligned (Yes/No)'].str.lower().str.strip() == 'yes'
                    
                    all_alignment.append(alignment_df)
                    
                except Exception as e:
                    print(f"    Error loading {file.name}: {e}")
        
        if all_alignment:
            self.alignment_data = pd.concat(all_alignment, ignore_index=True)
            print(f"\n  Total questions loaded: {len(self.alignment_data)}")
            print(f"  Aligned: {self.alignment_data['is_aligned'].sum()} ({self.alignment_data['is_aligned'].sum()/len(self.alignment_data)*100:.1f}%)")
            print(f"  Not aligned: {(~self.alignment_data['is_aligned']).sum()} ({(~self.alignment_data['is_aligned']).sum()/len(self.alignment_data)*100:.1f}%)")
            return self.alignment_data
        else:
            return pd.DataFrame()
    
    def export_data_tables(self, output_dir=None):
        """Export all LO alignment data to CSV files."""
        if output_dir is None:
            output_dir = self.base_path
        else:
            output_dir = Path(output_dir)
        
        print("\n=== Exporting LO Alignment Tables ===")
        
        # 1. Overall summary
        overall_stats = {
            'Total_Questions': len(self.alignment_data),
            'Aligned_Questions': self.alignment_data['is_aligned'].sum(),
            'Not_Aligned_Questions': (~self.alignment_data['is_aligned']).sum(),
            'Alignment_Rate_%': round(self.alignment_data['is_aligned'].sum() / len(self.alignment_data) * 100, 2)
        }
        overall_df = pd.DataFrame([overall_stats])
        overall_df.to_csv(output_dir / 'lo_alignment_overall_summary.csv', index=False)
        print("✓ Saved lo_alignment_overall_summary.csv")
        
        # 2. By discipline
        discipline_stats = []
        for discipline in self.alignment_data['discipline'].unique():
            disc_data = self.alignment_data[self.alignment_data['discipline'] == discipline]
            discipline_stats.append({
                'Discipline': discipline,
                'Total_Questions': len(disc_data),
                'Aligned_Questions': disc_data['is_aligned'].sum(),
                'Not_Aligned_Questions': (~disc_data['is_aligned']).sum(),
                'Alignment_Rate_%': round(disc_data['is_aligned'].sum() / len(disc_data) * 100, 2)
            })
        
        discipline_df = pd.DataFrame(discipline_stats).sort_values('Alignment_Rate_%', ascending=False)
        discipline_df.to_csv(output_dir / 'lo_alignment_by_discipline.csv', index=False)
        print("✓ Saved lo_alignment_by_discipline.csv")
        
        # 3. By topic
        topic_stats = []
        for _, group in self.alignment_data.groupby(['discipline', 'topic']):
            topic_stats.append({
                'Discipline': group['discipline'].iloc[0],
                'Topic': group['topic'].iloc[0],
                'Total_Questions': len(group),
                'Aligned_Questions': group['is_aligned'].sum(),
                'Not_Aligned_Questions': (~group['is_aligned']).sum(),
                'Alignment_Rate_%': round(group['is_aligned'].sum() / len(group) * 100, 2)
            })
        
        topic_df = pd.DataFrame(topic_stats).sort_values(['Discipline', 'Topic'])
        topic_df.to_csv(output_dir / 'lo_alignment_by_topic.csv', index=False)
        print("✓ Saved lo_alignment_by_topic.csv")
        
        # 4. All non-aligned questions with details
        non_aligned = self.alignment_data[~self.alignment_data['is_aligned']].copy()
        non_aligned_export = non_aligned[['discipline', 'topic', 'Question ID', 'Question Text', 
                                         'Run Number', 'Explanation']].copy()
        non_aligned_export.columns = ['Discipline', 'Topic', 'Question_ID', 'Question_Text',
                                      'Run_Number', 'Justification']
        non_aligned_export = non_aligned_export.sort_values(['Discipline', 'Topic', 'Run_Number'])
        non_aligned_export.to_csv(output_dir / 'lo_alignment_non_aligned_questions.csv', index=False)
        print("✓ Saved lo_alignment_non_aligned_questions.csv")
        
        print("\n✓ All LO alignment tables exported")
        
        return {
            'overall': overall_df,
            'by_discipline': discipline_df,
            'by_topic': topic_df,
            'non_aligned': non_aligned_export
        }
    
    def analyze_correlations(self, quality_data_path=None):
        """Analyze correlations between LO alignment and quality metrics."""
        print("\n=== Analyzing Correlations ===")
        
        # Load quality data if available
        if quality_data_path:
            quality_df = pd.read_csv(quality_data_path)
        else:
            # Try to load from the default location
            quality_file = self.base_path / 'table_quality_by_topic.csv'
            if quality_file.exists():
                quality_df = pd.read_csv(quality_file)
            else:
                print("No quality data found for correlation analysis")
                return None
        
        # Aggregate alignment by topic
        alignment_by_topic = self.alignment_data.groupby(['discipline', 'topic']).agg({
            'is_aligned': ['sum', 'count']
        }).reset_index()
        alignment_by_topic.columns = ['Discipline', 'Topic', 'Aligned_Count', 'Total_Count']
        alignment_by_topic['Alignment_Rate_%'] = (alignment_by_topic['Aligned_Count'] / 
                                                  alignment_by_topic['Total_Count'] * 100)
        
        # Merge with quality data (only AI-generated)
        quality_ai = quality_df[quality_df['Source'] == 'AI'].copy()
        
        merged = pd.merge(alignment_by_topic, quality_ai,
                         left_on=['Discipline', 'Topic'],
                         right_on=['Discipline', 'Topic'],
                         how='inner')
        
        if len(merged) > 0:
            print(f"\n✓ Merged {len(merged)} topics with both alignment and quality data")
            
            # Calculate correlations
            corr_acceptance = merged['Alignment_Rate_%'].corr(merged['Acceptance_Rate_%'])
            corr_failures = merged['Alignment_Rate_%'].corr(merged['Average_Failures'])
            
            print(f"\nCorrelations:")
            print(f"  LO Alignment vs Quality Acceptance Rate: {corr_acceptance:.3f}")
            print(f"  LO Alignment vs Average Failures: {corr_failures:.3f}")
            
            # Save correlation data
            correlation_stats = {
                'Metric': ['Quality Acceptance Rate', 'Average Failures'],
                'Correlation_with_LO_Alignment': [round(corr_acceptance, 3), round(corr_failures, 3)]
            }
            correlation_df = pd.DataFrame(correlation_stats)
            correlation_df.to_csv(self.base_path / 'lo_alignment_correlations.csv', index=False)
            print("✓ Saved lo_alignment_correlations.csv")
            
            return merged, correlation_df
        else:
            print("No matching topics found for correlation analysis")
            return None, None
    
    def create_visualizations(self, output_dir=None):
        """Create LO alignment visualizations."""
        if output_dir is None:
            output_dir = self.base_path
        else:
            output_dir = Path(output_dir)
        
        print("\n=== Creating Visualizations ===")
        
        # 1. Overall alignment by discipline
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))
        
        # Subplot 1: Alignment rate by discipline
        discipline_summary = self.alignment_data.groupby('discipline').agg({
            'is_aligned': ['sum', 'count']
        }).reset_index()
        discipline_summary.columns = ['Discipline', 'Aligned', 'Total']
        discipline_summary['Not_Aligned'] = discipline_summary['Total'] - discipline_summary['Aligned']
        discipline_summary['Alignment_Rate'] = (discipline_summary['Aligned'] / 
                                                discipline_summary['Total'] * 100)
        
        x = np.arange(len(discipline_summary))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, discipline_summary['Aligned'], width,
                       label='Aligned', color=COLORS['aligned'], edgecolor='black', linewidth=1)
        bars2 = ax1.bar(x + width/2, discipline_summary['Not_Aligned'], width,
                       label='Not Aligned', color=COLORS['not_aligned'], edgecolor='black', linewidth=1)
        
        ax1.set_ylabel('Number of Questions', fontsize=11)
        ax1.set_title('LO Alignment by Discipline', fontsize=12, pad=15)
        ax1.set_xticks(x)
        ax1.set_xticklabels(discipline_summary['Discipline'], fontsize=10)
        ax1.legend(frameon=True, fontsize=9)
        ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                            f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        # Subplot 2: Alignment rates as percentages
        bars = ax2.bar(discipline_summary['Discipline'], discipline_summary['Alignment_Rate'],
                      color=COLORS['aligned'], edgecolor='black', linewidth=1)
        ax2.set_ylabel('Alignment Rate (%)', fontsize=11)
        ax2.set_title('LO Alignment Rate by Discipline', fontsize=12, pad=15)
        ax2.set_ylim([0, 105])
        ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        for bar, val in zip(bars, discipline_summary['Alignment_Rate']):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Subplot 3: Topics with lowest alignment
        topic_alignment = self.alignment_data.groupby(['discipline', 'topic']).agg({
            'is_aligned': ['sum', 'count']
        }).reset_index()
        topic_alignment.columns = ['Discipline', 'Topic', 'Aligned', 'Total']
        topic_alignment['Alignment_Rate'] = (topic_alignment['Aligned'] / 
                                             topic_alignment['Total'] * 100)
        
        worst_topics = topic_alignment.nsmallest(10, 'Alignment_Rate')
        
        y = np.arange(len(worst_topics))
        bars = ax3.barh(y, worst_topics['Alignment_Rate'], 
                       color=COLORS['not_aligned'], edgecolor='black', linewidth=0.5)
        ax3.set_yticks(y)
        ax3.set_yticklabels([f"{row['Topic'][:25]}..." if len(row['Topic']) > 25 
                            else row['Topic'] 
                            for _, row in worst_topics.iterrows()], fontsize=8)
        ax3.set_xlabel('Alignment Rate (%)', fontsize=11)
        ax3.set_title('10 Topics with Lowest LO Alignment', fontsize=12, pad=15)
        ax3.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        for bar, val in zip(bars, worst_topics['Alignment_Rate']):
            width = bar.get_width()
            ax3.text(width + 1.5, bar.get_y() + bar.get_height()/2.,
                    f'{val:.0f}%', ha='left', va='center', fontsize=8)
        
        # Subplot 4: Summary table
        summary_data = [
            ['Total Questions', len(self.alignment_data)],
            ['Aligned', self.alignment_data['is_aligned'].sum()],
            ['Not Aligned', (~self.alignment_data['is_aligned']).sum()],
            ['Alignment Rate', f"{self.alignment_data['is_aligned'].sum()/len(self.alignment_data)*100:.1f}%"]
        ]
        
        ax4.axis('tight')
        ax4.axis('off')
        table = ax4.table(cellText=summary_data,
                         colLabels=['Metric', 'Value'],
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        for i in range(2):
            table[(0, i)].set_facecolor('#666666')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, 5):
            for j in range(2):
                table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 1 else 'white')
                table[(i, j)].set_edgecolor('black')
                table[(i, j)].set_linewidth(0.5)
        
        plt.tight_layout(pad=2.0)
        filename = 'lo_alignment_summary.png'
        plt.savefig(output_dir / filename, dpi=600, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.eps'), bbox_inches='tight', facecolor='white', format='eps')
        plt.close()
        print(f"✓ Saved {filename} (PNG, PDF, EPS)")


def main():
    """Main execution."""
    print("=" * 70)
    print("LO Alignment Analysis")
    print("=" * 70)
    
    base_path = Path(__file__).parent
    analyzer = LOAlignmentAnalyzer(base_path)
    
    # Load data
    print("\n[1/4] Loading LO alignment data...")
    data = analyzer.load_all_files()
    
    if data is None or len(data) == 0:
        print("Error: No LO alignment data loaded. Exiting.")
        return
    
    # Export tables
    print("\n[2/4] Exporting data tables...")
    tables = analyzer.export_data_tables()
    
    # Analyze correlations
    print("\n[3/4] Analyzing correlations with quality metrics...")
    merged_data, correlations = analyzer.analyze_correlations()
    
    # Create visualizations
    print("\n[4/4] Creating visualizations...")
    analyzer.create_visualizations()
    
    # Print tables
    print("\n" + "=" * 70)
    print("DATA TABLES")
    print("=" * 70)
    
    print("\n1. OVERALL SUMMARY")
    print(tables['overall'].to_string(index=False))
    
    print("\n\n2. ALIGNMENT BY DISCIPLINE")
    print(tables['by_discipline'].to_string(index=False))
    
    print("\n\n3. TOPICS WITH LOWEST ALIGNMENT (Bottom 10)")
    topic_summary = tables['by_topic'].nsmallest(10, 'Alignment_Rate_%')
    print(topic_summary.to_string(index=False))
    
    print("\n\n4. NON-ALIGNED QUESTIONS (First 10)")
    print(tables['non_aligned'].head(10).to_string(index=False))
    print(f"\n... and {len(tables['non_aligned']) - 10} more non-aligned questions")
    
    if correlations is not None:
        print("\n\n5. CORRELATIONS WITH QUALITY METRICS")
        print(correlations.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("✓ Analysis complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  Visualizations:")
    print("    - lo_alignment_summary.png/pdf/eps")
    print("  Data Tables:")
    print("    - lo_alignment_overall_summary.csv")
    print("    - lo_alignment_by_discipline.csv")
    print("    - lo_alignment_by_topic.csv")
    print("    - lo_alignment_non_aligned_questions.csv")
    print("    - lo_alignment_correlations.csv (if quality data available)")
    print(f"\nAll files saved to: {base_path}")


if __name__ == "__main__":
    main()
