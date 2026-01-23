"""
Analyze and visualize prompt consistency with separate analyses for:
1. Within-run similarity (from main other_analysis files)
2. Cross-run consistency (from Consistency runs directories)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Color palette (WCAG AA compliant)
COLORS = {
    'duplicates': '#E63946',      # Coral red (problems)
    'similar': '#F77F00',          # Vibrant orange
    'unique': '#06A77D',           # Teal (good)
    'consistency': '#118AB2',      # Ocean blue
    'discipline1': '#F77F00',      # Orange
    'discipline2': '#118AB2',      # Blue
    'discipline3': '#6A4C93',      # Purple
}

class ConsistencyAnalyzer:
    """Analyze within-run similarity and cross-run consistency separately."""
    
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.within_run_data = None
        self.cross_run_data = None
        
    def load_all_data(self):
        """Load both within-run and cross-run data."""
        print("\n=== Loading Within-Run Similarity Data ===")
        self.within_run_data = self._load_within_run_files()
        
        print("\n=== Loading Cross-Run Consistency Data ===")
        self.cross_run_data = self._load_cross_run_files()
        
        return self.within_run_data, self.cross_run_data
    
    def _load_within_run_files(self):
        """Load original other_analysis files for within-run similarity."""
        all_data = []
        
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
                
            # Exclude Consistency runs subdirectory
            files = [f for f in folder_path.glob('other_analysis_*.csv') 
                    if 'Consistency runs' not in str(f)]
            print(f"  {folder}: {len(files)} files")
            
            for file in files:
                try:
                    data = self._parse_within_run_file(file, discipline, folder)
                    if data:
                        all_data.append(data)
                except Exception as e:
                    print(f"    Error parsing {file.name}: {e}")
                    
        df = pd.DataFrame(all_data)
        print(f"\n  Total topics loaded: {len(df)}")
        return df
    
    def _load_cross_run_files(self):
        """Load Consistency runs files for cross-run consistency."""
        all_data = []
        
        disciplines = {
            'DataSciencePython': 'Data Science',
            'ProgrammingInJava': 'Computer Science',
            'ProgrammingInPython3': 'Computer Science',
            'DiscreteMath': 'Discrete Mathematics'
        }
        
        for folder, discipline in disciplines.items():
            consistency_path = self.base_path / folder / 'Consistency runs'
            if not consistency_path.exists():
                continue
                
            files = list(consistency_path.glob('other_analysis_*.csv'))
            print(f"  {folder}: {len(files)} files")
            
            for file in files:
                try:
                    data = self._parse_cross_run_file(file, discipline, folder)
                    if data:
                        all_data.append(data)
                except Exception as e:
                    print(f"    Error parsing {file.name}: {e}")
                    
        df = pd.DataFrame(all_data)
        print(f"\n  Total topics loaded: {len(df)}")
        return df
    
    def _parse_within_run_file(self, filepath, discipline, source_folder):
        """Parse a single other_analysis CSV file for within-run similarity."""
        filename = filepath.stem
        topic = filename.replace('other_analysis_', '').rsplit('_', 1)[0]
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Parse Within-Run Analysis
        within_run_data = []
        
        for i, line in enumerate(lines):
            if 'Within-Run Analysis' in line:
                for j in range(i + 2, min(i + 6, len(lines))):
                    parts = lines[j].strip().split(',')
                    if len(parts) >= 4:
                        try:
                            within_run_data.append({
                                'run': int(parts[0]),
                                'question_count': int(parts[1]),
                                'internal_duplicates': int(parts[2]),
                                'internal_similar': int(parts[3])
                            })
                        except ValueError:
                            break
                break
        
        if not within_run_data:
            return None
        
        # Calculate within-run metrics
        total_questions = sum(r['question_count'] for r in within_run_data)
        total_within_duplicates = sum(r['internal_duplicates'] for r in within_run_data)
        total_within_similar = sum(r['internal_similar'] for r in within_run_data)
        runs_with_duplicates = sum(1 for r in within_run_data if r['internal_duplicates'] > 0)
        runs_with_similar = sum(1 for r in within_run_data if r['internal_similar'] > 0)
        
        return {
            'discipline': discipline,
            'source_folder': source_folder,
            'topic': topic,
            'total_questions': total_questions,
            'avg_questions_per_run': total_questions / 4,
            'within_run_duplicates': total_within_duplicates,
            'within_run_similar': total_within_similar,
            'runs_with_duplicates': runs_with_duplicates,
            'runs_with_similar': runs_with_similar,
            'has_within_run_duplicates': total_within_duplicates > 0,
            'has_within_run_similar': total_within_similar > 0,
        }
    
    def _parse_cross_run_file(self, filepath, discipline, source_folder):
        """Parse a Consistency runs file for cross-run consistency."""
        filename = filepath.stem
        topic = filename.replace('other_analysis_', '').rsplit('_', 1)[0]
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Parse Cross-Run section
        cross_run_data = []
        for i, line in enumerate(lines):
            if 'Cross-Run Analysis' in line:
                for j in range(i + 2, min(i + 12, len(lines))):
                    if 'Summary' in lines[j]:
                        break
                    parts = lines[j].strip().split(',')
                    if len(parts) >= 5:
                        try:
                            cross_run_data.append({
                                'run_a': int(parts[0]),
                                'run_b': int(parts[1]),
                                'duplicates': int(parts[2]),
                                'similar': int(parts[3]),
                                'total_comparisons': int(parts[4])
                            })
                        except (ValueError, IndexError):
                            break
                break
        
        if not cross_run_data:
            return None
        
        # Calculate cross-run consistency metrics
        total_cross_duplicates = sum(r['duplicates'] for r in cross_run_data)
        total_cross_similar = sum(r['similar'] for r in cross_run_data)
        total_cross_comparisons = sum(r['total_comparisons'] for r in cross_run_data)
        
        cross_consistency_pct = ((total_cross_duplicates + total_cross_similar) / total_cross_comparisons * 100) if total_cross_comparisons > 0 else 0
        
        return {
            'discipline': discipline,
            'source_folder': source_folder,
            'topic': topic,
            'cross_run_duplicates': total_cross_duplicates,
            'cross_run_similar': total_cross_similar,
            'cross_run_comparisons': total_cross_comparisons,
            'cross_run_consistency_pct': cross_consistency_pct,
            'cross_run_duplicate_pct': (total_cross_duplicates / total_cross_comparisons * 100) if total_cross_comparisons > 0 else 0,
            'cross_run_similar_pct': (total_cross_similar / total_cross_comparisons * 100) if total_cross_comparisons > 0 else 0,
        }
    
    def export_data_tables(self, output_dir=None):
        """Export all calculated data to CSV files."""
        if output_dir is None:
            output_dir = self.base_path
        else:
            output_dir = Path(output_dir)
        
        print("\n=== Exporting Data Tables ===")
        
        # === WITHIN-RUN SIMILARITY TABLES ===
        print("\n--- Within-Run Similarity Tables ---")
        
        # 1. Overall summary for within-run
        overall_stats = {
            'Total_Topics': len(self.within_run_data),
            'Topics_With_Duplicates': self.within_run_data['has_within_run_duplicates'].sum(),
            'Topics_With_Similar': self.within_run_data['has_within_run_similar'].sum(),
            'Percent_With_Duplicates': round(self.within_run_data['has_within_run_duplicates'].sum() / len(self.within_run_data) * 100, 2),
            'Percent_With_Similar': round(self.within_run_data['has_within_run_similar'].sum() / len(self.within_run_data) * 100, 2),
        }
        overall_df = pd.DataFrame([overall_stats])
        overall_df.to_csv(output_dir / 'within_run_table_overall_summary.csv', index=False)
        print("✓ Saved within_run_table_overall_summary.csv")
        
        # 2. Statistics by discipline for within-run
        discipline_stats = []
        for discipline in sorted(self.within_run_data['discipline'].unique()):
            disc_data = self.within_run_data[self.within_run_data['discipline'] == discipline]
            discipline_stats.append({
                'Discipline': discipline,
                'Total_Topics': len(disc_data),
                'Topics_With_Duplicates': disc_data['has_within_run_duplicates'].sum(),
                'Topics_With_Similar': disc_data['has_within_run_similar'].sum(),
                'Percent_With_Duplicates': round(disc_data['has_within_run_duplicates'].sum() / len(disc_data) * 100, 2),
                'Percent_With_Similar': round(disc_data['has_within_run_similar'].sum() / len(disc_data) * 100, 2),
            })
        
        discipline_df = pd.DataFrame(discipline_stats)
        discipline_df.to_csv(output_dir / 'within_run_table_by_discipline.csv', index=False)
        print("✓ Saved within_run_table_by_discipline.csv")
        
        # 3. Detailed by topic for within-run
        topic_details = self.within_run_data[['discipline', 'topic', 'has_within_run_duplicates', 
                                    'has_within_run_similar', 'within_run_duplicates',
                                    'within_run_similar']].copy()
        topic_details.columns = ['Discipline', 'Topic', 'Has_Duplicates', 'Has_Similar', 
                                'Duplicate_Count', 'Similar_Count']
        topic_details = topic_details.sort_values(['Discipline', 'Topic'])
        topic_details.to_csv(output_dir / 'within_run_table_by_topic.csv', index=False)
        print("✓ Saved within_run_table_by_topic.csv")
        
        # 4. Topics with within-run problems
        problem_topics = self.within_run_data[
            (self.within_run_data['has_within_run_duplicates']) | (self.within_run_data['has_within_run_similar'])
        ].copy()
        problem_topics = problem_topics[['discipline', 'topic', 'within_run_duplicates',
                                        'within_run_similar', 'runs_with_duplicates',
                                        'runs_with_similar']]
        problem_topics.columns = ['Discipline', 'Topic', 'Duplicate_Count', 'Similar_Count',
                                  'Runs_With_Duplicates', 'Runs_With_Similar']
        problem_topics = problem_topics.sort_values('Duplicate_Count', ascending=False)
        problem_topics.to_csv(output_dir / 'within_run_table_problem_topics.csv', index=False)
        print("✓ Saved within_run_table_problem_topics.csv")
        
        # === CROSS-RUN CONSISTENCY TABLES ===
        print("\n--- Cross-Run Consistency Tables ---")
        
        # 5. Cross-run consistency overall summary
        cross_overall = {
            'Total_Consistency_Topics': len(self.cross_run_data),
            'Avg_Consistency_%': round(self.cross_run_data['cross_run_consistency_pct'].mean(), 2),
            'Median_Consistency_%': round(self.cross_run_data['cross_run_consistency_pct'].median(), 2),
            'Min_Consistency_%': round(self.cross_run_data['cross_run_consistency_pct'].min(), 2),
            'Max_Consistency_%': round(self.cross_run_data['cross_run_consistency_pct'].max(), 2),
        }
        cross_overall_df = pd.DataFrame([cross_overall])
        cross_overall_df.to_csv(output_dir / 'cross_run_table_overall_summary.csv', index=False)
        print("✓ Saved cross_run_table_overall_summary.csv")
        
        # 6. Cross-run consistency by discipline
        cross_discipline_stats = []
        for discipline in sorted(self.cross_run_data['discipline'].unique()):
            disc_data = self.cross_run_data[self.cross_run_data['discipline'] == discipline]
            cross_discipline_stats.append({
                'Discipline': discipline,
                'Total_Topics': len(disc_data),
                'Avg_Consistency_%': round(disc_data['cross_run_consistency_pct'].mean(), 2),
                'Median_Consistency_%': round(disc_data['cross_run_consistency_pct'].median(), 2),
                'Min_Consistency_%': round(disc_data['cross_run_consistency_pct'].min(), 2),
                'Max_Consistency_%': round(disc_data['cross_run_consistency_pct'].max(), 2),
            })
        
        cross_discipline_df = pd.DataFrame(cross_discipline_stats)
        cross_discipline_df.to_csv(output_dir / 'cross_run_table_by_discipline.csv', index=False)
        print("✓ Saved cross_run_table_by_discipline.csv")
        
        # 7. Cross-run consistency rankings
        consistency_rankings = self.cross_run_data[['discipline', 'topic', 'cross_run_consistency_pct',
                                         'cross_run_duplicate_pct', 'cross_run_similar_pct',
                                         'cross_run_comparisons']].copy()
        consistency_rankings.columns = ['Discipline', 'Topic', 'Consistency_%', 'Duplicate_%',
                                       'Similar_%', 'Total_Comparisons']
        consistency_rankings = consistency_rankings.sort_values('Consistency_%', ascending=False)
        consistency_rankings.to_csv(output_dir / 'cross_run_table_rankings.csv', index=False)
        print("✓ Saved cross_run_table_rankings.csv")
        
        print("\n✓ All data tables exported")
        
        return {
            'within_run_overall': overall_df,
            'within_run_by_discipline': discipline_df,
            'within_run_by_topic': topic_details,
            'within_run_problems': problem_topics,
            'cross_run_overall': cross_overall_df,
            'cross_run_by_discipline': cross_discipline_df,
            'cross_run_rankings': consistency_rankings
        }    
    def create_visualizations(self, output_dir=None):
        """Create all visualizations for within-run and cross-run analyses."""
        if output_dir is None:
            output_dir = self.base_path
        else:
            output_dir = Path(output_dir)
        
        print("\n=== Creating Visualizations ===")
        
        # Configure matplotlib
        plt.rcParams['font.family'] = ['Arial', 'Helvetica', 'sans-serif']
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 9
        
        print("\n--- Within-Run Similarity Visualizations ---")
        self._create_within_run_visualizations(output_dir)
        
        print("\n--- Cross-Run Consistency Visualizations ---")
        self._create_cross_run_visualizations(output_dir)
        
        print("\n✓ All visualizations complete")
    
    def _create_within_run_visualizations(self, output_dir):
        """Create visualizations for within-run similarity analysis."""
        
        # 1. Overall summary of within-run similarity
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        
        categories = ['Topics with\nDuplicates', 'Topics with\nSimilar Questions', 'Topics with\nUnique Questions']
        duplicates = self.within_run_data['has_within_run_duplicates'].sum()
        similar = self.within_run_data['has_within_run_similar'].sum()
        unique = len(self.within_run_data) - duplicates - similar + \
                 len(self.within_run_data[self.within_run_data['has_within_run_duplicates'] & 
                                          self.within_run_data['has_within_run_similar']])
        
        values = [duplicates, similar, unique]
        colors = [COLORS['duplicates'], COLORS['similar'], COLORS['unique']]
        
        bars = ax1.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Number of Topics', fontsize=11)
        ax1.set_title('Within-Run Similarity: Overall Summary\n(Analysis within each of 4 runs per topic)',
                     fontsize=12, pad=15)
        ax1.set_ylim([0, max(values) * 1.15])
        ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            pct = val / len(self.within_run_data) * 100
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                    f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        filename = 'within_run_similarity_summary.png'
        plt.savefig(output_dir / filename, dpi=600, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.eps'), bbox_inches='tight', facecolor='white', format='eps')
        plt.close()
        print(f"✓ Saved {filename} (PNG, PDF, EPS)")
        
        # 2. Within-run similarity by discipline
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        
        discipline_stats = []
        for discipline in sorted(self.within_run_data['discipline'].unique()):
            disc_data = self.within_run_data[self.within_run_data['discipline'] == discipline]
            discipline_stats.append({
                'Discipline': discipline,
                'Duplicates_%': disc_data['has_within_run_duplicates'].sum() / len(disc_data) * 100,
                'Similar_%': disc_data['has_within_run_similar'].sum() / len(disc_data) * 100
            })
        
        disc_df = pd.DataFrame(discipline_stats)
        x = np.arange(len(disc_df))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, disc_df['Duplicates_%'], width,
                       label='Has Duplicates', color=COLORS['duplicates'], edgecolor='black', linewidth=1)
        bars2 = ax2.bar(x + width/2, disc_df['Similar_%'], width,
                       label='Has Similar', color=COLORS['similar'], edgecolor='black', linewidth=1)
        
        ax2.set_ylabel('Percentage of Topics (%)', fontsize=11)
        ax2.set_title('Within-Run Similarity by Discipline', fontsize=12, pad=15)
        ax2.set_xticks(x)
        ax2.set_xticklabels(disc_df['Discipline'], fontsize=10)
        ax2.legend(frameon=True, fontsize=9)
        ax2.set_ylim([0, max(max(disc_df['Duplicates_%']), max(disc_df['Similar_%'])) * 1.2])
        ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        filename = 'within_run_similarity_by_discipline.png'
        plt.savefig(output_dir / filename, dpi=600, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.eps'), bbox_inches='tight', facecolor='white', format='eps')
        plt.close()
        print(f"✓ Saved {filename} (PNG, PDF, EPS)")
        
        # 3. Topics with highest duplicate rates
        fig3, ax3 = plt.subplots(1, 1, figsize=(8, 8))
        
        top_duplicates = self.within_run_data.nlargest(10, 'within_run_duplicates')
        
        y = np.arange(len(top_duplicates))
        bars = ax3.barh(y, top_duplicates['within_run_duplicates'],
                       color=COLORS['duplicates'], edgecolor='black', linewidth=0.5)
        ax3.set_yticks(y)
        ax3.set_yticklabels([f"{row['topic'][:30]}..." if len(row['topic']) > 30 
                            else row['topic'] 
                            for _, row in top_duplicates.iterrows()], fontsize=8)
        ax3.set_xlabel('Number of Duplicate Questions', fontsize=11)
        ax3.set_title('Top 10 Topics with Most Duplicate Questions\n(Within-Run Analysis)',
                     fontsize=12, pad=15)
        ax3.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        for bar, val in zip(bars, top_duplicates['within_run_duplicates']):
            width = bar.get_width()
            ax3.text(width + 0.3, bar.get_y() + bar.get_height()/2.,
                    f'{int(val)}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        filename = 'within_run_top_duplicates.png'
        plt.savefig(output_dir / filename, dpi=600, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.eps'), bbox_inches='tight', facecolor='white', format='eps')
        plt.close()
        print(f"✓ Saved {filename} (PNG, PDF, EPS)")
        
        # 4. Topics with highest similar question rates
        fig4, ax4 = plt.subplots(1, 1, figsize=(8, 8))
        
        top_similar = self.within_run_data.nlargest(10, 'within_run_similar')
        
        y = np.arange(len(top_similar))
        bars = ax4.barh(y, top_similar['within_run_similar'],
                       color=COLORS['similar'], edgecolor='black', linewidth=0.5)
        ax4.set_yticks(y)
        ax4.set_yticklabels([f"{row['topic'][:30]}..." if len(row['topic']) > 30 
                            else row['topic'] 
                            for _, row in top_similar.iterrows()], fontsize=8)
        ax4.set_xlabel('Number of Similar Questions', fontsize=11)
        ax4.set_title('Top 10 Topics with Most Similar Questions\n(Within-Run Analysis)',
                     fontsize=12, pad=15)
        ax4.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
        for bar, val in zip(bars, top_similar['within_run_similar']):
            width = bar.get_width()
            ax4.text(width + 0.3, bar.get_y() + bar.get_height()/2.,
                    f'{int(val)}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        filename = 'within_run_top_similar.png'
        plt.savefig(output_dir / filename, dpi=600, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.eps'), bbox_inches='tight', facecolor='white', format='eps')
        plt.close()
        print(f"✓ Saved {filename} (PNG, PDF, EPS)")
    
    def _create_cross_run_visualizations(self, output_dir):
        """Create visualizations for cross-run consistency analysis."""
        
        # 1. Overall cross-run consistency distribution
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        
        ax1.hist(self.cross_run_data['cross_run_consistency_pct'], bins=15,
                color=COLORS['consistency'], edgecolor='black', linewidth=1, alpha=0.8)
        ax1.axvline(self.cross_run_data['cross_run_consistency_pct'].mean(),
                   color=COLORS['duplicates'], linestyle='--', linewidth=2,
                   label=f'Mean: {self.cross_run_data["cross_run_consistency_pct"].mean():.1f}%')
        ax1.axvline(self.cross_run_data['cross_run_consistency_pct'].median(),
                   color=COLORS['similar'], linestyle='--', linewidth=2,
                   label=f'Median: {self.cross_run_data["cross_run_consistency_pct"].median():.1f}%')
        
        ax1.set_xlabel('Cross-Run Consistency (%)', fontsize=11)
        ax1.set_ylabel('Number of Topics', fontsize=11)
        ax1.set_title('Distribution of Cross-Run Consistency Rates\n(Comparison across 4 runs, 10 questions each)',
                     fontsize=12, pad=15)
        ax1.legend(frameon=True, fontsize=9)
        ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        plt.tight_layout()
        filename = 'cross_run_consistency_distribution.png'
        plt.savefig(output_dir / filename, dpi=600, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.eps'), bbox_inches='tight', facecolor='white', format='eps')
        plt.close()
        print(f"✓ Saved {filename} (PNG, PDF, EPS)")
        
        # 2. Cross-run consistency by discipline
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        
        discipline_consistency = []
        for discipline in sorted(self.cross_run_data['discipline'].unique()):
            disc_data = self.cross_run_data[self.cross_run_data['discipline'] == discipline]
            discipline_consistency.append({
                'Discipline': discipline,
                'Mean': disc_data['cross_run_consistency_pct'].mean(),
                'Median': disc_data['cross_run_consistency_pct'].median(),
                'Min': disc_data['cross_run_consistency_pct'].min(),
                'Max': disc_data['cross_run_consistency_pct'].max()
            })
        
        disc_cons_df = pd.DataFrame(discipline_consistency)
        
        colors_disc = [COLORS['discipline1'], COLORS['discipline2'], COLORS['discipline3']]
        bars = ax2.bar(disc_cons_df['Discipline'], disc_cons_df['Mean'],
                      color=colors_disc[:len(disc_cons_df)], edgecolor='black', linewidth=1)
        
        # Add error bars for min/max range
        yerr_lower = disc_cons_df['Mean'] - disc_cons_df['Min']
        yerr_upper = disc_cons_df['Max'] - disc_cons_df['Mean']
        ax2.errorbar(disc_cons_df['Discipline'], disc_cons_df['Mean'],
                    yerr=[yerr_lower, yerr_upper],
                    fmt='none', ecolor='black', capsize=5, linewidth=1.5, alpha=0.7)
        
        ax2.set_ylabel('Cross-Run Consistency (%)', fontsize=11)
        ax2.set_title('Cross-Run Behavior across multiple runs\n(Mean with min/max range)',
                     fontsize=12, pad=15)
        ax2.set_ylim([0, max(disc_cons_df['Max']) * 1.15])
        ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        for bar, mean_val in zip(bars, disc_cons_df['Mean']):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{mean_val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        filename = 'cross_run_consistency_by_discipline.png'
        plt.savefig(output_dir / filename, dpi=600, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.eps'), bbox_inches='tight', facecolor='white', format='eps')
        plt.close()
        print(f"✓ Saved {filename} (PNG, PDF, EPS)")
        
        # 3. Topics with highest consistency
        fig3, ax3 = plt.subplots(1, 1, figsize=(8, 8))
        
        top_consistent = self.cross_run_data.nlargest(10, 'cross_run_consistency_pct')
        
        y = np.arange(len(top_consistent))
        bars = ax3.barh(y, top_consistent['cross_run_consistency_pct'],
                       color=COLORS['unique'], edgecolor='black', linewidth=0.5)
        ax3.set_yticks(y)
        ax3.set_yticklabels([f"{row['topic'][:30]}..." if len(row['topic']) > 30 
                            else row['topic'] 
                            for _, row in top_consistent.iterrows()], fontsize=8)
        ax3.set_xlabel('Cross-Run Consistency (%)', fontsize=11)
        ax3.set_title('Top 10 Topics with Highest Cross-Run Consistency',
                     fontsize=12, pad=15)
        ax3.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        for bar, val in zip(bars, top_consistent['cross_run_consistency_pct']):
            width = bar.get_width()
            ax3.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                    f'{val:.1f}%', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        filename = 'cross_run_highest_consistency.png'
        plt.savefig(output_dir / filename, dpi=600, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.eps'), bbox_inches='tight', facecolor='white', format='eps')
        plt.close()
        print(f"✓ Saved {filename} (PNG, PDF, EPS)")
        
        # 4. Topics with lowest consistency
        fig4, ax4 = plt.subplots(1, 1, figsize=(8, 8))
        
        low_consistent = self.cross_run_data.nsmallest(10, 'cross_run_consistency_pct')
        
        y = np.arange(len(low_consistent))
        bars = ax4.barh(y, low_consistent['cross_run_consistency_pct'],
                       color=COLORS['duplicates'], edgecolor='black', linewidth=0.5)
        ax4.set_yticks(y)
        ax4.set_yticklabels([f"{row['topic'][:30]}..." if len(row['topic']) > 30 
                            else row['topic'] 
                            for _, row in low_consistent.iterrows()], fontsize=8)
        ax4.set_xlabel('Cross-Run Consistency (%)', fontsize=11)
        ax4.set_title('Top 10 Topics with Lowest Cross-Run Consistency',
                     fontsize=12, pad=15)
        ax4.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
        for bar, val in zip(bars, low_consistent['cross_run_consistency_pct']):
            width = bar.get_width()
            ax4.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                    f'{val:.1f}%', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        filename = 'cross_run_lowest_consistency.png'
        plt.savefig(output_dir / filename, dpi=600, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.eps'), bbox_inches='tight', facecolor='white', format='eps')
        plt.close()
        print(f"✓ Saved {filename} (PNG, PDF, EPS)")

def main():
    """Main execution function."""
    print("=" * 70)
    print("PROMPT CONSISTENCY ANALYZER")
    print("Analyzing Within-Run Similarity & Cross-Run Consistency")
    print("=" * 70)
    
    base_path = Path('/Users/ericaperich/Desktop/MCQ research')
    analyzer = ConsistencyAnalyzer(base_path)
    
    # Load data from both sources
    print(f"\n[1/2] Loading data...")
    within_run, cross_run = analyzer.load_all_data()
    
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    print("\n--- WITHIN-RUN SIMILARITY (All Topics) ---")
    print(f"Total topics analyzed: {len(within_run)}")
    print(f"Topics with duplicates (≥85%): {within_run['has_within_run_duplicates'].sum()} ({within_run['has_within_run_duplicates'].sum()/len(within_run)*100:.1f}%)")
    print(f"Topics with similar (70-85%): {within_run['has_within_run_similar'].sum()} ({within_run['has_within_run_similar'].sum()/len(within_run)*100:.1f}%)")
    
    print("\n--- CROSS-RUN CONSISTENCY (Top 5 Topics per Discipline) ---")
    print(f"Total topics analyzed: {len(cross_run)}")
    print(f"Average consistency: {cross_run['cross_run_consistency_pct'].mean():.1f}%")
    print(f"Median consistency: {cross_run['cross_run_consistency_pct'].median():.1f}%")
    print(f"Range: {cross_run['cross_run_consistency_pct'].min():.1f}% - {cross_run['cross_run_consistency_pct'].max():.1f}%")
    
    print("\nBy Discipline:")
    for discipline in sorted(within_run['discipline'].unique()):
        within_disc = within_run[within_run['discipline'] == discipline]
        cross_disc = cross_run[cross_run['discipline'] == discipline]
        print(f"\n  {discipline}:")
        print(f"    Within-run topics: {len(within_disc)}")
        print(f"      - With duplicates: {within_disc['has_within_run_duplicates'].sum()} ({within_disc['has_within_run_duplicates'].sum()/len(within_disc)*100:.1f}%)")
        print(f"      - With similar: {within_disc['has_within_run_similar'].sum()} ({within_disc['has_within_run_similar'].sum()/len(within_disc)*100:.1f}%)")
        print(f"    Cross-run topics: {len(cross_disc)}")
        print(f"      - Avg consistency: {cross_disc['cross_run_consistency_pct'].mean():.1f}%")
    
    # Export tables
    print(f"\n[2/3] Exporting data tables...")
    tables = analyzer.export_data_tables()
    
    # Create visualizations
    print(f"\n[3/3] Creating visualizations...")
    analyzer.create_visualizations()
    
    print("\n" + "=" * 70)
    print("✓ Analysis complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  Within-Run Similarity:")
    print("    Tables:")
    print("      - within_run_table_overall_summary.csv")
    print("      - within_run_table_by_discipline.csv")
    print("      - within_run_table_by_topic.csv")
    print("      - within_run_table_problem_topics.csv")
    print("    Visualizations:")
    print("      - within_run_similarity_summary.png/pdf/eps")
    print("      - within_run_similarity_by_discipline.png/pdf/eps")
    print("      - within_run_top_duplicates.png/pdf/eps")
    print("      - within_run_top_similar.png/pdf/eps")
    print("  Cross-Run Consistency:")
    print("    Tables:")
    print("      - cross_run_table_overall_summary.csv")
    print("      - cross_run_table_by_discipline.csv")
    print("      - cross_run_table_rankings.csv")
    print("    Visualizations:")
    print("      - cross_run_consistency_distribution.png/pdf/eps")
    print("      - cross_run_consistency_by_discipline.png/pdf/eps")
    print("      - cross_run_highest_consistency.png/pdf/eps")
    print("      - cross_run_lowest_consistency.png/pdf/eps")
    print(f"\nAll files saved to: {base_path}")


if __name__ == "__main__":
    main()
