"""
Analyze question quality based on SAQUET evaluation criteria.

SAQUET evaluates questions on 18 criteria:
- 0-1 failures = Acceptable question
- 2+ failures = Unacceptable question

Compares AI-generated questions (SAQUET_results) with SME-written questions (SME_SAQUET_results).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Modern, aesthetically pleasing color palette (WCAG AA-compliant)
COLORS = {
    'acceptable': '#06A77D',      # Teal - good quality
    'unacceptable': '#E63946',    # Coral red - poor quality
    'ai': '#118AB2',              # Ocean blue - AI-generated
    'sme': '#6A4C93',             # Royal purple - SME-written
    'discipline1': '#118AB2',     # Ocean blue - Data Science
    'discipline2': '#6A4C93',     # Royal purple - Computer Science
    'discipline3': '#D62828',     # Deep red - Discrete Math
}

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

# SAQUET criteria columns (18 total - excluding fill_in_the_blank)
CRITERIA_COLUMNS = [
    'implausible_distractors', 'none_of_the_above', 'all_of_the_above',
    'true_or_false', 'absolute_terms',
    'longest_answer_correct', 'negative_worded_stem',
    'word_repeats_in_stem_and_correct_answer', 'avoid_logical_cues',
    'lost_sequence', 'more_than_one_correct', 'complex_k_type',
    'ambiguous_unclear_information', 'gratuitous_information_in_stem',
    'avoid_convergence_cues', 'grammatical_cues_in_stem', 'vague_terms',
    'unfocused_stem'
]

# SAQUET criteria grouped by category
CRITERIA_CATEGORIES = {
    'Clarity': [
        'complex_k_type',
        'unfocused_stem',
        'negative_worded_stem',
        'ambiguous_unclear_information',
        'gratuitous_information_in_stem',
        'lost_sequence',
        'vague_terms'
    ],
    'Accuracy': [
        'more_than_one_correct',
        'absolute_terms'
    ],
    'Test-wiseness Prevention': [
        'implausible_distractors',
        'true_or_false',
        'avoid_convergence_cues',
        'longest_answer_correct',
        'none_of_the_above',
        'word_repeats_in_stem_and_correct_answer',
        'avoid_logical_cues',
        'all_of_the_above',
        'grammatical_cues_in_stem'
    ]
}

# Category colors
CATEGORY_COLORS = {
    'Clarity': '#F77F00',          # Vibrant orange
    'Accuracy': '#E63946',         # Coral red
    'Test-wiseness Prevention': '#118AB2'  # Ocean blue
}

class QuestionQualityAnalyzer:
    """Analyze question quality from SAQUET evaluation results."""
    
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.ai_data = None
        self.sme_data = None
        self.category_analysis = None
        
    def load_all_files(self):
        """Load all SAQUET results files (AI and SME)."""
        print("\n[Loading AI-Generated Questions]")
        self.ai_data = self._load_files_by_pattern('SAQUET_results_*.csv', 'AI')
        
        print("\n[Loading SME-Written Questions]")
        self.sme_data = self._load_files_by_pattern('SME_SAQUET_results_*.csv', 'SME')
        
        return self.ai_data, self.sme_data
    
    def _load_files_by_pattern(self, pattern, source_type):
        """Load files matching a pattern."""
        all_questions = []
        
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
                
            files = list(folder_path.glob(pattern))
            print(f"  {folder}: {len(files)} files")
            
            for file in files:
                try:
                    topic = file.stem.replace('SAQUET_results_', '').replace('SME_SAQUET_results_', '')
                    df = pd.read_csv(file)
                    
                    # Count failures for each question
                    df['failure_count'] = 0
                    for col in CRITERIA_COLUMNS:
                        if col in df.columns:
                            df['failure_count'] += (df[col] == 'Fail').astype(int)
                    
                    # Categorize quality
                    df['quality'] = df['failure_count'].apply(
                        lambda x: 'Acceptable' if x <= 1 else 'Unacceptable'
                    )
                    
                    # Add metadata
                    df['discipline'] = discipline
                    df['topic'] = topic
                    df['source'] = source_type
                    df['source_folder'] = folder
                    
                    all_questions.append(df)
                    
                except Exception as e:
                    print(f"    Error loading {file.name}: {e}")
        
        if all_questions:
            combined = pd.concat(all_questions, ignore_index=True)
            print(f"\n  Total {source_type} questions loaded: {len(combined)}")
            print(f"  Acceptable: {(combined['quality'] == 'Acceptable').sum()} ({(combined['quality'] == 'Acceptable').sum()/len(combined)*100:.1f}%)")
            print(f"  Unacceptable: {(combined['quality'] == 'Unacceptable').sum()} ({(combined['quality'] == 'Unacceptable').sum()/len(combined)*100:.1f}%)")
            return combined
        else:
            return pd.DataFrame()
    
    def export_data_tables(self, output_dir=None):
        """Export all calculated data to CSV files for review."""
        if output_dir is None:
            output_dir = self.base_path
        else:
            output_dir = Path(output_dir)
        
        print("\n=== Exporting Data Tables ===")
        
        all_data = pd.concat([self.ai_data, self.sme_data], ignore_index=True)
        
        # 1. Overall summary statistics
        overall_stats = []
        for source in ['AI', 'SME']:
            data = self.ai_data if source == 'AI' else self.sme_data
            total = len(data)
            acceptable = (data['quality'] == 'Acceptable').sum()
            unacceptable = (data['quality'] == 'Unacceptable').sum()
            acceptance_rate = (acceptable / total * 100) if total > 0 else 0
            avg_failures = data['failure_count'].mean()
            median_failures = data['failure_count'].median()
            
            overall_stats.append({
                'Source': f'{source}-Generated' if source == 'AI' else f'{source}-Written',
                'Total_Questions': total,
                'Acceptable_Questions': acceptable,
                'Unacceptable_Questions': unacceptable,
                'Acceptance_Rate_%': round(acceptance_rate, 2),
                'Average_Failures': round(avg_failures, 2),
                'Median_Failures': median_failures
            })
        
        overall_df = pd.DataFrame(overall_stats)
        overall_df.to_csv(output_dir / 'table_overall_summary.csv', index=False)
        print("✓ Saved table_overall_summary.csv")
        
        # 2. Quality by discipline
        discipline_stats = []
        for discipline in all_data['discipline'].unique():
            for source in ['AI', 'SME']:
                data = all_data[(all_data['discipline'] == discipline) & (all_data['source'] == source)]
                if len(data) > 0:
                    total = len(data)
                    acceptable = (data['quality'] == 'Acceptable').sum()
                    unacceptable = (data['quality'] == 'Unacceptable').sum()
                    acceptance_rate = (acceptable / total * 100)
                    avg_failures = data['failure_count'].mean()
                    
                    discipline_stats.append({
                        'Discipline': discipline,
                        'Source': source,
                        'Total_Questions': total,
                        'Acceptable_Questions': acceptable,
                        'Unacceptable_Questions': unacceptable,
                        'Acceptance_Rate_%': round(acceptance_rate, 2),
                        'Average_Failures': round(avg_failures, 2)
                    })
        
        discipline_df = pd.DataFrame(discipline_stats)
        discipline_df.to_csv(output_dir / 'table_quality_by_discipline.csv', index=False)
        print("✓ Saved table_quality_by_discipline.csv")
        
        # 3. Quality by topic (detailed)
        topic_stats = []
        for _, row in all_data.groupby(['source_folder', 'topic', 'source', 'discipline']).size().reset_index().iterrows():
            folder = row['source_folder']
            topic = row['topic']
            source = row['source']
            discipline = row['discipline']
            
            data = all_data[(all_data['source_folder'] == folder) & 
                           (all_data['topic'] == topic) & 
                           (all_data['source'] == source)]
            
            total = len(data)
            acceptable = (data['quality'] == 'Acceptable').sum()
            unacceptable = (data['quality'] == 'Unacceptable').sum()
            acceptance_rate = (acceptable / total * 100) if total > 0 else 0
            avg_failures = data['failure_count'].mean() if total > 0 else 0
            
            topic_stats.append({
                'Discipline': discipline,
                'Folder': folder,
                'Topic': topic,
                'Source': source,
                'Total_Questions': total,
                'Acceptable_Questions': acceptable,
                'Unacceptable_Questions': unacceptable,
                'Acceptance_Rate_%': round(acceptance_rate, 2),
                'Average_Failures': round(avg_failures, 2)
            })
        
        topic_df = pd.DataFrame(topic_stats).sort_values(['Discipline', 'Topic', 'Source'])
        topic_df.to_csv(output_dir / 'table_quality_by_topic.csv', index=False)
        print("✓ Saved table_quality_by_topic.csv")
        
        # 4. Failure counts by criterion
        criterion_stats = []
        for col in CRITERIA_COLUMNS:
            if col in all_data.columns:
                ai_failures = (self.ai_data[col] == 'Fail').sum() if col in self.ai_data.columns else 0
                ai_total = len(self.ai_data)
                ai_failure_rate = (ai_failures / ai_total * 100) if ai_total > 0 else 0
                
                sme_failures = (self.sme_data[col] == 'Fail').sum() if col in self.sme_data.columns else 0
                sme_total = len(self.sme_data)
                sme_failure_rate = (sme_failures / sme_total * 100) if sme_total > 0 else 0
                
                criterion_stats.append({
                    'Criterion': col.replace('_', ' ').title(),
                    'AI_Failures': ai_failures,
                    'AI_Total': ai_total,
                    'AI_Failure_Rate_%': round(ai_failure_rate, 2),
                    'SME_Failures': sme_failures,
                    'SME_Total': sme_total,
                    'SME_Failure_Rate_%': round(sme_failure_rate, 2),
                    'Difference_%': round(ai_failure_rate - sme_failure_rate, 2)
                })
        
        criterion_df = pd.DataFrame(criterion_stats).sort_values('AI_Failures', ascending=False)
        criterion_df.to_csv(output_dir / 'table_failures_by_criterion.csv', index=False)
        print("✓ Saved table_failures_by_criterion.csv")
        
        # 5. Failure distribution
        failure_dist = []
        for source in ['AI', 'SME']:
            data = self.ai_data if source == 'AI' else self.sme_data
            counts = data['failure_count'].value_counts().sort_index()
            for fail_count, num_questions in counts.items():
                failure_dist.append({
                    'Source': source,
                    'Number_of_Failures': fail_count,
                    'Number_of_Questions': num_questions,
                    'Percentage_%': round((num_questions / len(data) * 100), 2)
                })
        
        failure_dist_df = pd.DataFrame(failure_dist)
        failure_dist_df.to_csv(output_dir / 'table_failure_distribution.csv', index=False)
        print("✓ Saved table_failure_distribution.csv")
        
        # 6. Summary comparison table
        comparison = []
        for discipline in all_data['discipline'].unique():
            ai_data_disc = self.ai_data[self.ai_data['discipline'] == discipline]
            sme_data_disc = self.sme_data[self.sme_data['discipline'] == discipline]
            
            ai_accept = (ai_data_disc['quality'] == 'Acceptable').sum() / len(ai_data_disc) * 100 if len(ai_data_disc) > 0 else 0
            sme_accept = (sme_data_disc['quality'] == 'Acceptable').sum() / len(sme_data_disc) * 100 if len(sme_data_disc) > 0 else 0
            
            comparison.append({
                'Discipline': discipline,
                'AI_Total': len(ai_data_disc),
                'AI_Acceptance_%': round(ai_accept, 2),
                'SME_Total': len(sme_data_disc),
                'SME_Acceptance_%': round(sme_accept, 2),
                'Difference_%': round(ai_accept - sme_accept, 2)
            })
        
        comparison_df = pd.DataFrame(comparison)
        comparison_df.to_csv(output_dir / 'table_discipline_comparison.csv', index=False)
        print("✓ Saved table_discipline_comparison.csv")
        
        print("\n✓ All data tables exported")
        
        return {
            'overall': overall_df,
            'by_discipline': discipline_df,
            'by_topic': topic_df,
            'by_criterion': criterion_df,
            'failure_distribution': failure_dist_df,
            'comparison': comparison_df
        }
    
    def analyze_by_category(self, output_dir=None):
        """Analyze failures grouped by SAQUET categories."""
        if output_dir is None:
            output_dir = self.base_path
        else:
            output_dir = Path(output_dir)
        
        print("\n=== Analyzing by SAQUET Category ===")
        
        # Calculate failures by category for AI and SME
        category_results = []
        
        for category, criteria in CRITERIA_CATEGORIES.items():
            # AI failures in this category
            ai_failures = 0
            ai_total = len(self.ai_data) * len(criteria)
            
            for criterion in criteria:
                if criterion in self.ai_data.columns:
                    ai_failures += (self.ai_data[criterion] == 'Fail').sum()
            
            ai_failure_rate = (ai_failures / ai_total * 100) if ai_total > 0 else 0
            
            # SME failures in this category
            sme_failures = 0
            sme_total = len(self.sme_data) * len(criteria)
            
            for criterion in criteria:
                if criterion in self.sme_data.columns:
                    sme_failures += (self.sme_data[criterion] == 'Fail').sum()
            
            sme_failure_rate = (sme_failures / sme_total * 100) if sme_total > 0 else 0
            
            category_results.append({
                'Category': category,
                'Criteria_Count': len(criteria),
                'AI_Failures': ai_failures,
                'AI_Total_Opportunities': ai_total,
                'AI_Failure_Rate_%': round(ai_failure_rate, 2),
                'SME_Failures': sme_failures,
                'SME_Total_Opportunities': sme_total,
                'SME_Failure_Rate_%': round(sme_failure_rate, 2),
                'Difference_%': round(ai_failure_rate - sme_failure_rate, 2)
            })
        
        category_df = pd.DataFrame(category_results)
        category_df.to_csv(output_dir / 'table_failures_by_category.csv', index=False)
        print("✓ Saved table_failures_by_category.csv")
        
        # Also analyze by category AND discipline
        category_discipline_results = []
        
        for discipline in sorted(self.ai_data['discipline'].unique()):
            for category, criteria in CRITERIA_CATEGORIES.items():
                ai_disc = self.ai_data[self.ai_data['discipline'] == discipline]
                sme_disc = self.sme_data[self.sme_data['discipline'] == discipline]
                
                # AI failures
                ai_failures = 0
                ai_total = len(ai_disc) * len(criteria)
                for criterion in criteria:
                    if criterion in ai_disc.columns:
                        ai_failures += (ai_disc[criterion] == 'Fail').sum()
                
                ai_failure_rate = (ai_failures / ai_total * 100) if ai_total > 0 else 0
                
                # SME failures
                sme_failures = 0
                sme_total = len(sme_disc) * len(criteria)
                for criterion in criteria:
                    if criterion in sme_disc.columns:
                        sme_failures += (sme_disc[criterion] == 'Fail').sum()
                
                sme_failure_rate = (sme_failures / sme_total * 100) if sme_total > 0 else 0
                
                category_discipline_results.append({
                    'Discipline': discipline,
                    'Category': category,
                    'AI_Failure_Rate_%': round(ai_failure_rate, 2),
                    'SME_Failure_Rate_%': round(sme_failure_rate, 2),
                    'Difference_%': round(ai_failure_rate - sme_failure_rate, 2)
                })
        
        category_discipline_df = pd.DataFrame(category_discipline_results)
        category_discipline_df.to_csv(output_dir / 'table_failures_by_category_and_discipline.csv', index=False)
        print("✓ Saved table_failures_by_category_and_discipline.csv")
        
        self.category_analysis = category_df
        
        return category_df, category_discipline_df
    
    def create_visualizations(self, output_dir=None):
        """Create all visualizations."""
        if output_dir is None:
            output_dir = self.base_path
        else:
            output_dir = Path(output_dir)
        
        print("\n=== Creating Visualizations ===")
        
        # 1. Overall quality comparison
        self._create_overall_quality_comparison(output_dir)
        
        # 2. Quality by discipline
        self._create_discipline_comparison(output_dir)
        
        # 3. Failure analysis
        self._create_failure_analysis(output_dir)
        
        # 4. Interactive dashboard
        self._create_interactive_dashboard(output_dir)
        
        print("✓ All visualizations complete")
    
    def _create_overall_quality_comparison(self, output_dir):
        """Compare AI vs SME overall quality."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))
        
        # 1. Acceptance rate comparison
        ai_acceptable = (self.ai_data['quality'] == 'Acceptable').sum() / len(self.ai_data) * 100
        sme_acceptable = (self.sme_data['quality'] == 'Acceptable').sum() / len(self.sme_data) * 100
        
        bars = ax1.bar(['AI-Generated', 'SME-Written'], [ai_acceptable, sme_acceptable],
                      color=[COLORS['ai'], COLORS['sme']], edgecolor='black', linewidth=1)
        ax1.set_ylabel('Acceptance Rate (%)', fontsize=11)
        ax1.set_title('Question Acceptance Rate: AI vs SME\n(0-1 failures = Acceptable)',
                     fontsize=12, pad=15)
        ax1.set_ylim([0, 105])
        ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        for bar, val in zip(bars, [ai_acceptable, sme_acceptable]):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        filename = 'quality_comparison_overall.png'
        plt.savefig(output_dir / filename, dpi=600, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.eps'), bbox_inches='tight', facecolor='white', format='eps')
        plt.close()
        print(f"✓ Saved {filename} (PNG, PDF, EPS)")
        
        # 2. Distribution of failure counts - AI
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
        ai_failure_dist = self.ai_data['failure_count'].value_counts().sort_index()
        ax2.bar(ai_failure_dist.index, ai_failure_dist.values, 
               color=COLORS['ai'], edgecolor='black', linewidth=1, alpha=0.8)
        ax2.axvline(x=1.5, color=COLORS['unacceptable'], linestyle='--', linewidth=1.5, 
                   label='Threshold (1 failure)')
        ax2.set_xlabel('Number of Failures', fontsize=11)
        ax2.set_ylabel('Number of Questions', fontsize=11)
        ax2.set_title('AI-Generated Questions: Failure Distribution',
                     fontsize=12, pad=15)
        ax2.legend(frameon=True, fontsize=9)
        ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        plt.tight_layout()
        filename = 'quality_distribution_ai.png'
        plt.savefig(output_dir / filename, dpi=600, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.eps'), bbox_inches='tight', facecolor='white', format='eps')
        plt.close()
        print(f"✓ Saved {filename} (PNG, PDF, EPS)")
        
        # 3. Distribution of failure counts - SME
        fig3, ax3 = plt.subplots(1, 1, figsize=(8, 6))
        sme_failure_dist = self.sme_data['failure_count'].value_counts().sort_index()
        ax3.bar(sme_failure_dist.index, sme_failure_dist.values,
               color=COLORS['sme'], edgecolor='black', linewidth=1, alpha=0.8)
        ax3.axvline(x=1.5, color=COLORS['unacceptable'], linestyle='--', linewidth=1.5,
                   label='Threshold (1 failure)')
        ax3.set_xlabel('Number of Failures', fontsize=11)
        ax3.set_ylabel('Number of Questions', fontsize=11)
        ax3.set_title('SME-Written Questions: Failure Distribution',
                     fontsize=12, pad=15)
        ax3.legend(frameon=True, fontsize=9)
        ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        plt.tight_layout()
        filename = 'quality_distribution_sme.png'
        plt.savefig(output_dir / filename, dpi=600, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.eps'), bbox_inches='tight', facecolor='white', format='eps')
        plt.close()
        print(f"✓ Saved {filename} (PNG, PDF, EPS)")
        
        # 4. Comparison table
        fig4, ax4 = plt.subplots(1, 1, figsize=(8, 4))
        comparison_data = [
            ['AI-Generated', len(self.ai_data), 
             (self.ai_data['quality'] == 'Acceptable').sum(),
             (self.ai_data['quality'] == 'Unacceptable').sum(),
             f"{ai_acceptable:.1f}%",
             f"{self.ai_data['failure_count'].mean():.2f}"],
            ['SME-Written', len(self.sme_data),
             (self.sme_data['quality'] == 'Acceptable').sum(),
             (self.sme_data['quality'] == 'Unacceptable').sum(),
             f"{sme_acceptable:.1f}%",
             f"{self.sme_data['failure_count'].mean():.2f}"]
        ]
        
        ax4.axis('tight')
        ax4.axis('off')
        table = ax4.table(cellText=comparison_data,
                         colLabels=['Source', 'Total', 'Acceptable', 'Unacceptable', 
                                   'Accept. Rate', 'Avg Fail.'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.2, 0.15, 0.15, 0.15, 0.17, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.2)
        
        # Style header
        for i in range(6):
            table[(0, i)].set_facecolor('#666666')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style data rows
        for i in range(1, 3):
            for j in range(6):
                table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 1 else 'white')
                table[(i, j)].set_edgecolor('black')
                table[(i, j)].set_linewidth(0.5)
        
        plt.tight_layout()
        filename = 'quality_comparison_table.png'
        plt.savefig(output_dir / filename, dpi=600, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.eps'), bbox_inches='tight', facecolor='white', format='eps')
        plt.close()
        print(f"✓ Saved {filename} (PNG, PDF, EPS)")
    
    def _create_discipline_comparison(self, output_dir):
        """Compare quality across disciplines."""
        # Create each visualization separately
        
        # 1. Acceptance rate by discipline and source
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        
        # Combine AI and SME data for discipline analysis
        all_data = pd.concat([self.ai_data, self.sme_data], ignore_index=True)
        
        # 1. Acceptance rate by discipline and source
        discipline_source = all_data.groupby(['discipline', 'source']).agg({
            'quality': lambda x: (x == 'Acceptable').sum() / len(x) * 100
        }).unstack(fill_value=0)
        
        discipline_source.columns = discipline_source.columns.droplevel()
        x = np.arange(len(discipline_source))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, discipline_source['AI'], width,
                       label='AI-Generated', color=COLORS['ai'], edgecolor='black', linewidth=1)
        bars2 = ax1.bar(x + width/2, discipline_source['SME'], width,
                       label='SME-Written', color=COLORS['sme'], edgecolor='black', linewidth=1)
        
        ax1.set_ylabel('Acceptance Rate (%)', fontsize=11)
        ax1.set_title('Question Acceptance Rate by Discipline',
                     fontsize=12, pad=15)
        ax1.set_xticks(x)
        ax1.set_xticklabels(discipline_source.index, fontsize=10)
        ax1.legend(frameon=True, fontsize=9)
        ax1.set_ylim([0, 100])
        ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                            f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        filename = 'quality_by_discipline_comparison.png'
        plt.savefig(output_dir / filename, dpi=600, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.eps'), bbox_inches='tight', facecolor='white', format='eps')
        plt.close()
        print(f"✓ Saved {filename} (PNG, PDF, EPS)")
        
        # 2. AI quality by discipline
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
        ai_by_disc = self.ai_data.groupby('discipline')['quality'].apply(
            lambda x: (x == 'Acceptable').sum() / len(x) * 100
        ).sort_values(ascending=True)
        
        colors_ai = [COLORS['discipline1'], COLORS['discipline2'], COLORS['discipline3']]
        bars = ax2.barh(ai_by_disc.index, ai_by_disc.values,
                       color=colors_ai[:len(ai_by_disc)], edgecolor='black', linewidth=1)
        ax2.set_xlabel('Acceptance Rate (%)', fontsize=11)
        ax2.set_title('AI-Generated Questions by Discipline',
                     fontsize=12, pad=15)
        ax2.set_xlim([0, 100])
        ax2.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        for bar, val in zip(bars, ai_by_disc.values):
            width = bar.get_width()
            ax2.text(width + 1.5, bar.get_y() + bar.get_height()/2.,
                    f'{val:.1f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        filename = 'quality_by_discipline_ai.png'
        plt.savefig(output_dir / filename, dpi=600, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.eps'), bbox_inches='tight', facecolor='white', format='eps')
        plt.close()
        print(f"✓ Saved {filename} (PNG, PDF, EPS)")
        
        # 3. SME quality by discipline
        fig3, ax3 = plt.subplots(1, 1, figsize=(8, 6))
        sme_by_disc = self.sme_data.groupby('discipline')['quality'].apply(
            lambda x: (x == 'Acceptable').sum() / len(x) * 100
        ).sort_values(ascending=True)
        
        bars = ax3.barh(sme_by_disc.index, sme_by_disc.values,
                       color=colors_ai[:len(sme_by_disc)], edgecolor='black', linewidth=1)
        ax3.set_xlabel('Acceptance Rate (%)', fontsize=11)
        ax3.set_title('SME-Written Questions by Discipline',
                     fontsize=12, pad=15)
        ax3.set_xlim([0, 100])
        ax3.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        for bar, val in zip(bars, sme_by_disc.values):
            width = bar.get_width()
            ax3.text(width + 1.5, bar.get_y() + bar.get_height()/2.,
                    f'{val:.1f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        filename = 'quality_by_discipline_sme.png'
        plt.savefig(output_dir / filename, dpi=600, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.eps'), bbox_inches='tight', facecolor='white', format='eps')
        plt.close()
        print(f"✓ Saved {filename} (PNG, PDF, EPS)")
        
        # 4. Average failures by discipline
        fig4, ax4 = plt.subplots(1, 1, figsize=(10, 6))
        avg_failures = all_data.groupby(['discipline', 'source'])['failure_count'].mean().unstack()
        
        x = np.arange(len(avg_failures))
        bar_width = 0.35
        bars1 = ax4.bar(x - bar_width/2, avg_failures['AI'], bar_width,
                       label='AI-Generated', color=COLORS['ai'], edgecolor='black', linewidth=1)
        bars2 = ax4.bar(x + bar_width/2, avg_failures['SME'], bar_width,
                       label='SME-Written', color=COLORS['sme'], edgecolor='black', linewidth=1)
        
        ax4.set_ylabel('Average Number of Failures', fontsize=11)
        ax4.set_title('Average Failures per Question by Discipline',
                     fontsize=12, pad=15)
        ax4.set_xticks(x)
        ax4.set_xticklabels(avg_failures.index, fontsize=10)
        ax4.legend(frameon=True, fontsize=9)
        ax4.axhline(y=1.5, color=COLORS['unacceptable'], linestyle='--', linewidth=1.5, alpha=0.6, label='Threshold')
        ax4.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
        plt.tight_layout()
        filename = 'quality_avg_failures_by_discipline.png'
        plt.savefig(output_dir / filename, dpi=600, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.eps'), bbox_inches='tight', facecolor='white', format='eps')
        plt.close()
        print(f"✓ Saved {filename} (PNG, PDF, EPS)")
    
    def _create_failure_analysis(self, output_dir):
        """Analyze which criteria fail most often."""
        # Count failures for each criterion
        ai_failures = {}
        sme_failures = {}
        
        for col in CRITERIA_COLUMNS:
            if col in self.ai_data.columns:
                ai_failures[col] = (self.ai_data[col] == 'Fail').sum()
            if col in self.sme_data.columns:
                sme_failures[col] = (self.sme_data[col] == 'Fail').sum()
        
        # Sort by AI failures
        ai_sorted = sorted(ai_failures.items(), key=lambda x: x[1], reverse=True)
        criteria_names = [k.replace('_', ' ').title() for k, _ in ai_sorted]
        ai_vals = [v for _, v in ai_sorted]
        sme_vals = [sme_failures.get(k, 0) for k, _ in ai_sorted]
        
        # AI failures
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 8))
        y = np.arange(len(criteria_names))
        bars = ax1.barh(y, ai_vals, color=COLORS['ai'], edgecolor='black', linewidth=0.5)
        ax1.set_yticks(y)
        ax1.set_yticklabels(criteria_names, fontsize=8)
        ax1.set_xlabel('Number of Failures', fontsize=11)
        ax1.set_title('AI-Generated Questions: Failures by Criterion',
                     fontsize=12, pad=15)
        ax1.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        for bar, val in zip(bars, ai_vals):
            if val > 0:
                ax1.text(val + max(ai_vals)*0.015, bar.get_y() + bar.get_height()/2.,
                        f'{val}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        filename = 'failure_by_criterion_ai.png'
        plt.savefig(output_dir / filename, dpi=600, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.eps'), bbox_inches='tight', facecolor='white', format='eps')
        plt.close()
        print(f"✓ Saved {filename} (PNG, PDF, EPS)")
        
        # SME failures
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))
        bars = ax2.barh(y, sme_vals, color=COLORS['sme'], edgecolor='black', linewidth=0.5)
        ax2.set_yticks(y)
        ax2.set_yticklabels(criteria_names, fontsize=8)
        ax2.set_xlabel('Number of Failures', fontsize=11)
        ax2.set_title('SME-Written Questions: Failures by Criterion',
                     fontsize=12, pad=15)
        ax2.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        for bar, val in zip(bars, sme_vals):
            if val > 0:
                ax2.text(val + max(sme_vals)*0.015, bar.get_y() + bar.get_height()/2.,
                        f'{val}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        filename = 'failure_by_criterion_sme.png'
        plt.savefig(output_dir / filename, dpi=600, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.eps'), bbox_inches='tight', facecolor='white', format='eps')
        plt.close()
        print(f"✓ Saved {filename} (PNG, PDF, EPS)")
    
    def create_category_visualizations(self, output_dir=None):
        """Create visualizations for category-based analysis."""
        if output_dir is None:
            output_dir = self.base_path
        else:
            output_dir = Path(output_dir)
        
        if self.category_analysis is None:
            print("Warning: No category analysis data. Run analyze_by_category() first.")
            return
        
        print("\n=== Creating Category Visualizations ===")
        
        # 1. Overall failure rates by category
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        
        categories = self.category_analysis['Category']
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, self.category_analysis['AI_Failure_Rate_%'], width,
                       label='AI-Generated', color=COLORS['ai'], edgecolor='black', linewidth=1)
        bars2 = ax1.bar(x + width/2, self.category_analysis['SME_Failure_Rate_%'], width,
                       label='SME-Written', color=COLORS['sme'], edgecolor='black', linewidth=1)
        
        ax1.set_ylabel('Failure Rate (%)', fontsize=11)
        ax1.set_title('SAQUET Failure Rate by Category\n(AI-Generated vs SME-Written)',
                     fontsize=12, pad=15)
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, fontsize=10)
        ax1.legend(frameon=True, fontsize=9, loc='upper right')
        ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                            f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        filename = 'category_failure_rates_comparison.png'
        plt.savefig(output_dir / filename, dpi=600, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.eps'), bbox_inches='tight', facecolor='white', format='eps')
        plt.close()
        print(f"✓ Saved {filename} (PNG, PDF, EPS)")
        
        # 2. Difference in failure rates by category
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        
        differences = self.category_analysis['Difference_%']
        colors_diff = [COLORS['unacceptable'] if d > 0 else COLORS['acceptable'] for d in differences]
        
        bars = ax2.barh(categories, differences, color=colors_diff, edgecolor='black', linewidth=1)
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
        ax2.set_xlabel('Difference in Failure Rate (AI - SME) %', fontsize=11)
        ax2.set_title('Difference in Failure Rates by Category\n(Positive = AI worse, Negative = AI better)',
                     fontsize=12, pad=15)
        ax2.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        for bar, val in zip(bars, differences):
            width = bar.get_width()
            label_x = width + (0.3 if width > 0 else -0.3)
            ha = 'left' if width > 0 else 'right'
            ax2.text(label_x, bar.get_y() + bar.get_height()/2.,
                    f'{val:.1f}%', ha=ha, va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        filename = 'category_failure_difference.png'
        plt.savefig(output_dir / filename, dpi=600, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.eps'), bbox_inches='tight', facecolor='white', format='eps')
        plt.close()
        print(f"✓ Saved {filename} (PNG, PDF, EPS)")
        
        # 3. AI failure rates by category (detailed)
        fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
        
        cat_colors = [CATEGORY_COLORS[cat] for cat in categories]
        bars = ax3.bar(categories, self.category_analysis['AI_Failure_Rate_%'],
                      color=cat_colors, edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('Failure Rate (%)', fontsize=11)
        ax3.set_title('AI-Generated Questions: Failure Rate by Category',
                     fontsize=12, pad=15)
        ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        for bar, val, count in zip(bars, self.category_analysis['AI_Failure_Rate_%'],
                                   self.category_analysis['Criteria_Count']):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{val:.1f}%\n({count} criteria)', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        filename = 'category_ai_failure_rates.png'
        plt.savefig(output_dir / filename, dpi=600, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.eps'), bbox_inches='tight', facecolor='white', format='eps')
        plt.close()
        print(f"✓ Saved {filename} (PNG, PDF, EPS)")
        
        # 4. SME failure rates by category (detailed)
        fig4, ax4 = plt.subplots(1, 1, figsize=(10, 6))
        
        bars = ax4.bar(categories, self.category_analysis['SME_Failure_Rate_%'],
                      color=cat_colors, edgecolor='black', linewidth=1.5)
        ax4.set_ylabel('Failure Rate (%)', fontsize=11)
        ax4.set_title('SME-Written Questions: Failure Rate by Category',
                     fontsize=12, pad=15)
        ax4.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
        for bar, val, count in zip(bars, self.category_analysis['SME_Failure_Rate_%'],
                                   self.category_analysis['Criteria_Count']):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{val:.1f}%\n({count} criteria)', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        filename = 'category_sme_failure_rates.png'
        plt.savefig(output_dir / filename, dpi=600, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.eps'), bbox_inches='tight', facecolor='white', format='eps')
        plt.close()
        print(f"✓ Saved {filename} (PNG, PDF, EPS)")
        
        print("✓ All category visualizations complete")
    
    def create_category_by_discipline_visualizations(self, output_dir=None):
        """Create visualizations for category analysis by discipline."""
        if output_dir is None:
            output_dir = self.base_path
        else:
            output_dir = Path(output_dir)
        
        print("\n=== Creating Category by Discipline Visualizations ===")
        
        # Load the category by discipline data
        cat_disc_file = output_dir / 'table_failures_by_category_and_discipline.csv'
        if not cat_disc_file.exists():
            print("Warning: Category by discipline data not found. Run analyze_by_category() first.")
            return
        
        cat_disc_df = pd.read_csv(cat_disc_file)
        
        # 1. Grouped bar chart: All disciplines and categories
        fig1, ax1 = plt.subplots(1, 1, figsize=(12, 7))
        
        disciplines = sorted(cat_disc_df['Discipline'].unique())
        categories = sorted(cat_disc_df['Category'].unique())
        
        x = np.arange(len(categories))
        width = 0.12
        
        colors_disc = [COLORS['discipline1'], COLORS['discipline2'], COLORS['discipline3']]
        
        # Plot AI bars for each discipline
        for i, discipline in enumerate(disciplines):
            disc_data = cat_disc_df[cat_disc_df['Discipline'] == discipline]
            disc_data = disc_data.sort_values('Category')
            offset = (i - 1) * width
            ax1.bar(x + offset - width, disc_data['AI_Failure_Rate_%'], width,
                   label=f'{discipline} (AI)', color=colors_disc[i], edgecolor='black', 
                   linewidth=0.5, alpha=0.8)
        
        # Plot SME bars for each discipline
        for i, discipline in enumerate(disciplines):
            disc_data = cat_disc_df[cat_disc_df['Discipline'] == discipline]
            disc_data = disc_data.sort_values('Category')
            offset = (i - 1) * width
            ax1.bar(x + offset + width, disc_data['SME_Failure_Rate_%'], width,
                   label=f'{discipline} (SME)', color=colors_disc[i], edgecolor='black',
                   linewidth=0.5, alpha=0.4, hatch='//')
        
        ax1.set_ylabel('Failure Rate (%)', fontsize=11)
        ax1.set_title('SAQUET Failure Rates by Category and Discipline\n(Solid = AI, Hatched = SME)',
                     fontsize=12, pad=15)
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, fontsize=10)
        ax1.legend(frameon=True, fontsize=8, loc='upper right', ncol=2)
        ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        plt.tight_layout()
        filename = 'category_by_discipline_all.png'
        plt.savefig(output_dir / filename, dpi=600, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.eps'), bbox_inches='tight', facecolor='white', format='eps')
        plt.close()
        print(f"✓ Saved {filename} (PNG, PDF, EPS)")
        
        # 2. Individual charts for each discipline
        for discipline in disciplines:
            disc_data = cat_disc_df[cat_disc_df['Discipline'] == discipline]
            disc_data = disc_data.sort_values('Category')
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            x = np.arange(len(disc_data))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, disc_data['AI_Failure_Rate_%'], width,
                          label='AI-Generated', color=COLORS['ai'], edgecolor='black', linewidth=1)
            bars2 = ax.bar(x + width/2, disc_data['SME_Failure_Rate_%'], width,
                          label='SME-Written', color=COLORS['sme'], edgecolor='black', linewidth=1)
            
            ax.set_ylabel('Failure Rate (%)', fontsize=11)
            ax.set_title(f'{discipline}: SAQUET Failure Rate by Category',
                        fontsize=12, pad=15)
            ax.set_xticks(x)
            ax.set_xticklabels(disc_data['Category'], fontsize=10)
            ax.legend(frameon=True, fontsize=9)
            ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                               f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            filename = f'category_by_discipline_{discipline.lower().replace(" ", "_")}.png'
            plt.savefig(output_dir / filename, dpi=600, bbox_inches='tight', facecolor='white')
            plt.savefig(output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
            plt.savefig(output_dir / filename.replace('.png', '.eps'), bbox_inches='tight', facecolor='white', format='eps')
            plt.close()
            print(f"✓ Saved {filename} (PNG, PDF, EPS)")
        
        # 3. Heatmap of differences (AI - SME) by discipline and category
        fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
        
        # Pivot the data for heatmap
        heatmap_data = cat_disc_df.pivot(index='Discipline', columns='Category', values='Difference_%')
        
        # Create custom colormap: negative (green) to zero (white) to positive (red)
        from matplotlib.colors import TwoSlopeNorm
        vmin = heatmap_data.min().min()
        vmax = heatmap_data.max().max()
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        
        im = ax3.imshow(heatmap_data.values, cmap='RdYlGn_r', aspect='auto', norm=norm)
        
        # Set ticks and labels
        ax3.set_xticks(np.arange(len(heatmap_data.columns)))
        ax3.set_yticks(np.arange(len(heatmap_data.index)))
        ax3.set_xticklabels(heatmap_data.columns, fontsize=10)
        ax3.set_yticklabels(heatmap_data.index, fontsize=10)
        
        # Rotate x labels
        plt.setp(ax3.get_xticklabels(), rotation=0, ha="center")
        
        # Add text annotations
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                val = heatmap_data.values[i, j]
                text = ax3.text(j, i, f'{val:.1f}%',
                              ha="center", va="center", color="black", fontsize=9,
                              fontweight='bold')
        
        ax3.set_title('Difference in Failure Rates by Discipline and Category\n(AI - SME: Red = AI worse, Green = AI better)',
                     fontsize=12, pad=15)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Difference in Failure Rate (%)', rotation=270, labelpad=20, fontsize=10)
        
        plt.tight_layout()
        filename = 'category_by_discipline_heatmap.png'
        plt.savefig(output_dir / filename, dpi=600, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / filename.replace('.png', '.eps'), bbox_inches='tight', facecolor='white', format='eps')
        plt.close()
        print(f"✓ Saved {filename} (PNG, PDF, EPS)")
        
        print("✓ All category by discipline visualizations complete")
    
    def _create_interactive_dashboard(self, output_dir):
        """Create interactive Plotly dashboard."""
        all_data = pd.concat([self.ai_data, self.sme_data], ignore_index=True)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Acceptance Rate: AI vs SME',
                'Acceptance Rate by Discipline',
                'Failure Distribution Comparison',
                'Top Failing Criteria'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'bar'}],
                [{'type': 'histogram'}, {'type': 'bar'}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        # 1. Overall comparison
        ai_accept = (self.ai_data['quality'] == 'Acceptable').sum() / len(self.ai_data) * 100
        sme_accept = (self.sme_data['quality'] == 'Acceptable').sum() / len(self.sme_data) * 100
        
        fig.add_trace(
            go.Bar(x=['AI-Generated', 'SME-Written'], y=[ai_accept, sme_accept],
                  marker_color=[COLORS['ai'], COLORS['sme']],
                  marker_line_color='black', marker_line_width=1.5,
                  text=[f'{ai_accept:.1f}%', f'{sme_accept:.1f}%'],
                  textposition='outside',
                  showlegend=False,
                  hovertemplate='<b>%{x}</b><br>Acceptance: %{y:.1f}%<extra></extra>'),
            row=1, col=1
        )
        
        # 2. By discipline
        discipline_stats = all_data.groupby(['discipline', 'source']).agg({
            'quality': lambda x: (x == 'Acceptable').sum() / len(x) * 100
        }).reset_index()
        
        for source, color in [('AI', COLORS['ai']), ('SME', COLORS['sme'])]:
            source_data = discipline_stats[discipline_stats['source'] == source]
            fig.add_trace(
                go.Bar(name=f'{source}-Generated' if source == 'AI' else f'{source}-Written',
                      x=source_data['discipline'], y=source_data['quality'],
                      marker_color=color, marker_line_color='black', marker_line_width=1,
                      hovertemplate='<b>%{x}</b><br>' + source + ': %{y:.1f}%<extra></extra>'),
                row=1, col=2
            )
        
        # 3. Failure distribution
        fig.add_trace(
            go.Histogram(x=self.ai_data['failure_count'], name='AI-Generated',
                        marker_color=COLORS['ai'], marker_line_color='black',
                        marker_line_width=1, opacity=0.7,
                        hovertemplate='Failures: %{x}<br>Count: %{y}<extra></extra>'),
            row=2, col=1
        )
        fig.add_trace(
            go.Histogram(x=self.sme_data['failure_count'], name='SME-Written',
                        marker_color=COLORS['sme'], marker_line_color='black',
                        marker_line_width=1, opacity=0.7,
                        hovertemplate='Failures: %{x}<br>Count: %{y}<extra></extra>'),
            row=2, col=1
        )
        
        # 4. Top failing criteria
        ai_failures = {col: (self.ai_data[col] == 'Fail').sum() 
                      for col in CRITERIA_COLUMNS if col in self.ai_data.columns}
        top_failures = sorted(ai_failures.items(), key=lambda x: x[1], reverse=True)[:10]
        
        fig.add_trace(
            go.Bar(x=[x[1] for x in top_failures],
                  y=[x[0].replace('_', ' ').title() for x in top_failures],
                  orientation='h',
                  marker_color=COLORS['unacceptable'],
                  marker_line_color='black', marker_line_width=1,
                  showlegend=False,
                  hovertemplate='<b>%{y}</b><br>Failures: %{x}<extra></extra>'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_xaxes(title_text="Source", row=1, col=1, showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text="Acceptance Rate (%)", row=1, col=1, showgrid=True, gridcolor='lightgray')
        
        fig.update_xaxes(title_text="Discipline", row=1, col=2, showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text="Acceptance Rate (%)", row=1, col=2, showgrid=True, gridcolor='lightgray')
        
        fig.update_xaxes(title_text="Number of Failures", row=2, col=1, showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text="Count", row=2, col=1, showgrid=True, gridcolor='lightgray')
        
        fig.update_xaxes(title_text="Number of Failures", row=2, col=2, showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text="Criterion", row=2, col=2, showgrid=True, gridcolor='lightgray')
        
        fig.update_layout(
            title_text="MCQ Quality Analysis Dashboard - AI vs SME Questions",
            title_font_size=20,
            height=1000,
            showlegend=True,
            barmode='group',
            font=dict(family='Arial', size=11, color='black'),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        filename = 'quality_dashboard.html'
        fig.write_html(output_dir / filename)
        print(f"✓ Saved {filename}")


def main():
    """Main execution."""
    print("=" * 70)
    print("MCQ Quality Analysis - SAQUET Evaluation")
    print("=" * 70)
    
    base_path = Path(__file__).parent
    analyzer = QuestionQualityAnalyzer(base_path)
    
    # Load data
    print("\n[1/5] Loading SAQUET evaluation results...")
    ai_data, sme_data = analyzer.load_all_files()
    
    if ai_data is None or len(ai_data) == 0:
        print("Error: No AI data loaded. Exiting.")
        return
    
    if sme_data is None or len(sme_data) == 0:
        print("Warning: No SME data loaded.")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\nAI-Generated Questions:")
    print(f"  Total: {len(ai_data)}")
    print(f"  Acceptable (0-1 failures): {(ai_data['quality'] == 'Acceptable').sum()} ({(ai_data['quality'] == 'Acceptable').sum()/len(ai_data)*100:.1f}%)")
    print(f"  Unacceptable (2+ failures): {(ai_data['quality'] == 'Unacceptable').sum()} ({(ai_data['quality'] == 'Unacceptable').sum()/len(ai_data)*100:.1f}%)")
    print(f"  Average failures per question: {ai_data['failure_count'].mean():.2f}")
    
    if len(sme_data) > 0:
        print(f"\nSME-Written Questions:")
        print(f"  Total: {len(sme_data)}")
        print(f"  Acceptable (0-1 failures): {(sme_data['quality'] == 'Acceptable').sum()} ({(sme_data['quality'] == 'Acceptable').sum()/len(sme_data)*100:.1f}%)")
        print(f"  Unacceptable (2+ failures): {(sme_data['quality'] == 'Unacceptable').sum()} ({(sme_data['quality'] == 'Unacceptable').sum()/len(sme_data)*100:.1f}%)")
        print(f"  Average failures per question: {sme_data['failure_count'].mean():.2f}")
    
    # Export data tables
    print(f"\n[2/6] Exporting data tables...")
    tables = analyzer.export_data_tables()
    
    # Analyze by category
    print(f"\n[3/6] Analyzing by SAQUET category...")
    category_df, category_discipline_df = analyzer.analyze_by_category()
    
    # Create standard visualizations
    print(f"\n[4/6] Creating standard visualizations...")
    analyzer.create_visualizations()
    
    # Create category visualizations
    print(f"\n[5/6] Creating category visualizations...")
    analyzer.create_category_visualizations()
    
    # Create category by discipline visualizations
    print(f"\n[6/6] Creating category by discipline visualizations...")
    analyzer.create_category_by_discipline_visualizations()
    
    # Print tables to console for easy viewing
    print("\n" + "=" * 70)
    print("DATA TABLES")
    print("=" * 70)
    
    print("\n1. OVERALL SUMMARY")
    print(tables['overall'].to_string(index=False))
    
    print("\n\n2. QUALITY BY DISCIPLINE")
    print(tables['by_discipline'].to_string(index=False))
    
    print("\n\n3. TOP 10 MOST PROBLEMATIC CRITERIA")
    print(tables['by_criterion'].head(10).to_string(index=False))
    
    print("\n\n4. FAILURE DISTRIBUTION")
    print(tables['failure_distribution'].to_string(index=False))
    
    print("\n\n5. DISCIPLINE COMPARISON (AI vs SME)")
    print(tables['comparison'].to_string(index=False))
    
    print("\n\n6. FAILURE RATES BY CATEGORY")
    print(category_df.to_string(index=False))
    
    print("\n\n7. FAILURE RATES BY CATEGORY AND DISCIPLINE")
    print(category_discipline_df.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("✓ Analysis complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  Standard Visualizations:")
    print("    - quality_comparison_overall.png/pdf/eps")
    print("    - quality_distribution_ai.png/pdf/eps")
    print("    - quality_distribution_sme.png/pdf/eps")
    print("    - quality_comparison_table.png/pdf/eps")
    print("    - quality_by_discipline_comparison.png/pdf/eps")
    print("    - quality_by_discipline_ai.png/pdf/eps")
    print("    - quality_by_discipline_sme.png/pdf/eps")
    print("    - quality_avg_failures_by_discipline.png/pdf/eps")
    print("    - failure_by_criterion_ai.png/pdf/eps")
    print("    - failure_by_criterion_sme.png/pdf/eps")
    print("    - quality_dashboard.html")
    print("  Category Visualizations:")
    print("    - category_failure_rates_comparison.png/pdf/eps")
    print("    - category_failure_difference.png/pdf/eps")
    print("    - category_ai_failure_rates.png/pdf/eps")
    print("    - category_sme_failure_rates.png/pdf/eps")
    print("  Category by Discipline Visualizations:")
    print("    - category_by_discipline_all.png/pdf/eps")
    print("    - category_by_discipline_computer_science.png/pdf/eps")
    print("    - category_by_discipline_data_science.png/pdf/eps")
    print("    - category_by_discipline_discrete_mathematics.png/pdf/eps")
    print("    - category_by_discipline_heatmap.png/pdf/eps")
    print("  Data Tables:")
    print("    - table_overall_summary.csv")
    print("    - table_quality_by_discipline.csv")
    print("    - table_quality_by_topic.csv")
    print("    - table_failures_by_criterion.csv")
    print("    - table_failure_distribution.csv")
    print("    - table_discipline_comparison.csv")
    print("    - table_failures_by_category.csv")
    print("    - table_failures_by_category_and_discipline.csv")
    print(f"\nAll files saved to: {base_path}")


if __name__ == "__main__":
    main()
