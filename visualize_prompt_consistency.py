"""
Analyze and visualize prompt consistency and within-run duplicate detection.

This script analyzes MCQ generation consistency by examining:
1. Within-run duplicates: Does a single prompt run produce duplicate questions?
2. Cross-run consistency: How similar are questions when running the same prompt multiple times?

Each other_analysis file represents one topic with 4 runs of the same prompt.
Creates both static (matplotlib/seaborn) and interactive (plotly) visualizations
with WCAG-compliant accessible color schemes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# WCAG AA-compliant colorblind-safe palette with modern, pleasant aesthetics
COLORS = {
    'problem': '#E63946',           # Coral red - problems (within-run duplicates)
    'high_consistency': '#06A77D',  # Teal - good consistency
    'medium_consistency': '#F77F00', # Vibrant orange - medium consistency
    'low_consistency': '#FFB703',   # Golden yellow - low consistency
    'discipline1': '#118AB2',       # Ocean blue - Data Science
    'discipline2': '#6A4C93',       # Royal purple - Computer Science
    'discipline3': '#D62828',       # Deep red - Discrete Math
}

# Ensure high contrast for accessibility
plt.rcParams['axes.edgecolor'] = '#000000'
plt.rcParams['axes.labelcolor'] = '#000000'
plt.rcParams['text.color'] = '#000000'
plt.rcParams['xtick.color'] = '#000000'
plt.rcParams['ytick.color'] = '#000000'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

class PromptConsistencyAnalyzer:
    """Analyze prompt consistency and within-run duplicate/similar question detection.
    
    Within-run metrics:
    - Duplicates (≥85% similarity): Definite problems - same questions repeated in one run
    - Similar (70-85% similarity): Potentially acceptable - related but distinct questions
    
    Cross-run metrics:
    - Consistency: How repeatable the prompt is across multiple runs (separate analysis)
    """
    
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.data = None
        
    def load_all_files(self):
        """Load and parse all other_analysis files."""
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
                print(f"Warning: Folder not found: {folder_path}")
                continue
                
            files = list(folder_path.glob('other_analysis_*.csv'))
            print(f"Found {len(files)} files in {folder}")
            
            for file in files:
                try:
                    data = self._parse_file(file, discipline, folder)
                    if data:
                        all_data.append(data)
                except Exception as e:
                    print(f"Error parsing {file.name}: {e}")
                    
        self.data = pd.DataFrame(all_data)
        print(f"\nTotal topics loaded: {len(self.data)}")
        print(f"Disciplines: {self.data['discipline'].unique()}")
        return self.data
    
    def _parse_file(self, filepath, discipline, source_folder):
        """Parse a single other_analysis CSV file."""
        filename = filepath.stem
        topic = filename.replace('other_analysis_', '').rsplit('_', 1)[0]
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Parse Within-Run Analysis
            within_run_data = []
            cross_run_data = []
            
            # Find Within-Run section
            for i, line in enumerate(lines):
                if 'Within-Run Analysis' in line:
                    # Parse next 4 lines (one for each run)
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
            
            # Find Cross-Run section
            for i, line in enumerate(lines):
                if 'Cross-Run Analysis' in line:
                    # Parse comparison lines
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
            
            if not within_run_data or not cross_run_data:
                print(f"  Warning: Incomplete data in {filepath.name}")
                return None
            
            # Calculate metrics
            total_questions = sum(r['question_count'] for r in within_run_data)
            total_within_duplicates = sum(r['internal_duplicates'] for r in within_run_data)
            total_within_similar = sum(r['internal_similar'] for r in within_run_data)
            runs_with_duplicates = sum(1 for r in within_run_data if r['internal_duplicates'] > 0)
            runs_with_similar = sum(1 for r in within_run_data if r['internal_similar'] > 0)
            
            total_cross_duplicates = sum(r['duplicates'] for r in cross_run_data)
            total_cross_similar = sum(r['similar'] for r in cross_run_data)
            total_cross_comparisons = sum(r['total_comparisons'] for r in cross_run_data)
            
            cross_consistency_pct = ((total_cross_duplicates + total_cross_similar) / total_cross_comparisons * 100) if total_cross_comparisons > 0 else 0
            
            return {
                'discipline': discipline,
                'source_folder': source_folder,
                'topic': topic,
                # Within-run metrics (problems!)
                'total_questions': total_questions,
                'avg_questions_per_run': total_questions / 4,
                'within_run_duplicates': total_within_duplicates,
                'within_run_similar': total_within_similar,
                'runs_with_duplicates': runs_with_duplicates,
                'runs_with_similar': runs_with_similar,
                'has_within_run_duplicates': total_within_duplicates > 0,
                'has_within_run_similar': total_within_similar > 0,
                # Cross-run metrics (consistency!)
                'cross_run_duplicates': total_cross_duplicates,
                'cross_run_similar': total_cross_similar,
                'cross_run_comparisons': total_cross_comparisons,
                'cross_run_consistency_pct': cross_consistency_pct,
                'cross_run_duplicate_pct': (total_cross_duplicates / total_cross_comparisons * 100) if total_cross_comparisons > 0 else 0,
                'cross_run_similar_pct': (total_cross_similar / total_cross_comparisons * 100) if total_cross_comparisons > 0 else 0,
                'filepath': str(filepath)
            }
                
        except Exception as e:
            print(f"  Error reading {filepath.name}: {e}")
            return None
    
    def create_static_visualizations(self, output_dir=None):
        """Create all static visualizations."""
        if output_dir is None:
            output_dir = self.base_path
        else:
            output_dir = Path(output_dir)
        
        print("\n=== Creating Static Visualizations ===")
        
        # 1. Within-run duplicates and similar analysis
        self._create_within_run_problems(output_dir)
        
        # 2. Cross-run consistency overview
        self._create_cross_run_consistency(output_dir)
        
        # 3. Discipline comparison
        self._create_discipline_comparison(output_dir)
        
        print("✓ Static visualizations complete")
    
    def _create_within_run_problems(self, output_dir):
        """Visualize within-run duplicates and similar questions."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Left: Topics with within-run duplicates and similar by discipline
        problem_summary = self.data.groupby('discipline').agg({
            'has_within_run_duplicates': 'sum',
            'has_within_run_similar': 'sum',
            'topic': 'count'
        }).reset_index()
        problem_summary['clean'] = (problem_summary['topic'] - 
                                    problem_summary['has_within_run_duplicates'] - 
                                    problem_summary['has_within_run_similar'])
        # Note: a topic can have both duplicates and similar
        problem_summary['only_duplicates'] = problem_summary['has_within_run_duplicates']
        problem_summary['only_similar'] = problem_summary['has_within_run_similar']
        
        x = np.arange(len(problem_summary))
        width = 0.35
        
        # Duplicates (definitely problems)
        p1 = ax1.bar(x - width/2, problem_summary['only_duplicates'], width,
                    label='With Duplicates (≥85%, Problem!)',
                    color=COLORS['problem'], edgecolor='black', linewidth=1.5)
        # Similar (potentially okay)
        p2 = ax1.bar(x + width/2, problem_summary['only_similar'], width,
                    label='With Similar (70-85%, Maybe OK)',
                    color=COLORS['medium_consistency'], edgecolor='black', linewidth=1.5)
        
        ax1.set_xlabel('Discipline', fontweight='bold', fontsize=13)
        ax1.set_ylabel('Number of Topics', fontweight='bold', fontsize=13)
        ax1.set_title('Within-Run Analysis: Duplicates vs Similar Questions\n(Duplicates = Problem | Similar = Potentially Acceptable)',
                     fontweight='bold', fontsize=14, pad=20)
        ax1.set_xticks(x)
        ax1.set_xticklabels(problem_summary['discipline'], fontsize=11)
        ax1.legend(loc='upper right', frameon=True, edgecolor='black', fontsize=10)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, (dup, sim) in enumerate(zip(problem_summary['only_duplicates'], 
                                            problem_summary['only_similar'])):
            if dup > 0:
                ax1.text(i - width/2, dup + 0.3, f'{int(dup)}', ha='center', va='bottom',
                        fontweight='bold', fontsize=11)
            if sim > 0:
                ax1.text(i + width/2, sim + 0.3, f'{int(sim)}', ha='center', va='bottom',
                        fontweight='bold', fontsize=11)
        
        # Right: Distribution of within-run duplicates (the real problems)
        duplicate_topics = self.data[self.data['has_within_run_duplicates']]
        
        if len(duplicate_topics) > 0:
            top_duplicates = duplicate_topics.nlargest(15, 'within_run_duplicates')
            
            y = np.arange(len(top_duplicates))
            ax2.barh(y, top_duplicates['within_run_duplicates'], height=0.4,
                    label='Duplicates (≥85%)', color=COLORS['problem'], edgecolor='black', linewidth=1)
            ax2.barh(y + 0.4, top_duplicates['within_run_similar'], height=0.4,
                    label='Similar (70-85%)', color=COLORS['medium_consistency'], edgecolor='black', linewidth=1)
            
            ax2.set_yticks(y + 0.2)
            ax2.set_yticklabels([f"{row['topic'][:30]}... ({row['discipline'][:8]})" 
                                 for _, row in top_duplicates.iterrows()], fontsize=9)
            ax2.set_xlabel('Count', fontweight='bold', fontsize=12)
            ax2.set_title('Top 15 Topics with Within-Run Duplicates\n(Duplicates = Definite Problem)',
                         fontweight='bold', fontsize=14, pad=15)
            ax2.legend(loc='lower right', frameon=True, edgecolor='black')
            ax2.grid(axis='x', alpha=0.3, linestyle='--')
        else:
            ax2.text(0.5, 0.5, 'No within-run duplicates detected!', 
                    ha='center', va='center', fontsize=14, transform=ax2.transAxes)
            ax2.set_xticks([])
            ax2.set_yticks([])
        
        plt.tight_layout()
        filename = 'within_run_duplicates_and_similar.png'
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"✓ Saved {filename}")
    
    def _create_cross_run_consistency(self, output_dir):
        """Visualize cross-run consistency."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
        
        # 1. Consistency distribution by discipline
        for discipline in self.data['discipline'].unique():
            disc_data = self.data[self.data['discipline'] == discipline]
            color = {'Data Science': COLORS['discipline1'],
                    'Computer Science': COLORS['discipline2'],
                    'Discrete Mathematics': COLORS['discipline3']}.get(discipline)
            
            ax1.hist(disc_data['cross_run_consistency_pct'], bins=20, alpha=0.6,
                    label=discipline, color=color, edgecolor='black', linewidth=1)
        
        ax1.set_xlabel('Cross-Run Consistency (%)', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Number of Topics', fontweight='bold', fontsize=12)
        ax1.set_title('Distribution of Cross-Run Consistency\n(Higher = More similar questions across runs)',
                     fontweight='bold', fontsize=14, pad=15)
        ax1.legend(frameon=True, edgecolor='black')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 2. Average consistency by discipline
        consistency_by_disc = self.data.groupby('discipline').agg({
            'cross_run_consistency_pct': 'mean',
            'cross_run_duplicate_pct': 'mean',
            'cross_run_similar_pct': 'mean'
        }).reset_index()
        
        x = np.arange(len(consistency_by_disc))
        width = 0.25
        
        ax2.bar(x - width, consistency_by_disc['cross_run_duplicate_pct'], width,
               label='Duplicates (≥85%)', color=COLORS['problem'], edgecolor='black', linewidth=1)
        ax2.bar(x, consistency_by_disc['cross_run_similar_pct'], width,
               label='Similar (70-85%)', color=COLORS['medium_consistency'], edgecolor='black', linewidth=1)
        ax2.bar(x + width, consistency_by_disc['cross_run_consistency_pct'], width,
               label='Total Consistency', color=COLORS['high_consistency'], edgecolor='black', linewidth=1)
        
        ax2.set_xlabel('Discipline', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Average Percentage (%)', fontweight='bold', fontsize=12)
        ax2.set_title('Average Cross-Run Consistency by Discipline',
                     fontweight='bold', fontsize=14, pad=15)
        ax2.set_xticks(x)
        ax2.set_xticklabels(consistency_by_disc['discipline'], fontsize=11)
        ax2.legend(frameon=True, edgecolor='black', fontsize=10)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 3. Most consistent topics
        most_consistent = self.data.nlargest(15, 'cross_run_consistency_pct')
        
        y = np.arange(len(most_consistent))
        ax3.barh(y, most_consistent['cross_run_consistency_pct'],
                color=COLORS['high_consistency'], edgecolor='black', linewidth=1)
        ax3.set_yticks(y)
        ax3.set_yticklabels([f"{row['topic'][:30]}... ({row['discipline'][:8]})"
                            for _, row in most_consistent.iterrows()], fontsize=9)
        ax3.set_xlabel('Consistency (%)', fontweight='bold', fontsize=12)
        ax3.set_title('Top 15 Most Consistent Topics\n(Similar questions across runs)',
                     fontweight='bold', fontsize=14, pad=15)
        ax3.grid(axis='x', alpha=0.3, linestyle='--')
        
        # 4. Least consistent topics
        least_consistent = self.data.nsmallest(15, 'cross_run_consistency_pct')
        
        y = np.arange(len(least_consistent))
        ax4.barh(y, least_consistent['cross_run_consistency_pct'],
                color=COLORS['low_consistency'], edgecolor='black', linewidth=1)
        ax4.set_yticks(y)
        ax4.set_yticklabels([f"{row['topic'][:30]}... ({row['discipline'][:8]})"
                            for _, row in least_consistent.iterrows()], fontsize=9)
        ax4.set_xlabel('Consistency (%)', fontweight='bold', fontsize=12)
        ax4.set_title('Top 15 Least Consistent Topics\n(Different questions across runs)',
                     fontweight='bold', fontsize=14, pad=15)
        ax4.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        filename = 'cross_run_consistency.png'
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"✓ Saved {filename}")
    
    def _create_discipline_comparison(self, output_dir):
        """Create comprehensive discipline comparison."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Summary statistics by discipline
        summary = self.data.groupby('discipline').agg({
            'topic': 'count',
            'has_within_run_duplicates': 'sum',
            'has_within_run_similar': 'sum',
            'within_run_duplicates': 'sum',
            'within_run_similar': 'sum',
            'cross_run_consistency_pct': ['mean', 'std'],
            'avg_questions_per_run': 'mean'
        }).round(2)
        
        disciplines = summary.index.tolist()
        
        # 1. Within-run duplicates rate (the actual problems)
        duplicate_rate = (summary[('has_within_run_duplicates', 'sum')] / summary[('topic', 'count')] * 100).values
        
        bars = ax1.bar(disciplines, duplicate_rate, color=[COLORS['discipline1'], COLORS['discipline2'], COLORS['discipline3']],
                      edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Percentage of Topics (%)', fontweight='bold', fontsize=12)
        ax1.set_title('Topics with Within-Run DUPLICATES by Discipline\n(≥85% similarity - Definite Problems)',
                     fontweight='bold', fontsize=14, pad=15)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar, val in zip(bars, duplicate_rate):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Average consistency
        consistency_mean = summary[('cross_run_consistency_pct', 'mean')].values
        consistency_std = summary[('cross_run_consistency_pct', 'std')].values
        
        bars = ax2.bar(disciplines, consistency_mean, 
                      yerr=consistency_std,
                      color=[COLORS['discipline1'], COLORS['discipline2'], COLORS['discipline3']],
                      edgecolor='black', linewidth=1.5,
                      error_kw={'linewidth': 2, 'ecolor': 'black'})
        ax2.set_ylabel('Average Consistency (%)', fontweight='bold', fontsize=12)
        ax2.set_title('Average Cross-Run Consistency by Discipline',
                     fontweight='bold', fontsize=14, pad=15)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar, val in zip(bars, consistency_mean):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. Questions per run
        questions = summary[('avg_questions_per_run', 'mean')].values
        
        bars = ax3.bar(disciplines, questions,
                      color=[COLORS['discipline1'], COLORS['discipline2'], COLORS['discipline3']],
                      edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('Average Questions per Run', fontweight='bold', fontsize=12)
        ax3.set_title('Average Questions Generated per Run',
                     fontweight='bold', fontsize=14, pad=15)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar, val in zip(bars, questions):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Summary table
        table_data = []
        for disc in disciplines:
            disc_data = self.data[self.data['discipline'] == disc]
            table_data.append([
                disc,
                len(disc_data),
                disc_data['has_within_run_duplicates'].sum(),
                disc_data['has_within_run_similar'].sum(),
                f"{disc_data['cross_run_consistency_pct'].mean():.1f}%",
                f"{disc_data['avg_questions_per_run'].mean():.1f}"
            ])
        
        ax4.axis('tight')
        ax4.axis('off')
        table = ax4.table(cellText=table_data,
                         colLabels=['Discipline', 'Topics', 'W/ Duplicates', 'W/ Similar', 
                                   'Avg Cross-Run %', 'Avg Q/Run'],
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.22, 0.12, 0.16, 0.14, 0.18, 0.14])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(6):
            table[(0, i)].set_facecolor(COLORS['discipline1'])
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style data rows
        for i in range(1, len(table_data) + 1):
            for j in range(6):
                table[(i, j)].set_facecolor('#E3F2FD' if i % 2 == 0 else 'white')
                table[(i, j)].set_edgecolor('black')
        
        plt.tight_layout()
        filename = 'discipline_comparison.png'
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"✓ Saved {filename}")
    
    def create_interactive_dashboard(self, output_dir=None):
        """Create interactive Plotly dashboard."""
        if output_dir is None:
            output_dir = self.base_path
        else:
            output_dir = Path(output_dir)
        
        print("\n=== Creating Interactive Dashboard ===")
        
        # Create comprehensive dashboard
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Within-Run: Duplicates vs Similar by Discipline',
                'Cross-Run: Consistency Distribution',
                'Within-Run: Topics with Duplicates',
                'Cross-Run: Top Topics by Consistency',
                'Discipline Statistics',
                'Topics with Within-Run Duplicates'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'box'}],
                [{'type': 'bar'}, {'type': 'bar'}],
                [{'type': 'table'}, {'type': 'table'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.15
        )
        
        # 1. Within-run duplicates and similar by discipline
        problem_summary = self.data.groupby('discipline').agg({
            'has_within_run_duplicates': 'sum',
            'has_within_run_similar': 'sum',
            'topic': 'count'
        }).reset_index()
        
        fig.add_trace(
            go.Bar(
                name='With Duplicates (≥85%)',
                x=problem_summary['discipline'],
                y=problem_summary['has_within_run_duplicates'],
                marker_color=COLORS['problem'],
                marker_line_color='black',
                marker_line_width=1.5,
                hovertemplate='<b>%{x}</b><br>Topics with duplicates: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                name='With Similar (70-85%)',
                x=problem_summary['discipline'],
                y=problem_summary['has_within_run_similar'],
                marker_color=COLORS['medium_consistency'],
                marker_line_color='black',
                marker_line_width=1.5,
                hovertemplate='<b>%{x}</b><br>Topics with similar: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Consistency distribution box plot
        for discipline in self.data['discipline'].unique():
            disc_data = self.data[self.data['discipline'] == discipline]
            color = {'Data Science': COLORS['discipline1'],
                    'Computer Science': COLORS['discipline2'],
                    'Discrete Mathematics': COLORS['discipline3']}.get(discipline)
            
            fig.add_trace(
                go.Box(
                    name=discipline,
                    y=disc_data['cross_run_consistency_pct'],
                    marker_color=color,
                    line=dict(color='black', width=1.5),
                    showlegend=False,
                    hovertemplate='<b>%{fullData.name}</b><br>Consistency: %{y:.1f}%<extra></extra>'
                ),
                row=1, col=2
            )
        
        # 3. Within-run: Topics with duplicates (bar chart)
        duplicate_topics = self.data[self.data['has_within_run_duplicates']].nlargest(10, 'within_run_duplicates')
        
        if len(duplicate_topics) > 0:
            fig.add_trace(
                go.Bar(
                    name='Within-Run Duplicates',
                    x=duplicate_topics['topic'],
                    y=duplicate_topics['within_run_duplicates'],
                    marker_color=COLORS['problem'],
                    marker_line_color='black',
                    marker_line_width=1,
                    showlegend=False,
                    hovertemplate='<b>%{x}</b><br>Duplicates: %{y}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 4. Cross-run: Top topics by consistency
        top_topics = self.data.nlargest(10, 'cross_run_consistency_pct')
        
        fig.add_trace(
            go.Bar(
                x=top_topics['topic'],
                y=top_topics['cross_run_consistency_pct'],
                marker_color=COLORS['high_consistency'],
                marker_line_color='black',
                marker_line_width=1,
                showlegend=False,
                hovertemplate='<b>%{x}</b><br>Consistency: %{y:.1f}%<extra></extra>'
            ),
            row=2, col=2
        )
        
        # 5. Discipline statistics table
        stats = self.data.groupby('discipline').agg({
            'topic': 'count',
            'has_within_run_duplicates': 'sum',
            'has_within_run_similar': 'sum',
            'cross_run_consistency_pct': 'mean',
            'avg_questions_per_run': 'mean'
        }).reset_index()
        
        stats['cross_run_consistency_pct'] = stats['cross_run_consistency_pct'].round(1)
        stats['avg_questions_per_run'] = stats['avg_questions_per_run'].round(1)
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Discipline', 'Topics', 'W/ Duplicates', 'W/ Similar', 'Avg Consistency %', 'Avg Q/Run'],
                    fill_color=COLORS['discipline1'],
                    font=dict(color='white', size=11),
                    align='left',
                    line_color='black',
                    line_width=1.5
                ),
                cells=dict(
                    values=[stats['discipline'], stats['topic'], stats['has_within_run_duplicates'],
                           stats['has_within_run_similar'], stats['cross_run_consistency_pct'], stats['avg_questions_per_run']],
                    fill_color='#E3F2FD',
                    font=dict(color='black', size=10),
                    align='left',
                    line_color='black',
                    line_width=1
                )
            ),
            row=3, col=1
        )
        
        # 6. Topics with within-run duplicates table
        duplicate_topics = self.data[self.data['has_within_run_duplicates']].nlargest(10, 'within_run_duplicates')
        
        if len(duplicate_topics) > 0:
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=['Topic', 'Discipline', 'Within-Run Dups', 'Within-Run Similar'],
                        fill_color=COLORS['problem'],
                        font=dict(color='white', size=11),
                        align='left',
                        line_color='black',
                        line_width=1.5
                    ),
                    cells=dict(
                        values=[
                            [t[:40] + '...' if len(t) > 40 else t for t in duplicate_topics['topic']],
                            [d[:15] for d in duplicate_topics['discipline']],
                            duplicate_topics['within_run_duplicates'],
                            duplicate_topics['within_run_similar']
                        ],
                        fill_color='#FFEBEE',
                        font=dict(color='black', size=10),
                        align='left',
                        line_color='black',
                        line_width=1
                    )
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_xaxes(title_text="Discipline", row=1, col=1, showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text="Number of Topics", row=1, col=1, showgrid=True, gridcolor='lightgray')
        
        fig.update_yaxes(title_text="Cross-Run Consistency (%)", row=1, col=2, showgrid=True, gridcolor='lightgray')
        
        fig.update_xaxes(title_text="Topic", row=2, col=1, showgrid=True, gridcolor='lightgray', tickangle=-45)
        fig.update_yaxes(title_text="Within-Run Duplicates", row=2, col=1, showgrid=True, gridcolor='lightgray')
        
        fig.update_xaxes(title_text="Topic", row=2, col=2, showgrid=True, gridcolor='lightgray', tickangle=-45)
        fig.update_yaxes(title_text="Cross-Run Consistency (%)", row=2, col=2, showgrid=True, gridcolor='lightgray')
        
        fig.update_layout(barmode='group')
        
        fig.update_layout(
            title_text="MCQ Prompt Consistency Analysis Dashboard",
            title_font_size=20,
            title_font_color='black',
            showlegend=True,
            height=1400,
            font=dict(family='Arial', size=11, color='black'),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        filename = 'prompt_consistency_dashboard.html'
        fig.write_html(output_dir / filename)
        print(f"✓ Saved {filename}")
        
        return fig


def main():
    """Main execution function."""
    print("=" * 70)
    print("MCQ Prompt Consistency Analysis")
    print("=" * 70)
    
    base_path = Path(__file__).parent
    analyzer = PromptConsistencyAnalyzer(base_path)
    
    # Load data
    print("\n[1/3] Loading data from all disciplines...")
    data = analyzer.load_all_files()
    
    if data is None or len(data) == 0:
        print("Error: No data loaded. Exiting.")
        return
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(f"Total topics analyzed: {len(data)}")
    print(f"\nWITHIN-RUN ANALYSIS (Same prompt run):")
    print(f"  Topics with duplicates (≥85%): {data['has_within_run_duplicates'].sum()} ({data['has_within_run_duplicates'].sum()/len(data)*100:.1f}%)")
    print(f"  Topics with similar (70-85%): {data['has_within_run_similar'].sum()} ({data['has_within_run_similar'].sum()/len(data)*100:.1f}%)")
    print(f"\nCROSS-RUN ANALYSIS (Multiple runs of same prompt):")
    print(f"  Average consistency: {data['cross_run_consistency_pct'].mean():.1f}%")
    print(f"\nBy Discipline:")
    for disc in data['discipline'].unique():
        disc_data = data[data['discipline'] == disc]
        print(f"  {disc}:")
        print(f"    - Topics: {len(disc_data)}")
        print(f"    - Within-run duplicates: {disc_data['has_within_run_duplicates'].sum()} ({disc_data['has_within_run_duplicates'].sum()/len(disc_data)*100:.1f}%)")
        print(f"    - Within-run similar: {disc_data['has_within_run_similar'].sum()} ({disc_data['has_within_run_similar'].sum()/len(disc_data)*100:.1f}%)")
        print(f"    - Avg cross-run consistency: {disc_data['cross_run_consistency_pct'].mean():.1f}%")
    
    # Create visualizations
    print(f"\n[2/3] Generating static visualizations...")
    analyzer.create_static_visualizations()
    
    print(f"\n[3/3] Generating interactive dashboard...")
    analyzer.create_interactive_dashboard()
    
    print("\n" + "=" * 70)
    print("✓ Analysis complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  Static (PNG/PDF):")
    print("    - within_run_duplicates_and_similar.png/pdf")
    print("    - cross_run_consistency.png/pdf")
    print("    - discipline_comparison.png/pdf")
    print("  Interactive (HTML):")
    print("    - prompt_consistency_dashboard.html")
    print(f"\nAll files saved to: {base_path}")


if __name__ == "__main__":
    main()
