import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Used to generate the plots


def single_plot():
    """Generates one plot for each metrics, containing the performance of each model across all questions
    @:requires: the manual scores for each metric, model and question"""
    data = {
        'Question': ['Q1', 'Q2', 'Q3', 'Q4', 'Q5'] * 5,
        'Model': ['phi-2'] * 5 + ['bloomz-1b7'] * 5 + ['stablelm-2-1_6b'] * 5 + ['tinyllama-1.1b-chat'] * 5 + [
            'opt-1.3b'] * 5,
        'Acc': [4, 4, 4, 3, 4, 3, 3, 1, 2, 2, 3, 2, 3, 3, 3, 2, 4, 4, 4, 3, 1, 2, 2, 4, 3],
        'Rel': [5, 5, 5, 4, 5, 4, 4, 1, 2, 2, 4, 2, 4, 3, 4, 2, 5, 5, 4, 4, 1, 3, 2, 4, 3],
        'Comp': [4, 5, 5, 3, 5, 3, 3, 1, 2, 1, 4, 2, 5, 3, 5, 4, 5, 5, 4, 5, 1, 2, 3, 4, 4],
        'Coh': [4, 4, 4, 3, 3, 4, 4, 1, 4, 1, 4, 2, 3, 3, 4, 2, 5, 4, 3, 3, 1, 2, 3, 2, 3],
        'Reas': [3, 5, 5, 2, 4, 1, 5, 1, 1, 1, 2, 1, 3, 2, 5, 2, 5, 5, 5, 4, 1, 2, 2, 2, 3]
    }

    df = pd.DataFrame(data)

    # Set the style for a dark theme with improved contrast
    plt.style.use('dark_background')
    sns.set_style("darkgrid", {'axes.facecolor': '#1e1e1e'})

    metrics = ['Acc', 'Rel', 'Comp', 'Coh', 'Reas']
    metric_names = ['Accuracy', 'Relevance', 'Completeness', 'Coherence', 'Reasoning']

    for metric, metric_name in zip(metrics, metric_names):
        plt.figure(figsize=(12, 7))

        # Use lineplot for each model
        for model in df['Model'].unique():
            model_data = df[df['Model'] == model]
            sns.lineplot(x='Question', y=metric, data=model_data, marker='o', linewidth=2, markersize=8, label=model)

        plt.title(f'{metric_name} Scores Across Questions', fontsize=16, color='white')
        plt.ylim(0, 6)
        plt.ylabel('Score', fontsize=12, color='white')
        plt.xlabel('Question', fontsize=12, color='white')
        plt.xticks(fontsize=12, color='white')
        plt.yticks(fontsize=12, color='white')
        # Improve grid visibility
        plt.grid(True, linestyle='--', alpha=0.7, color='gray')

        # Add legend
        legend = plt.legend(title='Model', title_fontsize='12', fontsize='10',
                            bbox_to_anchor=(1.05, 1), loc='upper left')
        legend.get_frame().set_facecolor('#D3D3D3')  # Light gray color
        legend.get_frame().set_edgecolor('white')  # White edge color
        legend.get_frame().set_alpha(0.8)  # Slight transparency

        # Use tight layout to prevent clipping of tick-labels
        plt.tight_layout()

        # Save the plot
        plt.savefig(f'../graphs/{metric_name.lower()}_across_questions_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

    print("All plots have been saved as separate files.")


def overall_plot():
    """Generates one plot displaying the performance of each model across all metrics
        @:requires: the manual overall scores for each metric and model"""
    models = ['phi-2', 'bloomz-1b7', 'stablelm-2-1_6b', 'tinyllama-1.1b-chat', 'opt-1.3b']
    metrics = ['Acc', 'Rel', 'Comp', 'Coh', 'Reas', 'RT']
    values = np.array([
        [4, 5, 4, 4, 4, 49.83],
        [2, 3, 2, 2, 1, 0.74],
        [3, 4, 3, 3, 2, 71.67],
        [3, 4, 5, 3, 4, 40.13],
        [3, 3, 3, 2, 2, 102.30]
    ])

    # Normalize RT values to be on the same scale as other metrics
    values[:, -1] = 5 - (values[:, -1] - np.min(values[:, -1])) / (np.max(values[:, -1]) - np.min(values[:, -1])) * 4

    # Set up the radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)

    # Add the first value to the end of each row to close the polygon
    values = np.c_[values, values[:, 0]]

    angles = np.concatenate((angles, [angles[0]]))

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Plot data and fill area
    for i, model in enumerate(models):
        ax.plot(angles, values[i], 'o-', linewidth=2, label=model)
        ax.fill(angles, values[i], alpha=0.25)

    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)

    # Set y-axis limits
    ax.set_ylim(0, 5)

    # Add legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('Model Performance Across Metrics', fontsize=20, y=1.08)

    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.savefig('graphs/OverallPerformanceAcrossMetrics.png')
    plt.show()


if __name__ == "__main__":
    overall_plot()
    single_plot()
