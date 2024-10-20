# Language Model Benchmark for Market Research Tasks

This repository contains the results and analysis of a benchmark study comparing the performance of five Language Models (LLMs) on market research-related tasks.

## Overview

We evaluated five state-of-the-art language models on their ability to perform various market research tasks, including trend analysis, competitive landscape assessment, and pricing strategy.

### Models Evaluated
- phi-2
- bloomz-1b7
- stablelm-2-1_6b
- tinyllama-1.1b-chat
- opt-1.3b

## Benchmark Questions

1. List the top 5 AI companies globally with the highest funding in 2023.
2. Identify the top 3 cloud service providers by market share in 2023 and provide their market share percentages.
3. List the top 5 trending libraries in the Data Science market as of Q4 2023, along with their primary use cases and growth rates over the past year.
4. What is the pricing range for JetBrains PyCharm? Which pricing package would you recommend a student?
5. Using the historical growth rate of the global AI market from 2020 to 2023, estimate the market size for 2025. Provide your reasoning and state any assumptions.

## Key Findings

### Performance Across Metrics
![Overall Performance](graphs/OverallPerformanceAcrossMetrics.png)
![Accuracy Scores](graphs/accuracy_across_questions_plot.png)
![Relevance Scores](graphs/relevance_across_questions_plot.png)
![Completeness Scores](graphs/completeness_across_questions_plot.png)
![Coherence Scores](graphs/coherence_across_questions_plot.png)
![Reasoning Scores](graphs/reasoning_across_questions_plot.png)

### Summary of Results

1. **phi-2** demonstrated the most consistent performance across all questions, excelling in accuracy, relevance, and completeness.
2. **bloomz-1b7** showed the fastest response times but had inconsistent performance across other metrics.
3. **tinyllama-1.1b-chat** performed exceptionally well in specific domains, particularly in questions about cloud service providers and data science libraries.
4. **stablelm-2-1_6b** showed moderate performance across most metrics, with strengths in market size estimation.
5. **opt-1.3b** consistently had the slowest response times and showed the weakest overall performance.

## Recommendations

- **phi-2** performs best overall, excelling in tasks requiring high accuracy and comprehensive responses.
- **tinyllama-1.1b-chat** offers a lightweight alternative with slightly slower performance, suitable for targeted research.
- Both models may lack precision in detailed numbers but provide generally accurate information.
- **bloomz-1b7** is ideal for generating quick, simple answers but struggles with detailed responses.

## Detailed Report

For a comprehensive analysis, including detailed evaluation matrices and in-depth discussion of results, please refer to the full report document in this repository.

## Repository Structure

- `README.md`: This file, providing an overview of the benchmark study.
- `full_report.pdf`: Comprehensive report with detailed analysis and evaluation matrices.
- `plots/`: Directory containing all performance plots.
- `data/`: Raw data and responses from the language models.
- `scripts/`: Code used for evaluation and plot generation.

## Methodology

Our evaluation was based on five criteria: Accuracy, Relevance, Completeness, Coherence, and Reasoning. We also measured response time for each model. The detailed methodology is available in the full report.

---

For any questions or further information, please open an issue in this repository.
