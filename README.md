# ICUM: Insight-Driven Codon Unveiling Matrix

**ICUM** is a Python-based framework for *protein inference and codon bias discovery* from noncanonical RNA sequences.  
It implements automated workflows for codon validation, identification of unknown codons, RSCU and ENC calculations, and entropy-based codon bias analysis. It processes codons, detects start/stop codons, calculates GC content, classifies amino acid frequencies, detects unknown codons, and visualizes key features using clustering and statistical plots.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17414482.svg)](https://doi.org/10.5281/zenodo.17414482)

---

## Features
- Extraction and validation of codons from RNA sequences.  
- Detection of noncanonical and unknown codons.  
- Calculation of **RSCU**, **ENC**, and **Shannon Entropy** metrics.  
- Summary statistics and result visualization-ready output.  
- Easily extendable for other genomic datasets.

---

## Repository Structure

ICUM/
├── ICUM.py           # main source code
├── data/             # input RNA sequence files (CSV)
├── outputs/          # output files (codon bias, metrics, etc.)
├── requirements.txt  # Python dependencies
├── LICENSE           # License file (MIT License)
├── README.md         # Project documentation (this file)
└── .gitignore        # files/folders excluded from Git tracking

---

## Requirements and Installation
### Prerequisites
- Python 3.10 or higher  
- pip or conda environment  

### Installation
git clone https://github.com/abdulkawiy/ICUM.git
cd ICUM
pip install -r requirements.txt
Run the framework using: python ICUM.py --input data/Homo_sapiens.GRCh38.cdna.all.csv --output output/

## How to Use

1. Place your dataset in the `Data/` folder (must include a `sequence` column).
2. Run the tool: python ICUM.py

## Results will be saved in the output/ folder.
### 1. Core Analysis Reports:
1.1. biological_analysis_report.csv: It is the full codon-level report which contains:
- Complete analysis of ALL input sequences.
- Contains counts for: start codons, stop codons, GC-rich codons, unknown codons, amino acid frequencies.

1.2. valid_sequences_analysis_report.csv: It is the full codon-level report for only biologically valid sequences which contains filtered subset of sequences that meet biological criteria:
- Has start codon (AUG) at beginning
- Has exactly one stop codon at end
- Proper start/stop codon positioning

### 2. Statistical Reports:
2.1. statistical_report.txt & statistical_report_valid.txt: They contain:
- Statistical comparisons between sequence groups.
- GC content ANOVA tests.
- Correlation analyses (GC-rich vs amino acid counts).
- Group-wise statistical comparisons.

2.2. statistical_summary.csv & statistical_summary.png: They contain:
- Tabular summary of key statistical tests.
- Includes: t-tests, Mann-Whitney U, chi-square tests.
- p-values and test statistics for publication.

2.3. gc_stats.txt & gc_stats_valid.txt: They contain:
- Detailed GC content statistics by cluster.
- ANOVA results for GC content differences.

### 3. VISUALIZATION FILES
3.1. plot_gc_content.png & plot_gc_content_valid.png: They contain:
- Distribution of GC content across sequences.
- Histogram with KDE (Kernel Density Estimate).

3.2. compare_gc_content_valid_vs_all.png: It contains:
- Overlay comparison between all vs valid sequences.
- Shows how filtering affects GC distribution

### 4. Codon Analysis
4.1. plot_start_stop_codons.png & plot_start_stop_codons_valid.png: Bar plots comparing total start vs stop codon counts.

4.2. plot_start_stop_codons_frequency.png & plot_start_stop_codons_frequency_valid.png: 
- Detailed frequencies of specific start/stop codons.
- Shows AUG (start) vs UAA/UAG/UGA (stop) usage.

4.3. codon_frequency_barplot.png:
- Frequency of all 64 codons across sequences.
- Sorted by usage frequency.

4.4. codon_frequency_grouped.png: 
- Codon frequencies grouped by amino acid.
- Color-coded by amino acid type.

### 5. Amino Acid Analysis
5.1. plot_aa_heatmap.png & plot_aa_heatmap_valid.png:
- Heatmap of amino acid frequencies per transcript.
- Rows = sequences, columns = amino acids.

5.2. plot_aa_heatmap_2.png & plot_aa_heatmap_valid_2.png: Normalized versions (relative frequencies per sequence).

5.3. plot_clustered_aa_heatmap.png & plot_clustered_aa_heatmap_valid.png:
- Sequences clustered by amino acid usage patterns.
- K-means clustering (3 clusters).

5.4. plot_avg_aa_composition.png & plot_avg_aa_composition_valid.png:
- Global average amino acid usage across all sequences.
- Bar plot sorted by frequency.

### 6. Unknown Codon Analysis
6.1. unknown_codon_pie.png:
- Pie chart showing proportion of sequences with unknown codons.
- Helps identify potential sequencing errors or mutations.

6.2. plot_top_unknown_codons.png:
- Bar plot of most frequent unknown codons.
- Useful for identifying common mutations or artifacts.

### 7. Sequence Validation
valid_sequences_pie.png:
- Proportion of valid vs invalid sequences.
- Shows data quality and filtering efficiency.

### 8. RSCU (Relative Synonymous Codon Usage)
8.1. rscu_table.csv & rscu_table_valid.csv: 
- RSCU values for each codon (all and valid sequences).
- RSCU = 1: uniform usage, >1: overrepresented, <1: underrepresented.

8.2. rscu_plot_1.png/pdf & rscu_plot_valid_1.png/pdf:
- Bar plots of RSCU values for all codons.
- Color-coded by amino acid.

8.3. rscu_table_grouping.csv & rscu_plot_grouping.png/pdf: Grouped versions with amino acid separators.

### 9. ENC (Effective Number of Codons)
9.1. enc_scored_sequences.csv & enc_scored_sequences_valid.csv:
- Sequences with calculated ENC scores (20-61 range).
- Lower ENC = more biased codon usage.

9.2. enc_distribution.png/pdf & enc_distribution_valid.png/pdf: Histograms of ENC score distribution.

9.3. enc_scored_sequences_GC3.csv & enc_scored_sequences_GC3_valid.csv: ENC scores with GC3 content (3rd codon position GC).

9.4. enc_statistics_combine.png/pdf & enc_statistics_combine_valid.png/pdf: Combined histograms and KDE plots of ENC distribution.

9.5. enc_vs_gc3.png/pdf & enc_vs_gc3_valid.png/pdf:
- ENC vs GC3 with Wright theoretical curve.
- Points below curve indicate selection pressure beyond GC bias.

### 10. Codon Usage Entropy
10.1. rscu_entropy.csv & rscu_entropy_valid.csv: 
- Entropy scores per sequence (measure of codon usage randomness).
- Higher entropy = more random/unbiased codon usage.
  
10.2. codon_entropy_distribution.png/pdf & codon_entropy_distribution_valid.png/pdf: Distribution of entropy scores.

10.3. codon_entropy_violin.png/pdf & codon_entropy_violin_valid.png/pdf: Violin + box plots showing entropy spread.

### 11. Protein Translation
11.1. valid_sequences_with_proteins.csv:
- Valid sequences with translated protein sequences.
- Uses standard genetic code.

11.2. valid_proteins_cleaned.fasta:
- Protein sequences in FASTA format.
- Suitable for protein database searches.

### 12. Cross-Metric Comparisons
12.1. rscu_enc_comparison.png/pdf & rscu_enc_comparison_valid.png/pdf:
- Side-by-side RSCU heatmap and ENC distribution.
- Comprehensive codon usage analysis.

12.2. rscu_enc_comparison_amino_acid.png/pdf & rscu_enc_comparison_amino_acid_valid.png/pdf: Enhanced version with amino acid grouping.

### 13. Codon Usage Heatmaps
13.1. codon_usage_heatmap_all.png & codon_usage_heatmap_all_valid.png:
- 1st vs 2nd base codon usage. 
- 4x4 heatmap visualization.

13.2. codon_usage_heatmap_full.png & codon_usage_heatmap_full_valid.png:
- 1st+2nd base vs 3rd base.
- 16x4 comprehensive view.

13.3. codon_usage_heatmap_64.png & codon_usage_heatmap_64_valid.png:
- Complete 64-codon visualization.
- 8x8 grid with codon labels and counts.

## Biological Insights:
RSCU plots: Identify codon preferences and potential optimization.
ENC plots: Assess overall codon usage bias and selective pressures.
GC content: Understand compositional constraints.
Amino acid heatmaps: Reveal sequence similarity patterns.

## Example Dataset
Make sure your .csv file contains at least:
id,sequence
1,AUGGCUUAA...
2,GCAUAGUGA...

## License
This project is licensed under the Academic Using License. See LICENSE for details.

## Citation
If you use this tool in your research, please cite:
```bash
Authors: 
- family-name: Al-Shamiri
  given-name: Abdulkawi Yahya Radman
- family-name: Yu
  given-name: Dong-Jun
Title: ICUM: A Framework for Protein Inference and Codon Bias Discovery from Noncanonical RNA Sequences
Journal: Journal of Chemical Information and Modeling
DOI: 10.5281/zenodo.xxxxxxx
URL: https://github.com/abdulkawiy/ICUM
Version: 1.0
```

## Contact
For questions, please contact: Abdulkawi Yahya Radman Al-Shamiri
Email: abdulkawiy@yahoo.com


## Data Availability
The example RNA datasets are derived from public Ensembl GRCh38 resources.
Processed data and scripts required to reproduce all results are available in this repository and will be archived in Zenodo upon publication.
