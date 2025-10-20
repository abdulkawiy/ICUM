import csv
from collections import Counter, defaultdict
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
from scipy.stats import f_oneway, pearsonr, ttest_ind, mannwhitneyu, chisquare
import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import re
from Bio.Data import CodonTable
from matplotlib import colors as mcolors


# ==========================================================================================
# ==========================================================================================


# ==============================================
#             BIOLOGICAL DEFINITIONS
# ==============================================


# Start codon initiates protein translation
start_codons = ['AUG']

# Stop codons terminate protein synthesis
stop_codons = ['UAA', 'UAG', 'UGA']

# GC-rich codons (custom category for this analysis)
gc_rich_codons = ['GCG', 'CGC', 'GCC', 'CCG', 'CGG', 'GGC']



# Create output directory if it doesn't exist
output_dir = os.path.join("output")
os.makedirs(output_dir, exist_ok=True)


# Standard genetic code mapping (codon → amino acid)
# Covers all 64 possible codons:
# - 61 amino acid-coding codons
# - 3 stop codons (handled separately)
genetic_code = {
    # Phenylalanine
    'UUU': 'F', 'UUC': 'F',
    # Leucine
    'UUA': 'L', 'UUG': 'L', 'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
    # Isoleucine
    'AUU': 'I', 'AUC': 'I', 'AUA': 'I',
    # Methionine (also start codon)
    'AUG': 'M',
    # Valine
    'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
    # Serine
    'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S', 'AGU': 'S', 'AGC': 'S',
    # Proline
    'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    # Threonine
    'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    # Alanine
    'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    # Tyrosine
    'UAU': 'Y', 'UAC': 'Y',
    # Histidine
    'CAU': 'H', 'CAC': 'H',
    # Glutamine
    'CAA': 'Q', 'CAG': 'Q',
    # Asparagine
    'AAU': 'N', 'AAC': 'N',
    # Lysine
    'AAA': 'K', 'AAG': 'K',
    # Aspartic Acid
    'GAU': 'D', 'GAC': 'D',
    # Glutamic Acid
    'GAA': 'E', 'GAG': 'E',
    # Cysteine
    'UGU': 'C', 'UGC': 'C',
    # Tryptophan
    'UGG': 'W',
    # Arginine
    'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',
    # Glycine
    'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
}

# Synonymous codon dictionary (RNA codons)
synonymous_codons = {
    'F': ['UUU', 'UUC'],
    'L': ['UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG'],
    'I': ['AUU', 'AUC', 'AUA'],
    'M': ['AUG'],
    'V': ['GUU', 'GUC', 'GUA', 'GUG'],
    'S': ['UCU', 'UCC', 'UCA', 'UCG', 'AGU', 'AGC'],
    'P': ['CCU', 'CCC', 'CCA', 'CCG'],
    'T': ['ACU', 'ACC', 'ACA', 'ACG'],
    'A': ['GCU', 'GCC', 'GCA', 'GCG'],
    'Y': ['UAU', 'UAC'],
    '*': ['UAA', 'UAG', 'UGA'],
    'H': ['CAU', 'CAC'],
    'Q': ['CAA', 'CAG'],
    'N': ['AAU', 'AAC'],
    'K': ['AAA', 'AAG'],
    'D': ['GAU', 'GAC'],
    'E': ['GAA', 'GAG'],
    'C': ['UGU', 'UGC'],
    'W': ['UGG'],
    'R': ['CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'G': ['GGU', 'GGC', 'GGA', 'GGG']
}


# ==========================================================================================
# ==========================================================================================


# ========================================
#             HELPER FUNCTIONS
# ========================================


def normalize_sequence(seq):
    """Convert DNA to RNA (T to U), and make uppercase."""
    return seq.upper().replace('T', 'U') 

def is_gc_rich(codon, threshold=0.66):
    """Optional dynamic check: codon is GC-rich if ≥66% of bases are G/C."""
    gc = sum(1 for nt in codon if nt in ('G', 'C'))
    return gc / 3 >= threshold


# ==========================================================================================
# ==========================================================================================


# =====================================
#             Main Function
# =====================================


def process_sequence(sequence, use_dynamic_gc=False):
    """
    Analyze an RNA sequence and categorize each codon.
    
    Biological processing steps:
    1. Scan sequence in 3-nucleotide chunks (codons)
    2. For each complete codon:
        a. Check if valid (known in genetic code or stop codon)
        b. If invalid: classify as unknown
        c. If valid:
            - Count if start codon (AUG)
            - Count if stop codon (UAA/UAG/UGA)
            - Count if GC-rich codon
            - For amino acid-coding codons: increment amino acid count
    
    Returns:
        Dictionary with counts for each category
    """
    # Initialize counters - all biological categories
    start_count = 0
    stop_count = 0
    gc_rich_count = 0
    unknown_codon_count = 0
    unknown_codons = set()  # Track distinct unknown codons
    
    aa_counts = Counter()
    
    sequence = normalize_sequence(sequence)
    
    """
    # Initialize amino acid counts with all possible amino acids
    amino_acids = sorted(set(genetic_code.values()))
    aa_counts = {aa: 0 for aa in amino_acids}
    """
    
    # Process each codon in the sequence
    for i in range(0, len(sequence), 3):
        # Extract codon (3 nucleotides)
        codon = sequence[i:i+3]
        
        # Skip incomplete codons (end of sequence)
        if len(codon) != 3:
            continue
        
        # UNKNOWN CODON HANDLING
        # -----------------------
        # Codons not in genetic code and not stop codons
        if codon not in genetic_code and codon not in stop_codons:
            unknown_codon_count += 1
            unknown_codons.add(codon)
            continue  # Skip biological processing
        
        # BIOLOGICAL CATEGORIZATION
        # -------------------------
        # Start codon check (AUG)
        if codon in start_codons:
            start_count += 1
        
        # Stop codon check
        if codon in stop_codons:
            stop_count += 1
        # Amino acid translation (only for coding codons)
        else:
            aa = genetic_code[codon]
            aa_counts[aa] += 1
        
        # GC-rich codon check
        if use_dynamic_gc:
            if is_gc_rich(codon):
                gc_rich_count += 1
        else:
            if codon in gc_rich_codons:
                gc_rich_count += 1
    
    
    all_amino_acids = sorted(set(genetic_code.values()))
    aa_counts_complete = {aa: aa_counts.get(aa, 0) for aa in all_amino_acids}

    
    
    # Format unknown codons for output
    unknown_codon_str = ', '.join(sorted(unknown_codons)) if unknown_codons else ''
    
    # Return comprehensive biological analysis
    return {
        'start': start_count,
        'stop': stop_count,
        'gc_rich': gc_rich_count,
        'unknown_codon_count': unknown_codon_count,
        'unknown_codon': unknown_codon_str,
        'amino_acids': aa_counts_complete
    }


# =============================================================================
# =============================================================================


# ================================================
#             MAIN PROCESSING PIPELINE
# ================================================


def main(input_file, output_file):
    """
    Process RNA sequences from CSV and generate biological analysis report.
    
    Steps:
    1. Read input CSV with RNA sequences
    2. For each sequence:
        - Perform codon analysis
        - Collect biological metrics
    3. Write comprehensive report to output CSV
    """
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return
    
    # Open input/output files
    with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        
        # Determine amino acids for consistent column order
        amino_acids = sorted(set(genetic_code.values()))
        
        
        # Configure output columns
        fieldnames = reader.fieldnames + [
            'start', 'stop', 'gc_rich'
        ] + amino_acids + [
            'unknown_codon_count', 'unknown_codon'
        ]
        
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each RNA sequence
        for row in reader:
            # Get RNA sequence from input
            seq = row.get('sequence', '').strip()
            if not seq:
                continue  # Skip empty rows
            
            
            # Perform biological analysis
            results = process_sequence(seq, use_dynamic_gc=True)
            
            # Prepare output row
            output_row = row.copy()
            
            # Add codon counts
            output_row.update({
                'start': results['start'],
                'stop': results['stop'],
                'gc_rich': results['gc_rich'],
                'unknown_codon_count': results['unknown_codon_count'],
                'unknown_codon': results['unknown_codon']
            })
            
            # Add amino acid counts
            for aa in amino_acids:
                output_row[aa] = results['amino_acids'][aa]
            
            # Write to output
            writer.writerow(output_row)
            
    print(f"\nAnalysis complete! Results saved to '{output_file}'.")


# =============================================================================
# =============================================================================


# =====================================================================
#             Visualization & Statistical Results as Report
# =====================================================================


def visualize(output_file, valid=False):
    df = pd.read_csv(output_file, low_memory=False)
    
    df["sequence"] = df["sequence"].apply(normalize_sequence)
    
    df["GC_content"] = df["sequence"].apply(lambda seq: 100 * (seq.upper().count("G") + seq.upper().count("C")) / len(seq) if len(seq) > 0 else 0)

    # GC-content histogram
    #plt.figure(figsize=(8, 5))
    plt.figure(figsize=(10, 6), dpi=1200)
    sns.histplot(df["GC_content"], bins=30, kde=True, color="teal")  
    #plt.title("GC Content Distribution")
    plt.title("GC Content Distribution", fontsize=14, weight="bold")
    #plt.xlabel("GC Content (%)")
    plt.xlabel("GC Content (%)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.tight_layout()
    if valid:
        plt.savefig("output/plot_gc_content_valid.png", dpi=1200, bbox_inches="tight")
    else:
        plt.savefig("output/plot_gc_content.png", dpi=1200, bbox_inches="tight")
    plt.show()

    # Start/Stop codon frequency
    start_stop = pd.DataFrame({"Codon": ["Start", "Stop"], "Count": [df["start"].sum(), df["stop"].sum()]})
    #plt.figure(figsize=(6, 4))
    plt.figure(figsize=(8, 5), dpi=1200)
    #sns.barplot(data=start_stop, x="Codon", y="Count")
    sns.barplot(data=start_stop, x="Codon", y="Count", palette="viridis")
    #plt.title("Start vs Stop Codon Count")
    plt.title("Start vs Stop Codon Count", fontsize=14, weight="bold")
    plt.xlabel("Codon", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.tight_layout()
    if valid:
        plt.savefig("output/plot_start_stop_codons_valid.png", dpi=1200, bbox_inches="tight")
    else:
        plt.savefig("output/plot_start_stop_codons.png", dpi=1200, bbox_inches="tight")
    plt.show()
    
    # Start/Stop Codons Count
    start_codon = []
    stop_codon = ['UAA', 'UAG', 'UGA']
    stop_count = Counter()

    for seq in df['sequence']:
        seq = seq.upper()
        if len(seq) >= 3:
            start = seq[:3]
            stop = seq[-3:]
            start_codon.append(start)
            if stop in stop_codon:
                stop_count[stop] += 1

    start_counts = Counter(start_codon)

    # Combine into DataFrame
    combined_counts = pd.DataFrame({
        "codon": list(set(['AUG'] + stop_codon)),
        "count": [start_counts['AUG']] + [stop_count.get(c, 0) for c in stop_codon]
    })
    
    # Bar plot
    #plt.figure(figsize=(8, 5))
    plt.figure(figsize=(8, 5), dpi=1200)
    sns.barplot(data=combined_counts, x="codon", y="count", palette="Set2")
    #plt.title("Start and Stop Codon Frequencies")
    plt.title("Start and Stop Codon Frequencies", fontsize=14, weight="bold")
    #plt.xlabel("Codon")
    plt.xlabel("Codon", fontsize=12)
    #plt.ylabel("Count")
    plt.ylabel("Count", fontsize=12)
    plt.tight_layout()
    if valid:
        plt.savefig("output/plot_start_stop_codons_frequency_valid.png", dpi=1200, bbox_inches="tight")
    else:
        plt.savefig("output/plot_start_stop_codons_frequency.png", dpi=1200, bbox_inches="tight")
    plt.show()

    # Amino acid heatmap
    aa_cols = sorted(set(c for c in df.columns if len(c) == 1 and c.isalpha()))
    aa_df = df[aa_cols].fillna(0)
    
    #plt.figure(figsize=(14, 8))
    plt.figure(figsize=(16, 10), dpi=1200)
    #sns.heatmap(aa_df, cmap="YlGnBu")
    sns.heatmap(aa_df, cmap="YlGnBu", cbar_kws={"label": "Frequency"})
    #plt.title("Amino Acid Frequency per Transcript")
    plt.title("Amino Acid Frequency per Transcript", fontsize=14, weight="bold")
    plt.xlabel("Amino Acid", fontsize=12)
    plt.ylabel("Transcript Index", fontsize=12)
    plt.tight_layout()
    if valid:
        plt.savefig("output/plot_aa_heatmap_valid.png", dpi=1200, bbox_inches="tight")
    else:
        plt.savefig("output/plot_aa_heatmap.png", dpi=1200, bbox_inches="tight")
    plt.show()

    # Clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(aa_df)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # Clustered AA heatmap
    clustered = df.copy().sort_values("cluster")
    
    #plt.figure(figsize=(14, 8))
    plt.figure(figsize=(16, 10), dpi=1200)
    #sns.heatmap(clustered[aa_cols], cmap="coolwarm")
    sns.heatmap(clustered[aa_cols], cmap="coolwarm", cbar_kws={"label": "Frequency"})
    #plt.title("Clustered Codon Usage Heatmap")
    plt.title("Clustered Codon Usage Heatmap", fontsize=14, weight="bold")
    plt.xlabel("Amino Acid", fontsize=12)
    plt.ylabel("Transcript Index (clustered)", fontsize=12)
    plt.tight_layout()
    if valid:
        plt.savefig("output/plot_clustered_aa_heatmap_valid.png", dpi=1200, bbox_inches="tight")
    else:
        plt.savefig("output/plot_clustered_aa_heatmap.png", dpi=1200, bbox_inches="tight")
    plt.show()
    
    # Amino acid heatmap (normalized frequencies)
    aa_cols = sorted(set(c for c in df.columns if len(c) == 1 and c.isalpha()))
    aa_df = df[aa_cols].fillna(0)

    # Normalize per transcript (row-wise)
    aa_freq_df = aa_df.div(aa_df.sum(axis=1), axis=0).fillna(0)

    plt.figure(figsize=(16, 10), dpi=1200)
    sns.heatmap(aa_freq_df, cmap="YlGnBu", cbar_kws={"label": "Relative Frequency"})
    plt.title("Amino Acid Frequency per Transcript (Normalized)", fontsize=14, weight="bold")
    plt.xlabel("Amino Acid", fontsize=12)
    plt.ylabel("Transcript Index", fontsize=12)
    plt.tight_layout()
    if valid:
        plt.savefig("output/plot_aa_heatmap_valid_2.png", dpi=1200, bbox_inches="tight")
    else:
        plt.savefig("output/plot_aa_heatmap_2.png", dpi=1200, bbox_inches="tight")
    plt.show()


    # Clustering heatmap (normalized frequencies)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(aa_freq_df)  # <-- use normalized values
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    clustered = df.copy().sort_values("cluster")

    plt.figure(figsize=(16, 10), dpi=1200)
    sns.heatmap(
        clustered[aa_cols].div(clustered[aa_cols].sum(axis=1), axis=0).fillna(0),
        cmap="coolwarm",
        cbar_kws={"label": "Relative Frequency"}
    )
    plt.title("Clustered Amino Acid Frequency Heatmap (Normalized)", fontsize=14, weight="bold")
    plt.xlabel("Amino Acid", fontsize=12)
    plt.ylabel("Transcript Index (clustered)", fontsize=12)
    plt.tight_layout()
    if valid:
        plt.savefig("output/plot_clustered_aa_heatmap_valid_2.png", dpi=1200, bbox_inches="tight")
    else:
        plt.savefig("output/plot_clustered_aa_heatmap_2.png", dpi=1200, bbox_inches="tight")
    plt.show()
    
    
    # Global average amino acid composition
    avg_freq = aa_freq_df.mean(axis=0).sort_values(ascending=False)

    plt.figure(figsize=(10, 6), dpi=1200)
    sns.barplot(x=avg_freq.index, y=avg_freq.values, palette="viridis")
    plt.title("Average Amino Acid Composition Across All Transcripts", fontsize=14, weight="bold")
    plt.xlabel("Amino Acid", fontsize=12)
    plt.ylabel("Average Relative Frequency", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if valid:
        plt.savefig("output/plot_avg_aa_composition_valid.png", dpi=1200, bbox_inches="tight")
    else:
        plt.savefig("output/plot_avg_aa_composition.png", dpi=1200, bbox_inches="tight")
    plt.show()
    
    
    # GC content stats
    gc_stats = df.groupby("cluster")["GC_content"].describe()
    print("\nGC Content Summary by Cluster:")
    print(gc_stats)
    
    
    groups = [group["GC_content"].values for _, group in df.groupby("cluster")]
    stat, pval = f_oneway(*groups)
    anova_result = f"ANOVA result: F={stat:.2f}, p={pval:.4e}"
    print(f"\n{anova_result}")
    #print(f"\nANOVA result: F={stat:.2f}, p={pval:.4e}")
    
    # Define output file name based on 'valid' flag
    output_file = "output/gc_stats_valid.txt" if valid else "output/gc_stats.txt"

    # Save results to text file
    with open(output_file, "w") as f:
        f.write("GC Content Statistics by Cluster:\n")
        f.write(gc_stats.to_string())  # Convert DataFrame to string
        f.write("\n\n" + anova_result)  # Append ANOVA result
        
# =============================================================================

# Save Statistical Test Results as Report
def save_statistical_report(csv_file, output_file='output/statistical_report.txt'):
    df = pd.read_csv(csv_file, low_memory=False)
    
    # Example 1: GC content by group (if there's a 'group' column)
    report_lines = []
    if 'group' in df.columns:
        report_lines.append("=== GC CONTENT COMPARISON BY GROUP ===\n")
        groups = df.groupby('group')['gc_rich']
        for name, values in groups:
            report_lines.append(f"{name}: mean = {values.mean():.2f}, std = {values.std():.2f}")
        report_lines.append("ANOVA Test across groups:")
        report_lines.append(str(stats.f_oneway(*(group for _, group in groups))))

    # Example 2: Correlation of GC-rich vs amino acid count
    report_lines.append("\n=== CORRELATION ANALYSIS (GC-rich vs Total AAs) ===")
    df['total_aa'] = df[[aa for aa in df.columns if aa in genetic_code.values()]].sum(axis=1)
    correlation, p_value = stats.pearsonr(df['gc_rich'], df['total_aa'])
    report_lines.append(f"Pearson correlation: {correlation:.4f}, p-value: {p_value:.4e}")

    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Statistical report saved to '{output_file}'")

# =============================================================================

# Codon Frequency Bar Plot
def plot_codon_frequency(input_csv, output_file='output/codon_frequency_barplot.png'):
    df = pd.read_csv(input_csv, low_memory=False)
    codon_counts = Counter()

    # Re-process sequences for codon extraction
    for seq in df['sequence']:
        seq = normalize_sequence(seq)
        for i in range(0, len(seq) - 2, 3):
            codon = seq[i:i+3]
            if len(codon) == 3:
                codon_counts[codon] += 1

    codon_list, freqs = zip(*sorted(codon_counts.items()))
    
    #plt.figure(figsize=(18, 6))
    plt.figure(figsize=(20, 8), dpi=1200)
    #plt.bar(codon_list, freqs, color='skyblue')
    bars = plt.bar(codon_list, freqs, color=plt.cm.viridis(np.linspace(0, 1, len(codon_list))))
    #plt.title('Codon Frequency')
    plt.title('Codon Frequency', fontsize=16, weight="bold")
    #plt.xlabel('Codon')
    plt.xlabel('Codon', fontsize=14)
    #plt.ylabel('Count')
    plt.ylabel('Count', fontsize=14)
    #plt.xticks(rotation=90)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=12)
    
    # Gridlines for clarity
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=1200, bbox_inches="tight")
    plt.show()
    print(f"Codon frequency barplot saved to '{output_file}'")

# =============================================================================

def plot_codon_frequency_grouped(input_csv, output_file='output/codon_frequency_grouped.png'):
    df = pd.read_csv(input_csv, low_memory=False)
    codon_counts = Counter()

    # Re-process sequences for codon extraction
    for seq in df['sequence']:
        seq = normalize_sequence(seq).replace("T", "U")  # use RNA codons for table
        for i in range(0, len(seq) - 2, 3):
            codon = seq[i:i+3]
            if len(codon) == 3:
                codon_counts[codon] += 1

    # Build DataFrame with codon → amino acid mapping
    codon_data = []
    for codon, count in codon_counts.items():
        aa = genetic_code.get(codon, "Unknown")
        codon_data.append((codon, aa, count))

    df_codons = pd.DataFrame(codon_data, columns=["Codon", "AminoAcid", "Count"])

    # Sort by amino acid, then codon
    df_codons = df_codons.sort_values(by=["AminoAcid", "Codon"]).reset_index(drop=True)

    # Assign colors by amino acid group
    unique_aas = df_codons["AminoAcid"].unique()
    color_map = plt.cm.tab20(np.linspace(0, 1, len(unique_aas)))
    aa_colors = {aa: color_map[i] for i, aa in enumerate(unique_aas)}
    bar_colors = [aa_colors[aa] for aa in df_codons["AminoAcid"]]

    # Plot
    plt.figure(figsize=(22, 8), dpi=1200)
    plt.bar(df_codons["Codon"], df_codons["Count"], color=bar_colors)

    # Add vertical lines to separate amino acids
    boundaries = df_codons.groupby("AminoAcid").size().cumsum()
    for b in boundaries[:-1]:
        plt.axvline(x=b-0.5, color="gray", linestyle="--", alpha=0.6)

    # Labels
    plt.title("Codon Frequency Grouped by Amino Acid", fontsize=18, weight="bold")
    plt.xlabel("Codon", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(fontsize=12)

    # Add legend (Amino Acid → color)
    handles = [plt.Rectangle((0,0),1,1,color=aa_colors[aa]) for aa in unique_aas]
    plt.legend(handles, unique_aas, title="Amino Acid", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(output_file, dpi=1200, bbox_inches="tight")
    plt.show()

    print(f"Codon frequency grouped barplot saved to '{output_file}'")

# =============================================================================

# Unknown Codon Pie Chart
def plot_unknown_codon_pie(input_csv, output_file='output/unknown_codon_pie.png'):
    df = pd.read_csv(input_csv, low_memory=False)
    
    # Convert column to numeric safely
    df['unknown_codon_count'] = pd.to_numeric(df['unknown_codon_count'], errors='coerce').fillna(0).astype(int)
        
    total = len(df)
    unknowns = (df['unknown_codon_count'] > 0).sum()
    fully_annotated = total - unknowns
    
    # Debug print
    print(f"Total: {total}, Unknowns: {unknowns}, Fully Annotated: {total - unknowns}")
    
    if total == 0:
        print("No sequences found. Skipping pie chart.")
        return
    
    sizes = [unknowns, fully_annotated]
        
    labels = [
        f'With Unknown Codons: ({unknowns})',
        f'Fully Annotated: ({fully_annotated})'
    ]
    
    #colors = ['#FF9999', '#99CCFF']
    colors = ['#E41A1C', '#377EB8']
    explode = [0.1, 0]  # Explode only Unknown Codons slice
    
    def autopct_func(pct):
        return f'{pct:.3f}%' if pct > 0 else ''
    
    #plt.figure(figsize=(7, 7))
    plt.figure(figsize=(8, 8), dpi=1200)
    #plt.pie(sizes, labels=labels, autopct=autopct_func, colors=colors, startangle=140)
    plt.pie(sizes, labels=labels, autopct=autopct_func, colors=colors, explode=explode, startangle=140, textprops={'fontsize': 12, 'weight': 'bold'})
    #plt.title('Unknown Codon Proportion')
    plt.title('Unknown Codon Proportion', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=1200, bbox_inches="tight")
    plt.show()
    print(f"Unknown codon pie chart saved to '{output_file}'")


# ==========================================================================================
# ==========================================================================================


# =================================================================================================
#             POST-PROCESSING: Add Start/Stop Codon Validation & Filter Valid Sequences
# =================================================================================================


# Valid Sequences Pie Chart
def plot_valid_sequences_pie(all_seq_len, valid_seq_len, output_file='output/valid_sequences_pie.png'):
         
    total = all_seq_len
    valids = valid_seq_len
    invalids = total - valids
    
    
    # Debug print
    print(f"Total Sequences: {total}, Valid Sequences: {valids}, Invalid Sequences: {invalids}")
    
    
    if total == 0:
        print("No sequences found. Skipping pie chart.")
        return
    
    #sizes = [valids, total]
    sizes = [valids, invalids]
    
    """
    labels = [
        f'Valid Sequences: ({valids})',
        f'Total Sequences: ({total})'
    ]
    """
    
    labels = [
        f'Valid Sequences: ({valids})',
        f'Invalid Sequences: ({invalids})'
    ]
    
    #colors = ['#AEC6CF', '#AEC6CF']
    
    # ✅ Distinct colors: green for valid, gray for invalid
    colors = ['#4DAF4A', '#999999']
    explode = [0.1, 0]  # ✅ Emphasize valid sequences by pulling it out

    
    def autopct_func(pct):
        return f'{pct:.3f}%' if pct > 0 else ''
    
    #plt.figure(figsize=(7, 7))
    plt.figure(figsize=(8, 8), dpi=1200)
    #plt.pie(sizes, labels=labels, autopct=autopct_func, colors=colors, startangle=140)
    plt.pie(sizes, labels=labels, autopct=autopct_func, colors=colors, explode=explode, startangle=140, textprops={'fontsize': 12, 'weight': 'bold'})
    #plt.title('Valid Sequences Proportion')
    plt.title('Valid Sequences Proportion', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=1200, bbox_inches="tight")
    plt.show()
    print(f"Valid Sequences pie chart saved to '{output_file}'")



def postprocess_valid_sequences(input_csv, output_csv_valid):
    """
    Post-process the biological_analysis_report.csv to:
    1. Add columns 'x' (start codon check) and 'y' (stop codon check).
    2. Save only valid sequences based on criteria to a separate CSV file.
    """
    
    
    start_codons = ["AUG"]
    stop_codons = ["UAA", "UAG", "UGA"]

    df = pd.read_csv(input_csv, low_memory=False)

    # Add start/stop codon checks
    df["normalized_sequence"] = df["sequence"].apply(normalize_sequence)
    df["first_codon"] = df["normalized_sequence"].apply(lambda s: s[:3] if len(s) >= 3 else "")
    df["last_codon"] = df["normalized_sequence"].apply(lambda s: s[-3:] if len(s) >= 3 else "")
    df["start_codon_in_beginning"] = df["first_codon"].apply(lambda codon: "Yes" if codon in start_codons else "No")
    df["stop_codon_in_end"] = df["last_codon"].apply(lambda codon: "Yes" if codon in stop_codons else "No")

    # Filter valid sequences based on criteria
    valid_df = df[
        (df["start"] > 0) &
        (df["stop"] == 1) &
        (df["start_codon_in_beginning"] == "Yes") &
        (df["stop_codon_in_end"] == "Yes")
    ].copy()

    # Save both files
    df.drop(columns=["normalized_sequence", "first_codon", "last_codon"], inplace=True)
    valid_df.drop(columns=["normalized_sequence", "first_codon", "last_codon"], inplace=True)

    df.to_csv(input_csv, index=False)
    valid_df.to_csv(output_csv_valid, index=False)

    print(f"Total processed sequences: {len(df)}")
    print(f"Valid sequences saved: {len(valid_df)} ==> '{output_csv_valid}'")
    
    all_seq_len = len(df)
    valid_seq_len = len(valid_df)
    
    plot_valid_sequences_pie(all_seq_len, valid_seq_len)


# ==========================================================================================
# ==========================================================================================


# ==================================================
#             Extract Biological Results
# ==================================================


def extract_biological_results():
    # Load full and valid datasets
    df_all = pd.read_csv('output/biological_analysis_report.csv')
    df_valid = pd.read_csv('output/valid_sequences_analysis_report.csv')

    print("All sequences:", df_all.shape)
    print("Valid sequences:", df_valid.shape)



    # GC Content Distribution
    df_all['GC_content'] = df_all['sequence'].str.upper().apply(lambda x: 100 * (x.count('G') + x.count('C')) / len(x))
    df_valid['GC_content'] = df_valid['sequence'].str.upper().apply(lambda x: 100 * (x.count('G') + x.count('C')) / len(x))
    
    plt.figure(figsize=(10, 6), dpi=1200)
    #sns.histplot(df_all['GC_content'], color='red', label='All', kde=True)
    #sns.histplot(df_valid['GC_content'], color='green', label='Valid', kde=True)
    sns.histplot(df_all['GC_content'], color='red', label='All', kde=True, alpha=0.6, bins=40)
    sns.histplot(df_valid['GC_content'], color='green', label='Valid', kde=True, alpha=0.6, bins=40)
    #plt.title("GC Content Distribution: Valid vs All")
    plt.title("GC Content Distribution: Valid vs All", fontsize=16, weight="bold")
    #plt.xlabel("GC %")
    plt.xlabel("GC %", fontsize=13)
    plt.ylabel("Frequency", fontsize=13)
    plt.legend(fontsize=12)
    #plt.legend()
    plt.tight_layout()
    plt.savefig("output/compare_gc_content_valid_vs_all.png", dpi=1200, bbox_inches="tight")
    plt.show()


    # Compare Amino Acid Frequencies

    # Get amino acid columns
    aa_cols = [col for col in df_all.columns if col in {'A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V'}]

    # Calculate total frequency across all sequences
    aa_freq_all = df_all[aa_cols].sum()
    aa_freq_valid = df_valid[aa_cols].sum()

    # Plot side-by-side comparison
    freq_df = pd.DataFrame({
        'Amino Acid': aa_cols,
        'All Sequences': aa_freq_all.values,
        'Valid Sequences': aa_freq_valid.values
    })
    
    plt.figure(figsize=(12, 7), dpi=1200)
    #freq_df.set_index('Amino Acid').plot(kind='bar', figsize=(12,6), colormap='Set2')
    freq_df.set_index('Amino Acid').plot(kind='bar', figsize=(12,7), colormap='Set2', width=0.8, edgecolor="black")
    #plt.title("Amino Acid Usage: All vs Valid Sequences")
    plt.title("Amino Acid Usage: All vs Valid Sequences", fontsize=16, weight="bold")
    #plt.ylabel("Total Codon Count")
    plt.ylabel("Total Codon Count", fontsize=13)
    plt.xlabel("Amino Acid", fontsize=13)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.savefig("output/amino_acid_usage_valid_vs_all.png", dpi=1200, bbox_inches="tight")
    plt.show()



    # Unknown Codons Count

    # Convert column to numeric safely
    df_all['unknown_codon_count'] = pd.to_numeric(df_all['unknown_codon_count'], errors='coerce').fillna(0).astype(int)
    df_valid['unknown_codon_count'] = pd.to_numeric(df_valid['unknown_codon_count'], errors='coerce').fillna(0).astype(int)

    unknown_all = df_all['unknown_codon_count']
    unknown_valid = df_valid['unknown_codon_count']

    total_all = len(df_all)
    unknowns_all = (df_all['unknown_codon_count'] > 0).sum()


    if total_all == 0:
        print("No unknown codons within All sequences found.")
        

    print(f"Total unknown codons per ALL sequence: {unknowns_all}")
    print(f"Mean unknown codons per ALL sequence: {unknown_all.mean()}")


    total_valid = len(df_valid)
    unknowns_valid = (df_valid['unknown_codon_count'] > 0).sum()


    if total_valid == 0:
        print("No unknown codons within VALID sequences found.")
        

    print(f"Total unknown codons per VALID sequence: {unknowns_valid}")
    print(f"Mean unknown codons per VALID sequence: {unknown_valid.mean()}")
    
    
    # =========================================================================
    
    # Statistical Test for GC Content

    # Compare GC content
    gc_all = df_all['GC_content']
    gc_valid = df_valid['GC_content']

    # Perform t-test and Mann-Whitney U-test
    t_stat, t_pval = ttest_ind(gc_all, gc_valid, equal_var=False)
    u_stat, u_pval = mannwhitneyu(gc_all, gc_valid, alternative='two-sided')

    print(f"GC Content - t-test: t = {t_stat:.2f}, p = {t_pval:.4e}")
    print(f"GC Content - Mann-Whitney U: U = {u_stat:.2f}, p = {u_pval:.4e}")


    # Amino Acid Usage Difference Test
    observed_all = df_all[aa_cols].sum().values
    observed_valid = df_valid[aa_cols].sum().values


    # Normalize to percentages
    observed_all_pct = observed_all / observed_all.sum()
    observed_valid_pct = observed_valid / observed_valid.sum()

    # Use total counts from valid as the new expected values
    expected_valid_counts_scaled = observed_all_pct * observed_valid.sum()

    chi_stat, chi_pval = chisquare(f_obs=observed_valid, f_exp=expected_valid_counts_scaled)
    print(f"Amino Acid Usage - Chi-square Test (scaled): Chi-square = {chi_stat:.2f}, p = {chi_pval:.4e}")


    # Unknown Codon Counts — Statistical Comparison
    unknown_all = df_all['unknown_codon_count'].fillna(0).astype(int)
    unknown_valid = df_valid['unknown_codon_count'].fillna(0).astype(int)

    t_stat, t_pval = ttest_ind(unknown_all, unknown_valid, equal_var=False)
    u_stat, u_pval = mannwhitneyu(unknown_all, unknown_valid, alternative='two-sided')

    print(f"Unknown Codons - t-test: t = {t_stat:.2f}, p = {t_pval:.4e}")
    print(f"Unknown Codons - Mann-Whitney U: U = {u_stat:.2f}, p = {u_pval:.4e}")


    # Identify Top Unknown Codons (Mutation Candidates?)


    # Aggregate unknown codon strings
    all_unknown_codons = ','.join(df_all['unknown_codon'].dropna().astype(str)).split(',')
    valid_unknown_codons = ','.join(df_valid['unknown_codon'].dropna().astype(str)).split(',')

    # Count frequencies
    count_all = Counter(c for c in all_unknown_codons if c)
    count_valid = Counter(c for c in valid_unknown_codons if c)

    # Convert to DataFrame
    df_unknowns = pd.DataFrame({
        "Codon": list(set(count_all.keys()) | set(count_valid.keys())),
        "All_Sequences": [count_all.get(c, 0) for c in count_all.keys()],
        "Valid_Sequences": [count_valid.get(c, 0) for c in count_all.keys()]
    }).sort_values("All_Sequences", ascending=False).head(10)

    print("Top Unknown Codons:")
    print(df_unknowns)
    
    plt.figure(figsize=(10, 6), dpi=1200)
    #df_unknowns.set_index("Codon").plot(kind="bar", figsize=(8,5))
    df_unknowns.set_index("Codon").plot(kind="bar", figsize=(10,6), edgecolor="black", width=0.8)
    #plt.title("Top Unknown Codons in All vs Valid Sequences")
    plt.title("Top Unknown Codons in All vs Valid Sequences", fontsize=16, weight="bold")
    #plt.ylabel("Frequency")
    plt.ylabel("Frequency", fontsize=13)
    plt.xlabel("Codon", fontsize=13)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.savefig("output/plot_top_unknown_codons.png", dpi=1200, bbox_inches="tight")
    plt.show()
    
    # =========================================================================
    
    def save_stats_summary(gc_all, gc_valid, aa_cols, df_all, df_valid, unknown_all, unknown_valid):

        # ==============================
        # GC Content
        # ==============================
        t_stat, t_pval = ttest_ind(gc_all, gc_valid, equal_var=False)
        u_stat, u_pval = mannwhitneyu(gc_all, gc_valid, alternative='two-sided')

        # ==============================
        # Amino Acid Usage
        # ==============================
        observed_all = df_all[aa_cols].sum().values
        observed_valid = df_valid[aa_cols].sum().values

        observed_all_pct = observed_all / observed_all.sum()
        expected_valid_counts_scaled = observed_all_pct * observed_valid.sum()

        chi_stat, chi_pval = chisquare(f_obs=observed_valid, f_exp=expected_valid_counts_scaled)

        # ==============================
        # Unknown Codons
        # ==============================
        t_stat_u, t_pval_u = ttest_ind(unknown_all, unknown_valid, equal_var=False)
        u_stat_u, u_pval_u = mannwhitneyu(unknown_all, unknown_valid, alternative='two-sided')

        # ==============================
        # Build summary DataFrame
        # ==============================
        stats_summary = pd.DataFrame([
            ["GC Content (All vs Valid)", f"t-test", f"{t_stat:.2f}", f"{t_pval:.4e}"],
            ["GC Content (All vs Valid)", f"Mann-Whitney U", f"{u_stat:.2f}", f"{u_pval:.4e}"],
            ["Amino Acid Usage", f"Chi-square (scaled)", f"{chi_stat:.2f}", f"{chi_pval:.4e}"],
            ["Unknown Codons (All vs Valid)", f"t-test", f"{t_stat_u:.2f}", f"{t_pval_u:.4e}"],
            ["Unknown Codons (All vs Valid)", f"Mann-Whitney U", f"{u_stat_u:.2f}", f"{u_pval_u:.4e}"],
        ], columns=["Comparison", "Test", "Statistic", "p-value"])


        # Save as CSV
        stats_summary.to_csv("output/statistical_summary.csv", index=False)
        print("Statistical summary saved to 'output/statistical_summary.csv'")

        # ==============================
        # Save as Figure (PNG)
        # ==============================
        fig, ax = plt.subplots(figsize=(10, 3), dpi=1200)
        ax.axis("off")
        table = ax.table(
            cellText=stats_summary.values,
            colLabels=stats_summary.columns,
            cellLoc="center",
            loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.4)  # scale table size

        plt.title("Statistical Test Summary", fontsize=14, weight="bold")
        plt.savefig("output/statistical_summary.png", dpi=1200, bbox_inches="tight")
        plt.show()

        #return stats_summary
        
    # =========================================================================
    
        # ==============================
        # Save as Figure (PNG) with highlights
        # ==============================
        fig, ax = plt.subplots(figsize=(10, 3), dpi=1200)
        ax.axis("off")

        # Convert to list of lists for table
        cell_text = stats_summary.values.tolist()

        # Build table
        table = ax.table(
            cellText=cell_text,
            colLabels=stats_summary.columns,
            cellLoc="center",
            loc="center"
        )

        # Font settings
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.4)

        # Highlight significant p-values
        for i, pval_str in enumerate(stats_summary["p-value"]):
            try:
                pval = float(pval_str)
                if pval < 0.05:
                    cell = table[i+1, 3]  # +1 because row 0 = header
                    cell.set_text_props(color="red", weight="bold")
            except ValueError:
                continue

        plt.title("Statistical Test Summary", fontsize=14, weight="bold")
        plt.savefig("output/statistical_summary.png", dpi=1200, bbox_inches="tight")
        plt.show()

    save_stats_summary(gc_all, gc_valid, aa_cols, df_all, df_valid, unknown_all, unknown_valid)
 

# ==========================================================================================
# ==========================================================================================


# ==============================================================
#             Relative Synonymous Codon Usage (RSCU)
# ==============================================================


def compute_RSCU(valid_csv_path, output_dir = "output", valid=False):
    """
    Compute Relative Synonymous Codon Usage (RSCU) from a valid sequences CSV file.
    
    Args:
        valid_csv_path (str): Path to the CSV file with valid sequences.
        output_dir (str): Folder to save the output table and plot.
    
    Returns:
        pd.DataFrame: RSCU values as a DataFrame.
    """
    
    # 1. Load valid sequences
    df_valid = pd.read_csv(valid_csv_path)
    if 'sequence' not in df_valid.columns:
        raise ValueError("CSV file must contain a 'sequence' column.")
    
    
    # 2. Count codons
    codon_counts = Counter()
    for seq in df_valid['sequence']:
        seq = seq.upper().replace('T', 'U')  # Convert DNA to RNA
        codons = [seq[i:i+3] for i in range(0, len(seq)-2, 3)]
        for codon in codons:
            if len(codon) == 3:
                codon_counts[codon] += 1

    # 3. Calculate RSCU
    rscu_scores = {}
    for aa, codons in synonymous_codons.items():
        total = sum([codon_counts.get(codon, 0) for codon in codons])
        n = len(codons)
        for codon in codons:
            observed = codon_counts.get(codon, 0)
            expected = total / n if n > 0 else 1
            rscu = observed / expected if expected > 0 else 0
            rscu_scores[codon] = round(rscu, 3)

    # 4. Create DataFrame
    df_rscu = pd.DataFrame([
        {"Codon": codon, "RSCU": rscu_scores[codon], "Amino Acid": aa}
        for aa, codons in synonymous_codons.items()
        for codon in codons
    ])
    df_rscu = df_rscu.sort_values(by='Amino Acid')

    # 5. Save table
    os.makedirs(output_dir, exist_ok=True)
    if valid:
        rscu_table_path = os.path.join(output_dir, "rscu_table_valid.csv")
    else:
        rscu_table_path = os.path.join(output_dir, "rscu_table.csv")
    df_rscu.to_csv(rscu_table_path, index=False)

    # 6. Visualize RSCU
    #plt.figure(figsize=(14, 6))
    plt.figure(figsize=(16, 8), dpi=1200)
    sns.barplot(x='Codon', y='RSCU', hue='Amino Acid', data=df_rscu, dodge=False, palette='tab20')
    #plt.axhline(1.0, color='red', linestyle='--', label='Uniform usage')
    plt.axhline(1.0, color='red', linestyle='--', linewidth=1.2, label='Uniform usage')
    #plt.xticks(rotation=90)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("Codon", fontsize=12, weight="bold")
    plt.ylabel("RSCU", fontsize=12, weight="bold")
    #plt.title("Relative Synonymous Codon Usage (RSCU)")
    plt.title("Relative Synonymous Codon Usage (RSCU)", fontsize=14, weight="bold")
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9, title="Amino Acid")
    plt.tight_layout()
    
    # Save as high-resolution PNG and PDF (vector graphics)
    rscu_plot_path_png = os.path.join(output_dir, "rscu_plot_valid_1.png" if valid else "rscu_plot_1.png")
    rscu_plot_path_pdf = os.path.join(output_dir, "rscu_plot_valid_1.pdf" if valid else "rscu_plot_1.pdf")

    """
    if valid:
        rscu_plot_path = os.path.join(output_dir, "rscu_plot_valid_2.png")
    else:
        rscu_plot_path = os.path.join(output_dir, "rscu_plot_2.png")

    #plt.savefig(rscu_plot_path)
    #plt.show()
    
    print(f"✅ RSCU table saved to: {rscu_table_path}")
    print(f"✅ RSCU Plot saved to: {rscu_plot_path}")
    """
    
    plt.savefig(rscu_plot_path_png, dpi=1200, bbox_inches="tight")  # High-res PNG
    plt.savefig(rscu_plot_path_pdf, bbox_inches="tight")          # Publication-ready vector PDF
    plt.show()
    
    print(f"RSCU table saved to: {rscu_table_path}")
    print(f"RSCU plot saved to: {rscu_plot_path_png} and {rscu_plot_path_pdf}")
    
    
# =============================================================================

def compute_RSCU_grouping(valid_csv_path, output_dir="output", valid=False):
    """
    Compute Relative Synonymous Codon Usage (RSCU) and generate high-quality grouped plots.
    """

    # 1. Load sequences
    df_valid = pd.read_csv(valid_csv_path)
    if 'sequence' not in df_valid.columns:
        raise ValueError("CSV file must contain a 'sequence' column.")

    # 2. Count codons
    codon_counts = Counter()
    for seq in df_valid['sequence']:
        seq = seq.upper().replace('T', 'U')  # DNA → RNA
        codons = [seq[i:i+3] for i in range(0, len(seq)-2, 3)]
        for codon in codons:
            if len(codon) == 3:
                codon_counts[codon] += 1

    # 3. Calculate RSCU
    rscu_scores = {}
    for aa, codons in synonymous_codons.items():
        total = sum([codon_counts.get(codon, 0) for codon in codons])
        n = len(codons)
        for codon in codons:
            observed = codon_counts.get(codon, 0)
            expected = total / n if n > 0 else 1
            rscu = observed / expected if expected > 0 else 0
            rscu_scores[codon] = round(rscu, 3)

    # 4. Create DataFrame
    df_rscu = pd.DataFrame([
        {"Codon": codon, "RSCU": rscu_scores[codon], "Amino Acid": aa}
        for aa, codons in synonymous_codons.items()
        for codon in codons
    ])

    # Sort by amino acid, then codon
    df_rscu = df_rscu.sort_values(by=["Amino Acid", "Codon"])

    # 5. Save RSCU table
    os.makedirs(output_dir, exist_ok=True)
    rscu_table_path = os.path.join(output_dir, "rscu_table_valid_grouping.csv" if valid else "rscu_table_grouping.csv")
    df_rscu.to_csv(rscu_table_path, index=False)

    # 6. High-quality grouped visualization
    plt.figure(figsize=(18, 8), dpi=1200)
    sns.barplot(
        x="Codon", y="RSCU", hue="Amino Acid", data=df_rscu,
        dodge=False, palette="tab20"
    )
    plt.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="Uniform usage")

    # Improve readability
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(fontsize=10)
    plt.xlabel("Codons (grouped by Amino Acid)", fontsize=12, weight="bold")
    plt.ylabel("RSCU", fontsize=12, weight="bold")
    plt.title("Relative Synonymous Codon Usage (RSCU)", fontsize=14, weight="bold")

    # Move legend outside plot
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, title="Amino Acid")

    # Add vertical separators between amino acids
    unique_aa = df_rscu["Amino Acid"].unique()
    pos = 0
    ticks = []
    labels = []
    for aa in unique_aa:
        group_size = df_rscu[df_rscu["Amino Acid"] == aa].shape[0]
        pos += group_size
        plt.axvline(pos - 0.5, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
        ticks.append(pos - group_size / 2)
        labels.append(aa)

    # Optional: add secondary x-axis with amino acid grouping
    plt.xticks(range(len(df_rscu)), df_rscu["Codon"], rotation=90, fontsize=9)

    plt.tight_layout()

    # Save in high resolution
    rscu_plot_path_png = os.path.join(output_dir, "rscu_plot_valid_grouping.png" if valid else "rscu_plot_grouping.png")
    rscu_plot_path_pdf = os.path.join(output_dir, "rscu_plot_valid_grouping.pdf" if valid else "rscu_plot_grouping.pdf")

    plt.savefig(rscu_plot_path_png, dpi=1200, bbox_inches="tight")  # High-res PNG
    plt.savefig(rscu_plot_path_pdf, bbox_inches="tight")          # Vector PDF
    plt.show()

    print(f"RSCU table saved to: {rscu_table_path}")
    print(f"RSCU plot saved to: {rscu_plot_path_png} and {rscu_plot_path_pdf}")


# ==========================================================================================
# ==========================================================================================


# ========================================================
#             Effective Number of Codons (ENC)
# ========================================================


def compute_ENC(valid_csv_path, output_dir="output", seq_column='sequence', valid=False):
    """
    Computes Effective Number of Codons (ENC) per sequence and saves result CSV.

    Args:
        valid_csv_path (str): Path to CSV containing valid sequences with a 'sequence' column.
        output_dir (str): Directory to save the updated CSV with ENC scores.
        seq_column (str): Name of the column containing nucleotide sequences.

    Returns:
        pd.DataFrame: DataFrame with ENC scores added.
    """

    # Load sequences
    df = pd.read_csv(valid_csv_path)
    if seq_column not in df.columns:
        raise ValueError(f"CSV file must contain a '{seq_column}' column.")

    # Load codon table and synonymous codons
    codon_table = CodonTable.unambiguous_rna_by_name["Standard"].forward_table.copy()
    codon_table.update({'UAA': '*', 'UAG': '*', 'UGA': '*'})

    synonymous_codons = {}
    for codon, aa in codon_table.items():
        synonymous_codons.setdefault(aa, []).append(codon)

    def compute_ENC_for_sequence(seq, genetic_code=codon_table, syn_codons=synonymous_codons):
        seq = seq.upper().replace('T', 'U')
        codons = [seq[i:i+3] for i in range(0, len(seq)-2, 3) if len(seq[i:i+3]) == 3]
        codon_counts = Counter(codons)

        aa_codons = {}
        for codon, count in codon_counts.items():
            aa = genetic_code.get(codon, None)
            if aa is None:
                continue
            aa_codons.setdefault(aa, []).extend([codon] * count)

        fk_values = {2: [], 3: [], 4: [], 6: []}

        for aa, codon_list in aa_codons.items():
            syns = syn_codons.get(aa, [])
            k = len(syns)
            if k < 2 or k not in fk_values:
                continue

            codon_freqs = np.array([codon_list.count(c) for c in syns])
            n = codon_freqs.sum()
            if n == 0:
                continue
            p2 = np.sum((codon_freqs / n) ** 2)
            fk = (n * p2 - 1) / (n - 1) if n > 1 else 0
            fk_values[k].append(fk)

        def safe_mean(k): return np.mean(fk_values[k]) if fk_values[k] else np.nan
        F2, F3, F4, F6 = safe_mean(2), safe_mean(3), safe_mean(4), safe_mean(6)

        enc = 2
        enc += 9 / F2 if F2 and F2 != 0 else 0
        enc += 1 / F3 if F3 and F3 != 0 else 0
        enc += 5 / F4 if F4 and F4 != 0 else 0
        enc += 3 / F6 if F6 and F6 != 0 else 0

        return round(enc, 2)

    # Apply ENC computation to all sequences
    df['ENC'] = df[seq_column].apply(lambda x: compute_ENC_for_sequence(x))

    # Save result
    os.makedirs(output_dir, exist_ok=True)
    if valid:
        output_path = os.path.join(output_dir, "enc_scored_sequences_valid.csv")
    else:
        output_path = os.path.join(output_dir, "enc_scored_sequences.csv")
    df.to_csv(output_path, index=False)
    print(f"ENC scores saved to {output_path}")
    
    
    # ===============================
    # 🔹 Visualization (High Quality)
    # ===============================
    plt.figure(figsize=(10, 6), dpi=1200)
    sns.histplot(df['ENC'], bins=30, kde=True, color='steelblue', edgecolor='black', alpha=0.7)
    plt.title("Distribution of Effective Number of Codons (ENC)", fontsize=14, weight='bold')
    plt.xlabel("ENC", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Save high-res + vector
    enc_plot_path_png = os.path.join(output_dir, "enc_distribution_valid.png" if valid else "enc_distribution.png")
    enc_plot_path_pdf = os.path.join(output_dir, "enc_distribution_valid.pdf" if valid else "enc_distribution.pdf")

    plt.tight_layout()
    plt.savefig(enc_plot_path_png, dpi=1200, bbox_inches="tight")  # High-res PNG
    plt.savefig(enc_plot_path_pdf, bbox_inches="tight")          # Vector PDF
    plt.show()

    print(f"ENC distribution plot saved to: {enc_plot_path_png} and {enc_plot_path_pdf}")

    return df

# =============================================================================

def compute_ENC_GC3(valid_csv_path, output_dir="output", seq_column='sequence', valid=False):
    """
    Computes Effective Number of Codons (ENC) per sequence,
    saves results, and generates high-quality plots (distribution + ENC-plot).
    """

    # Load sequences
    df = pd.read_csv(valid_csv_path)
    if seq_column not in df.columns:
        raise ValueError(f"CSV file must contain a '{seq_column}' column.")

    # Codon table
    codon_table = CodonTable.unambiguous_rna_by_name["Standard"].forward_table.copy()
    codon_table.update({'UAA': '*', 'UAG': '*', 'UGA': '*'})

    synonymous_codons = {}
    for codon, aa in codon_table.items():
        synonymous_codons.setdefault(aa, []).append(codon)

    def compute_ENC_for_sequence(seq, genetic_code=codon_table, syn_codons=synonymous_codons):
        seq = seq.upper().replace('T', 'U')
        codons = [seq[i:i+3] for i in range(0, len(seq)-2, 3) if len(seq[i:i+3]) == 3]
        codon_counts = Counter(codons)

        aa_codons = {}
        for codon, count in codon_counts.items():
            aa = genetic_code.get(codon, None)
            if aa is None:
                continue
            aa_codons.setdefault(aa, []).extend([codon] * count)

        fk_values = {2: [], 3: [], 4: [], 6: []}
        for aa, codon_list in aa_codons.items():
            syns = syn_codons.get(aa, [])
            k = len(syns)
            if k < 2 or k not in fk_values:
                continue

            codon_freqs = np.array([codon_list.count(c) for c in syns])
            n = codon_freqs.sum()
            if n == 0:
                continue
            p2 = np.sum((codon_freqs / n) ** 2)
            fk = (n * p2 - 1) / (n - 1) if n > 1 else 0
            fk_values[k].append(fk)

        def safe_mean(k): return np.mean(fk_values[k]) if fk_values[k] else np.nan
        F2, F3, F4, F6 = safe_mean(2), safe_mean(3), safe_mean(4), safe_mean(6)

        enc = 2
        enc += 9 / F2 if F2 and F2 != 0 else 0
        enc += 1 / F3 if F3 and F3 != 0 else 0
        enc += 5 / F4 if F4 and F4 != 0 else 0
        enc += 3 / F6 if F6 and F6 != 0 else 0

        return round(enc, 2)

    # Apply ENC
    df['ENC'] = df[seq_column].apply(lambda x: compute_ENC_for_sequence(x))

    # Compute GC3
    def compute_GC3(seq):
        seq = seq.upper().replace('T', 'U')
        codons = [seq[i:i+3] for i in range(0, len(seq)-2, 3)]
        third_bases = [c[2] for c in codons if len(c) == 3]
        if len(third_bases) == 0:
            return np.nan
        gc3 = (third_bases.count('G') + third_bases.count('C')) / len(third_bases)
        return round(gc3, 3)

    df['GC3'] = df[seq_column].apply(compute_GC3)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "enc_scored_sequences_valid_GC3.csv" if valid else "enc_scored_sequences_GC3.csv")
    df.to_csv(output_path, index=False)
    print(f"ENC scores saved to {output_path}")

    # ===============================
    # 🔹 Plot 1: ENC Distribution
    # ===============================
    plt.figure(figsize=(10, 6), dpi=1200)
    sns.histplot(df['ENC'].dropna(), bins=30, kde=True,
                 color='steelblue', edgecolor='black', alpha=0.7)
    plt.title("Distribution of Effective Number of Codons (ENC)", fontsize=14, weight='bold')
    plt.xlabel("ENC", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    enc_plot_path_png = os.path.join(output_dir, "enc_distribution_valid_GC3.png" if valid else "enc_distribution_GC3.png")
    enc_plot_path_pdf = os.path.join(output_dir, "enc_distribution_valid_GC3.pdf" if valid else "enc_distribution_GC3.pdf")

    plt.tight_layout()
    plt.savefig(enc_plot_path_png, dpi=1200, bbox_inches="tight")
    plt.savefig(enc_plot_path_pdf, bbox_inches="tight")
    plt.show()

    print(f"ENC distribution plot saved to: {enc_plot_path_png} and {enc_plot_path_pdf}")

    # ===============================
    # 🔹 Plot 2: ENC vs GC3 (ENC-plot)
    # ===============================
    #plt.figure(figsize=(15, 12), dpi=1200)
    plt.figure(figsize=(20, 10), dpi=1200)
    plt.scatter(df['GC3'], df['ENC'], alpha=0.6, s=30, color='teal', edgecolor='k')
    plt.title("ENC-Plot: ENC vs GC3", fontsize=14, weight='bold')
    plt.xlabel("GC3 (G+C at 3rd codon position)", fontsize=12)
    plt.ylabel("ENC", fontsize=12)
    plt.ylim(20, 62)  # ENC theoretical range

    # Wright's theoretical curve
    x = np.linspace(0, 1, 200)
    y = 2 + x + (29 / (x**2 + (1 - x)**2))
    plt.plot(x, y, 'r--', label="Expected ENC under GC3-only bias")

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    enc_gc3_plot_path_png = os.path.join(output_dir, "enc_vs_gc3_valid.png" if valid else "enc_vs_gc3.png")
    enc_gc3_plot_path_pdf = os.path.join(output_dir, "enc_vs_gc3_valid.pdf" if valid else "enc_vs_gc3.pdf")

    plt.tight_layout()
    plt.savefig(enc_gc3_plot_path_png, dpi=1200, bbox_inches="tight")
    plt.savefig(enc_gc3_plot_path_pdf, bbox_inches="tight")
    plt.show()

    print(f"ENC vs GC3 plot saved to: {enc_gc3_plot_path_png} and {enc_gc3_plot_path_pdf}")

    return df


# =============================================================================


def plot_enc_statistics(enc_csv_path, output_dir="output", valid=False):
    # Load
    df = pd.read_csv(enc_csv_path)

    if 'ENC' not in df.columns:
        raise ValueError("ENC column not found in the CSV file.")

    # Step 1: Compute average ENC
    avg_enc = df['ENC'].mean()
    print(f"Average ENC score: {avg_enc:.2f}")

    # Create output dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Apply a clean style
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.2)

    # Step 2: Histogram of ENC
    #plt.figure(figsize=(8, 5))
    plt.figure(figsize=(10, 6), dpi=1200)
    #sns.histplot(df['ENC'], bins=30, kde=False, color='skyblue')
    sns.histplot(df['ENC'], bins=30, kde=False, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(avg_enc, color='red', linestyle='--', linewidth=1.5, label=f"Mean = {avg_enc:.2f}")
    #plt.title("Histogram of ENC Scores")
    plt.title("Histogram of ENC Scores", fontsize=16, weight='bold')
    plt.xlabel("ENC Score", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    #plt.xlabel("ENC Score")
    #plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    
    hist_png = os.path.join(output_dir, "hist_enc_scores_valid_2.png" if valid else "hist_enc_scores_2.png")
    hist_pdf = os.path.join(output_dir, "hist_enc_scores_valid_2.pdf" if valid else "hist_enc_scores_2.pdf")
    plt.savefig(hist_png, dpi=1200, bbox_inches="tight")   # High-res PNG
    plt.savefig(hist_pdf, bbox_inches="tight")           # Vector PDF
    
    
    
    if valid:
        plt.savefig(os.path.join(output_dir, "hist_enc_scores_valid.png"), dpi=1200, bbox_inches="tight")
    else:
        plt.savefig(os.path.join(output_dir, "hist_enc_scores.png"), dpi=1200, bbox_inches="tight")
    plt.show()

    # Step 3: KDE plot of ENC
    #plt.figure(figsize=(8, 5))
    plt.figure(figsize=(10, 6), dpi=1200)
    #sns.kdeplot(df['ENC'], fill=True, color='darkblue')
    sns.kdeplot(df['ENC'], fill=True, color='darkblue', linewidth=2)
    plt.axvline(avg_enc, color='red', linestyle='--', linewidth=1.5, label=f"Mean = {avg_enc:.2f}")
    #plt.title("KDE of ENC Scores")
    plt.title("KDE of ENC Scores", fontsize=16, weight='bold')
    plt.xlabel("ENC Score", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend()
    #plt.xlabel("ENC Score")
    plt.tight_layout()
    
    kde_png = os.path.join(output_dir, "kde_enc_scores_valid_2.png" if valid else "kde_enc_scores_2.png")
    kde_pdf = os.path.join(output_dir, "kde_enc_scores_valid_2.pdf" if valid else "kde_enc_scores_2.pdf")
    plt.savefig(kde_png, dpi=1200, bbox_inches="tight")   # High-res PNG
    plt.savefig(kde_pdf, bbox_inches="tight")           # Vector PDF
    
    if valid:
        plt.savefig(os.path.join(output_dir, "kde_enc_scores_valid.png"), dpi=1200, bbox_inches="tight")
    else:
        plt.savefig(os.path.join(output_dir, "kde_enc_scores.png"), dpi=1200, bbox_inches="tight")
    plt.show()
    
    print(f"Plots saved to: \n - {hist_png}\n - {hist_pdf}\n - {kde_png}\n - {kde_pdf}")

# =============================================================================

def plot_enc_statistics_combine(enc_csv_path, output_dir="output", valid=False):
    # Load
    df = pd.read_csv(enc_csv_path)

    if 'ENC' not in df.columns:
        raise ValueError("ENC column not found in the CSV file.")

    # Step 1: Compute average ENC
    avg_enc = df['ENC'].mean()
    print(f"Average ENC score: {avg_enc:.2f}")

    # Create output dir
    os.makedirs(output_dir, exist_ok=True)

    # Apply a clean style
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.2)

    # Step 2: Create subplot figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=1200)  # 2 subplots side by side

    # Histogram
    sns.histplot(df['ENC'], bins=30, kde=False, color='skyblue',
                 edgecolor='black', alpha=0.7, ax=axes[0])
    axes[0].axvline(avg_enc, color='red', linestyle='--', linewidth=1.5, label=f"Mean = {avg_enc:.2f}")
    axes[0].set_title("Histogram of ENC Scores", fontsize=16, weight='bold')
    axes[0].set_xlabel("ENC Score", fontsize=14)
    axes[0].set_ylabel("Frequency", fontsize=14)
    axes[0].legend()

    # KDE plot
    sns.kdeplot(df['ENC'], fill=True, color='darkblue', linewidth=2, ax=axes[1])
    axes[1].axvline(avg_enc, color='red', linestyle='--', linewidth=1.5, label=f"Mean = {avg_enc:.2f}")
    axes[1].set_title("KDE of ENC Scores", fontsize=16, weight='bold')
    axes[1].set_xlabel("ENC Score", fontsize=14)
    axes[1].set_ylabel("Density", fontsize=14)
    axes[1].legend()

    # Tight layout
    plt.tight_layout()

    # Save in high-res PNG and PDF
    combined_png = os.path.join(output_dir, "enc_statistics_valid_combine.png" if valid else "enc_statistics_combine.png")
    combined_pdf = os.path.join(output_dir, "enc_statistics_valid_combine.pdf" if valid else "enc_statistics_combine.pdf")
    plt.savefig(combined_png, dpi=1200, bbox_inches="tight")   # High-res PNG
    plt.savefig(combined_pdf, bbox_inches="tight")           # Vector PDF
    plt.show()

    print(f"Combined figure saved to:\n - {combined_png}\n - {combined_pdf}")


# ==========================================================================================
# ==========================================================================================


# ==============================================================
#             Visual Comparison between RSCU and ENC
# ==============================================================


def plot_rscu_vs_enc(rscu_csv_paths, enc_csv_paths, output_path="output/rscu_enc_comparison.png"):
    """
    Plot RSCU heatmap and ENC boxplot side by side across datasets.

    Args:
        rscu_csv_paths (list): List of RSCU CSV files (each with Codon + RSCU column).
        enc_csv_paths (list): List of ENC CSV files (each with ENC column).
        output_path (str): Path to save the combined figure.
    """
    
    # Normalize to list
    if isinstance(rscu_csv_paths, str):
        rscu_csv_paths = [rscu_csv_paths]
    if isinstance(enc_csv_paths, str):
        enc_csv_paths = [enc_csv_paths]
        
    # Set seaborn style for publication
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.2)
    
    fig, axes = plt.subplots(1, 2, figsize=(24, 12), dpi=1200, gridspec_kw={'width_ratios': [1.6, 1]})

    # === RSCU heatmap (left panel) ===
    rscu_frames = []
    for i, path in enumerate(rscu_csv_paths):
        df = pd.read_csv(path)
        df = df[['Codon', 'RSCU']].copy()
        df = df.set_index('Codon')
        df.columns = ["Dataset"]
        rscu_frames.append(df)
    
    rscu_combined = pd.concat(rscu_frames, axis=1)
    #sns.heatmap(rscu_combined, cmap='YlGnBu', annot=True, fmt=".2f", ax=axes[0])
    sns.heatmap(rscu_combined, cmap="YlGnBu", annot=True, fmt=".2f", annot_kws={"size": 9}, cbar_kws={'label': 'RSCU'}, ax=axes[0])
    #axes[0].set_title("RSCU Heatmap")
    #axes[0].set_ylabel("Codon")
    axes[0].set_title("RSCU Heatmap", fontsize=18, weight="bold")
    axes[0].set_ylabel("Codon", fontsize=14)

    # === ENC distribution (right panel) ===
    enc_frames = []
    for i, path in enumerate(enc_csv_paths):
        df = pd.read_csv(path)
        df = df[['ENC']].copy()
        # df['Dataset'] = f"Dataset {i+1}"  # Assign label
        df['Dataset'] = "Dataset"  # Assign label
        enc_frames.append(df)

    
    enc_df = pd.concat(enc_frames)
    sns.boxplot(data=enc_df, x='Dataset', y='ENC', palette='Set2', ax=axes[1])
    #sns.stripplot(data=enc_df, x='Dataset', y='ENC', color='black', alpha=0.3, jitter=0.2, ax=axes[1])
    #sns.stripplot(data=enc_df, y='ENC', color='black', alpha=0.3, jitter=0.2, ax=axes[1])
    sns.stripplot(data=enc_df, x='Dataset', y='ENC', color='black', alpha=0.3, jitter=0.25, ax=axes[1])
    #axes[1].set_title("ENC Distribution")
    axes[1].set_title("ENC Distribution", fontsize=18, weight="bold")
    axes[1].set_xlabel("", fontsize=14)
    axes[1].set_ylabel("ENC", fontsize=14)
    
    # === Save ===
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=1200, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.show()

    print(f"Saved combined RSCU-ENC comparison to:\n - {output_path}\n - {output_path.replace('.png', '.pdf')}")

# =============================================================================

# Example codon ↔ amino acid mapping (RNA codons, U instead of T)
synonymous_codons = {
    'Phe': ['UUU', 'UUC'],
    'Leu': ['UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG'],
    'Ile': ['AUU', 'AUC', 'AUA'],
    'Met': ['AUG'],
    'Val': ['GUU', 'GUC', 'GUA', 'GUG'],
    'Ser': ['UCU', 'UCC', 'UCA', 'UCG', 'AGU', 'AGC'],
    'Pro': ['CCU', 'CCC', 'CCA', 'CCG'],
    'Thr': ['ACU', 'ACC', 'ACA', 'ACG'],
    'Ala': ['GCU', 'GCC', 'GCA', 'GCG'],
    'Tyr': ['UAU', 'UAC'],
    'His': ['CAU', 'CAC'],
    'Gln': ['CAA', 'CAG'],
    'Asn': ['AAU', 'AAC'],
    'Lys': ['AAA', 'AAG'],
    'Asp': ['GAU', 'GAC'],
    'Glu': ['GAA', 'GAG'],
    'Cys': ['UGU', 'UGC'],
    'Trp': ['UGG'],
    'Arg': ['CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'Gly': ['GGU', 'GGC', 'GGA', 'GGG'],
    'STOP': ['UAA', 'UAG', 'UGA']
}

def plot_rscu_vs_enc_amino_acid(rscu_csv_paths, enc_csv_paths, output_path="output/rscu_enc_comparison_amino_acid.png"):
    """
    Plot RSCU heatmap (grouped by amino acid) and ENC boxplot side by side across datasets.
    """

    # Normalize to list
    if isinstance(rscu_csv_paths, str):
        rscu_csv_paths = [rscu_csv_paths]
    if isinstance(enc_csv_paths, str):
        enc_csv_paths = [enc_csv_paths]

    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.1)
    
    # Build ordered codon list & corresponding AA labels
    ordered_codons = []
    aa_labels = []
    for aa, codons in synonymous_codons.items():
        ordered_codons.extend(codons)
        aa_labels.extend([aa] * len(codons))
    n_rows = len(ordered_codons)  # expected rows (usually 64)
    
    # === Build combined RSCU dataframe (rows = codons, cols = datasets) ===
    rscu_frames = []
    for i, path in enumerate(rscu_csv_paths):
        df_tmp = pd.read_csv(path, dtype=str)  # read as str to avoid surprises
        if 'Codon' not in df_tmp.columns or 'RSCU' not in df_tmp.columns:
            raise ValueError(f"RSCU CSV {path} must contain 'Codon' and 'RSCU' columns.")
        # normalize codon names to RNA upper-case (U instead of T)
        df_tmp['Codon'] = df_tmp['Codon'].astype(str).str.upper().str.replace('T', 'U')
        df_tmp['RSCU'] = pd.to_numeric(df_tmp['RSCU'], errors='coerce')
        df_tmp = df_tmp.set_index('Codon')[['RSCU']].rename(columns={'RSCU': "Dataset"})
        rscu_frames.append(df_tmp)
    
    rscu_combined = pd.concat(rscu_frames, axis=1)
    
    # Reindex to full ordered codon list (missing codons will appear as NaN)
    rscu_combined = rscu_combined.reindex(ordered_codons)
    
    # === Build ENC dataframe for boxplot ===
    enc_frames = []
    for i, path in enumerate(enc_csv_paths):
        enc_df = pd.read_csv(path)
        if 'ENC' not in enc_df.columns:
            raise ValueError(f"ENC CSV {path} must contain an 'ENC' column.")
        enc_df = enc_df[['ENC']].copy()
        enc_df['Dataset'] = "Dataset"
        enc_frames.append(enc_df)
    enc_df = pd.concat(enc_frames, ignore_index=True)

    # === Plotting ===
    #fig, axes = plt.subplots(1, 2, figsize=(26, 14), dpi=1200, gridspec_kw={'width_ratios': [1.8, 1]})
    fig, axes = plt.subplots(1, 2, figsize=(26, 14), gridspec_kw={'width_ratios': [1.8, 1]})
    fig.subplots_adjust(wspace=0.25)
    
    # Heatmap mask for NaNs (so we don't annotate NaNs)
    mask = rscu_combined.isna()
    
    # Left: RSCU heatmap
    sns.heatmap(
        rscu_combined,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        mask=mask,
        annot_kws={"size": 8},
        cbar_kws={'label': 'RSCU'},
        linewidths=0.3,
        linecolor='gray',
        ax=axes[0]
    )

    axes[0].set_title("RSCU Heatmap (Grouped by Amino Acid)", fontsize=18, weight="bold")
    axes[0].set_ylabel("Codon", fontsize=14)
    
    # Explicitly set tick positions and labels to match rows
    n_rows = rscu_combined.shape[0]
    # tick positions: center of each cell (matplotlib imshow/pcolormesh places ticks at integer + 0.5)
    tick_positions = np.arange(n_rows) + 0.5
    tick_labels = [f"{codon} ({aa})" for codon, aa in zip(ordered_codons, aa_labels)]

    axes[0].set_yticks(tick_positions)
    axes[0].set_yticklabels(tick_labels, fontsize=10, rotation=0)
    # If the order appears inverted, uncomment the next line:
    # axes[0].invert_yaxis()

    # draw horizontal separators between amino acid groups
    cum_sizes = np.cumsum([len(synonymous_codons[aa]) for aa in synonymous_codons.keys()])
    for boundary in cum_sizes[:-1]:
        # boundary line between rows boundary-1 and boundary
        axes[0].axhline(boundary, color='black', linestyle='--', linewidth=0.6, alpha=0.7)

    # Optionally add amino acid labels on the right side (middle of each block)
    x_right = rscu_combined.shape[1] + 0.2  # a bit to the right of the heatmap
    start = 0
    for aa in synonymous_codons.keys():
        group_len = len(synonymous_codons[aa])
        center = start + group_len / 2
        axes[0].text(x_right, center, aa, va='center', fontsize=9, weight='bold')
        start += group_len

    # Right: ENC boxplot + jittered points
    sns.boxplot(data=enc_df, x='Dataset', y='ENC', palette='Set2', ax=axes[1])
    sns.stripplot(data=enc_df, x='Dataset', y='ENC', color='black', alpha=0.35, jitter=0.25, ax=axes[1])
    axes[1].set_title("ENC Distribution", fontsize=18, weight="bold")
    axes[1].set_xlabel("", fontsize=14)
    axes[1].set_ylabel("ENC", fontsize=14)

    # Save high-res PNG and vector PDF
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=1200, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Saved grouped RSCU-ENC comparison to:\n - {output_path}\n - {output_path.replace('.png', '.pdf')}")


# ==========================================================================================
# ==========================================================================================


# ===================================================================
#             Codon Usage Entropy per Sequence
# ===================================================================


def calculate_entropy_per_sequence(valid_csv_path, output_dir = "output", valid=False):
    """
    Calculates codon usage entropy per sequence and saves result & plot.
    """
    
    # 1. Load valid sequences
    df = pd.read_csv(valid_csv_path)
    if 'sequence' not in df.columns:
        raise ValueError("CSV file must contain a 'sequence' column.")
    
    # 2. Define genetic code
    genetic_code = CodonTable.unambiguous_rna_by_name["Standard"].forward_table.copy()
    genetic_code.update({'UAA': '*', 'UAG': '*', 'UGA': '*'})  # Stop codons
    
    # 3. Codon entropy per sequence    
    def compute_entropy(seq):
        # Extract codons
        seq = seq.upper().replace('T', 'U')  # Convert DNA to RNA
        codons = [seq[i:i+3] for i in range(0, len(seq)-2, 3) if len(seq[i:i+3]) == 3]
        aa_to_codons = {}
        
        # Group codons by their amino acid
        for codon in codons:
            aa = genetic_code.get(codon, 'X')  # 'X' for unknown
            aa_to_codons.setdefault(aa, []).append(codon)
            
        entropies = []
        for codon_list in aa_to_codons.values():
            counts = np.array(list(pd.Series(codon_list).value_counts()))
            probs = counts / counts.sum()
            ent = -np.sum(probs * np.log2(probs))
            entropies.append(ent)
        return np.mean(entropies) if entropies else 0.0

    df['codon_entropy'] = df['sequence'].apply(compute_entropy)
    
    # 4. Save updated CSV
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "rscu_entropy_valid.csv" if valid else "rscu_entropy.csv")
    df.to_csv(output_file, index=False)
    print(f"Saved codon entropy to {output_file}")
    
    # 5. Plot entropy distribution
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.3)  # Bigger fonts
    
    #plt.figure(figsize=(8, 5))
    plt.figure(figsize=(15, 10), dpi=1200)
    #sns.histplot(df['codon_entropy'], bins=40, kde=True, color='skyblue')  # 'label' = valid/invalid
    sns.histplot(df['codon_entropy'], bins=40, kde=True, color='skyblue', edgecolor="black", linewidth=1.2)
    
    #plt.title("Codon Usage Entropy Distribution per Sequence")
    plt.title("Codon Usage Entropy Distribution per Sequence", fontsize=18, weight="bold")
    #plt.xlabel("Entropy Score")
    plt.xlabel("Entropy Score", fontsize=14)
    #plt.ylabel("Frequency")
    plt.ylabel("Frequency", fontsize=14)
    
    entropy_plot_png = os.path.join(output_dir, "codon_entropy_distribution_valid.png" if valid else "codon_entropy_distribution.png")
    entropy_plot_pdf = entropy_plot_png.replace(".png", ".pdf")
    #entropy_plot = os.path.join(output_dir, "codon_entropy_distribution_valid.png" if valid else "codon_entropy_distribution.png")    
    plt.tight_layout()
    # 🔹 Save in high-resolution PNG + vector PDF
    plt.savefig(entropy_plot_png, dpi=1200, bbox_inches="tight")  
    plt.savefig(entropy_plot_pdf, bbox_inches="tight")  
    
    #plt.savefig(entropy_plot)
    plt.show()
    print(f"Entropy plots saved to:\n - {entropy_plot_png}\n - {entropy_plot_pdf}")
        
    return df

# =============================================================================

def calculate_entropy_per_sequence_violin(valid_csv_path, output_dir="output", valid=False):
    """
    Calculates codon usage entropy per sequence and saves result & high-quality plots.
    """

    # 1. Load valid sequences
    df = pd.read_csv(valid_csv_path)
    if 'sequence' not in df.columns:
        raise ValueError("CSV file must contain a 'sequence' column.")

    # 2. Define genetic code
    genetic_code = CodonTable.unambiguous_rna_by_name["Standard"].forward_table.copy()
    genetic_code.update({'UAA': '*', 'UAG': '*', 'UGA': '*'})  # Stop codons

    # 3. Codon entropy per sequence
    def compute_entropy(seq):
        seq = seq.upper().replace('T', 'U')
        codons = [seq[i:i+3] for i in range(0, len(seq)-2, 3) if len(seq[i:i+3]) == 3]
        aa_to_codons = {}
        for codon in codons:
            aa = genetic_code.get(codon, 'X')
            aa_to_codons.setdefault(aa, []).append(codon)

        entropies = []
        for codon_list in aa_to_codons.values():
            counts = np.array(list(pd.Series(codon_list).value_counts()))
            probs = counts / counts.sum()
            ent = -np.sum(probs * np.log2(probs))
            entropies.append(ent)
        return np.mean(entropies) if entropies else 0.0

    df['codon_entropy'] = df['sequence'].apply(compute_entropy)

    # 4. Save updated CSV
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "rscu_entropy_violin_valid.csv" if valid else "rscu_entropy_violin.csv")
    df.to_csv(output_file, index=False)
    print(f"Saved codon entropy to {output_file}")

    # Set style
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.3)

    # === Plot 1: Histogram + KDE ===
    plt.figure(figsize=(8, 5), dpi=1200)
    sns.histplot(df['codon_entropy'], bins=40, kde=True, color='skyblue',
                 edgecolor="black", linewidth=1.2)

    plt.title("Codon Usage Entropy Distribution per Sequence", weight="bold") #, fontsize=18, weight="bold")
    plt.xlabel("Entropy Score") #, fontsize=14)
    plt.ylabel("Frequency") #, fontsize=14)

    entropy_hist_png = os.path.join(output_dir, "codon_entropy_hist_violin_valid.png" if valid else "codon_entropy_hist_violin.png")
    entropy_hist_pdf = entropy_hist_png.replace(".png", ".pdf")

    plt.tight_layout()
    plt.savefig(entropy_hist_png, dpi=1200, bbox_inches="tight")
    plt.savefig(entropy_hist_pdf, bbox_inches="tight")
    plt.show()
    plt.close()

    # === Plot 2: Violin + Boxplot ===
    plt.figure(figsize=(8, 5), dpi=1200)
    sns.violinplot(y=df['codon_entropy'], inner=None, color="skyblue", cut=0)
    sns.boxplot(y=df['codon_entropy'], width=0.2, boxprops={'zorder': 2}, showcaps=True,
                showfliers=True, whiskerprops={'linewidth': 1.5})

    plt.title("Codon Usage Entropy Spread", fontsize=18, weight="bold")
    plt.ylabel("Entropy Score", fontsize=14)

    entropy_violin_png = os.path.join(output_dir, "codon_entropy_violin_valid.png" if valid else "codon_entropy_violin.png")
    entropy_violin_pdf = entropy_violin_png.replace(".png", ".pdf")

    plt.tight_layout()
    plt.savefig(entropy_violin_png, dpi=1200, bbox_inches="tight")
    plt.savefig(entropy_violin_pdf, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Entropy plots saved to:\n"
          f" - Histogram: {entropy_hist_png}, {entropy_hist_pdf}\n"
          f" - Violin+Box: {entropy_violin_png}, {entropy_violin_pdf}")

    return df


# ==========================================================================================
# ==========================================================================================


# ===============================================
#             Generate Analysis Plots
# ===============================================


# === 1. Codon Usage Heatmap ===
def compute_codon_counts(sequences):
    """Compute codon counts from DNA sequences."""
    codon_counter = Counter()
    for seq in sequences:
        # Convert DNA to RNA-style and split into codons
        seq_rna = seq.replace('T', 'U').upper()
        codons = [seq_rna[i:i+3] for i in range(0, len(seq_rna), 3)]
        # Filter incomplete codons and count
        codon_counter.update(c for c in codons if len(c) == 3)
    return codon_counter

# === 2. Codon Usage Heatmap (1st vs 2nd Base) ===
def plot_codon_usage_heatmap(df, label="All", filename="codon_usage_heatmap_all.png"):
    codon_counts = compute_codon_counts(df['sequence'])
    codon_freq = pd.Series(codon_counts).reindex(STANDARD_CODONS).fillna(0)

    # Reshape for heatmap: 4x4x4 cube collapsed to 2D (first base vs second base)
    heatmap_data = pd.DataFrame(
        np.zeros((4, 4)), index=BASES, columns=BASES
    )

    for codon, count in codon_freq.items():
        if len(codon) == 3:
            b1, b2 = codon[0], codon[1]
            heatmap_data.loc[b1, b2] += count
    
    # High-quality plotting
    plt.figure(figsize=(10, 8), dpi=1200)   # larger size & high resolution
    sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="coolwarm", cbar_kws={'shrink': 0.8, 'label': 'Codon Count'})
    
    plt.title(f"Codon Usage Heatmap ({label}) [1st vs 2nd Base]", fontsize=14, weight="bold")
    plt.ylabel("1st Base", fontsize=12)
    plt.xlabel("2nd Base", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    # Save at high resolution
    plt.savefig(os.path.join(SAVE_DIR, filename), dpi=1200, bbox_inches="tight")
    plt.show()

# === 3. Codon Usage Heatmap (1st+2nd vs 3rd Base) ===
def plot_full_codon_usage_heatmap(df, label="All", filename="codon_usage_heatmap_full.png"):
    codon_counts = compute_codon_counts(df['sequence'])
    codon_freq = pd.Series(codon_counts).reindex(STANDARD_CODONS).fillna(0)

    # Build dataframe: rows = first+second base, cols = third base
    row_labels = [b1 + b2 for b1 in BASES for b2 in BASES]  # 16 rows
    heatmap_data = pd.DataFrame(
        np.zeros((16, 4)),
        index=row_labels,
        columns=BASES
    )

    for codon, count in codon_freq.items():
        if len(codon) == 3:
            row = codon[0] + codon[1]
            col = codon[2]
            heatmap_data.loc[row, col] += count

    # Plot heatmap
    plt.figure(figsize=(12, 10), dpi=1200)
    sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="coolwarm", cbar_kws={'shrink': 0.8, 'label': 'Codon Count'})

    plt.title(f"Full Codon Usage Heatmap ({label}) [1st+2nd vs 3rd Base]", fontsize=14, weight="bold")
    plt.ylabel("1st + 2nd Base", fontsize=12)
    plt.xlabel("3rd Base", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=9, rotation=0)  # keep row labels horizontal for readability
    plt.tight_layout()

    # Save high-res
    plt.savefig(os.path.join(SAVE_DIR, filename), dpi=1200, bbox_inches="tight")
    plt.show()


# === 4. Codon Usage Heatmap (All Bases) ===
def plot_codon_usage_64_heatmap(df, label="All", filename="codon_usage_heatmap_64.png"):
    codon_counts = compute_codon_counts(df['sequence'])
    codon_freq = pd.Series(codon_counts).reindex(STANDARD_CODONS).fillna(0)

    # Reshape into 4x4x4 cube → flatten into 8x8 for nice visualization
    # (1st base on Y, 2nd base on X-blocks, 3rd base inside block)
    heatmap_data = pd.DataFrame(
        np.zeros((8, 8)),
        index=[f"{i}" for i in range(8)],
        columns=[f"{j}" for j in range(8)]
    )
    codon_labels = pd.DataFrame("", index=heatmap_data.index, columns=heatmap_data.columns)

    # Mapping codons into 8x8 grid
    for i, b1 in enumerate(BASES):
        for j, b2 in enumerate(BASES):
            for k, b3 in enumerate(BASES):
                codon = b1 + b2 + b3
                row = i * 2 + (k // 2)   # group 3rd base into 2x2 subgrids
                col = j * 2 + (k % 2)
                heatmap_data.iloc[row, col] = codon_freq[codon]
                codon_labels.iloc[row, col] = codon

    # Plot heatmap
    plt.figure(figsize=(12, 10), dpi=1200)
    ax = sns.heatmap(
        heatmap_data,
        annot=codon_labels,    # annotate with codon labels
        fmt="",                # no formatting (text only)
        cmap="coolwarm",
        cbar_kws={'shrink': 0.8, 'label': 'Codon Count'},
        linewidths=0.5,
        linecolor="black",
        annot_kws={"size": 7, "color": "black", "weight": "bold"}
    )

    # Overlay counts on top of codon labels
    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            ax.text(
                j+0.5, i+0.7,        # slight offset below codon
                f"{int(heatmap_data.iloc[i, j])}",
                ha="center", va="center", fontsize=6  #, color="blue"
            )

    plt.title(f"Full Codon Usage Heatmap ({label})", fontsize=14, weight="bold")
    plt.axis("off")  # hide artificial 8x8 axes
    plt.tight_layout()

    # Save high-res
    plt.savefig(os.path.join(SAVE_DIR, filename), dpi=1200, bbox_inches="tight")
    plt.show()



# === 5. Top 10 Codons Table ===
def get_top_codon_table(df, top_n=10):
    codon_counts = compute_codon_counts(df['sequence'])
    return pd.Series(codon_counts).sort_values(ascending=False).head(top_n)


# === 6. Amino Acid Usage Comparison Plot ===
def amino_acid_usage_comparison_plot(df_all, df_valid):
    aa_cols = [col for col in df_all.columns if col in {
        'A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V'}]

    aa_all = df_all[aa_cols].sum()
    aa_valid = df_valid[aa_cols].sum()

    aa_df = pd.DataFrame({
        'Amino Acid': aa_cols,
        'All': aa_all.values,
        'Valid': aa_valid.values
    }).set_index("Amino Acid")

    aa_df.plot(kind='bar', figsize=(8,5), colormap="Paired")
    plt.title("Amino Acid Usage: All vs Valid Sequences")
    plt.ylabel("Total Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "amino_acid_usage_barplot.png"), dpi=1200, bbox_inches="tight")
    plt.show()


# === 7. Unknown Codons Plot ===
def plot_top_unknown_codons(df_all, df_valid, top_n=10):
    all_unknowns = ','.join(df_all['unknown_codon'].dropna().astype(str)).split(',')
    valid_unknowns = ','.join(df_valid['unknown_codon'].dropna().astype(str)).split(',')

    count_all = Counter(c for c in all_unknowns if c)
    count_valid = Counter(c for c in valid_unknowns if c)

    top_codons = list(dict(count_all.most_common(top_n)).keys())

    df = pd.DataFrame({
        'Codon': top_codons,
        'All': [count_all.get(c, 0) for c in top_codons],
        'Valid': [count_valid.get(c, 0) for c in top_codons],
    }).set_index("Codon")

    df.plot(kind='bar', figsize=(20,10), colormap="tab10")
    plt.title("Top Unknown Codons (All vs Valid Sequences)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "top_unknown_codons.png"), dpi=1200, bbox_inches="tight")
    plt.show()


# ==========================================================================================
# ==========================================================================================


# =====================================================
#             Generate Amino Acid Sequences
# =====================================================


def extract_protein_sequences():
    # Standard codon table (RNA codons to amino acids)
    codon_table = {
        'UUU':'F', 'UUC':'F', 'UUA':'L', 'UUG':'L',
        'UCU':'S', 'UCC':'S', 'UCA':'S', 'UCG':'S',
        'UAU':'Y', 'UAC':'Y', 'UAA':'*', 'UAG':'*',
        'UGU':'C', 'UGC':'C', 'UGA':'*', 'UGG':'W',
        'CUU':'L', 'CUC':'L', 'CUA':'L', 'CUG':'L',
        'CCU':'P', 'CCC':'P', 'CCA':'P', 'CCG':'P',
        'CAU':'H', 'CAC':'H', 'CAA':'Q', 'CAG':'Q',
        'CGU':'R', 'CGC':'R', 'CGA':'R', 'CGG':'R',
        'AUU':'I', 'AUC':'I', 'AUA':'I', 'AUG':'M',
        'ACU':'T', 'ACC':'T', 'ACA':'T', 'ACG':'T',
        'AAU':'N', 'AAC':'N', 'AAA':'K', 'AAG':'K',
        'AGU':'S', 'AGC':'S', 'AGA':'R', 'AGG':'R',
        'GUU':'V', 'GUC':'V', 'GUA':'V', 'GUG':'V',
        'GCU':'A', 'GCC':'A', 'GCA':'A', 'GCG':'A',
        'GAU':'D', 'GAC':'D', 'GAA':'E', 'GAG':'E',
        'GGU':'G', 'GGC':'G', 'GGA':'G', 'GGG':'G'
    }

    # Load valid sequences
    df = pd.read_csv("output/valid_sequences_analysis_report.csv")

    # Ensure U instead of T for RNA
    def rna_to_protein(rna_seq):
        rna_seq = rna_seq.upper().replace("T", "U")
        protein = ""
        for i in range(0, len(rna_seq) - 2, 3):
            codon = rna_seq[i:i+3]
            aa = codon_table.get(codon, 'X')  # 'X' = unknown codon
            protein += aa
        return protein

    # Translate all sequences
    df["protein_sequence"] = df["sequence"].apply(rna_to_protein)

    # Save result
    df.to_csv("output/valid_sequences_with_proteins.csv", index=False)

    print("Protein sequences extracted and saved to output/valid_sequences_with_proteins.csv")


    # =========================================================================

    
    df = pd.read_csv("output/valid_sequences_with_proteins.csv")

    records = []
    for idx, row in df.iterrows():
        seq = row['protein_sequence']
        seq_id = f"seq_{idx}"
        record = SeqRecord(Seq(seq), id=seq_id, description="")
        records.append(record)

    SeqIO.write(records, "output/valid_proteins.fasta", "fasta")
    print("FASTA written: output/valid_proteins.fasta")



    # =========================================================================
    
    
    # Load dataset
    df = pd.read_csv("output/valid_sequences_with_proteins.csv")

    def clean_protein(seq):
        # Remove anything not in standard 20 amino acids
        return re.sub(r'[^ARNDCEQGHILKMFPSTWYV]', '', seq.upper())

    # Prepare FASTA records
    records = []
    for idx, row in df.iterrows():
        clean_seq = clean_protein(row['protein_sequence'])
        if len(clean_seq) > 0:
            record = SeqRecord(Seq(clean_seq), id=f"seq_{idx}", description="")
            records.append(record)

    # Save cleaned FASTA
    SeqIO.write(records, "output/valid_proteins_cleaned.fasta", "fasta")
    print("Cleaned FASTA saved: output/valid_proteins_cleaned.fasta")


# ==========================================================================================
# ==========================================================================================


# =================================
#             EXECUTION
# =================================


if __name__ == "__main__":
    # Configure file paths
    input_filename = 'data/Homo_sapiens.GRCh38.cdna.all.csv'  # Update with your input file
    output_filename = 'output/biological_analysis_report.csv'
    
    # Run analysis pipeline
    main(input_filename, output_filename)
    
    # Visualization
    visualize(output_filename, valid=False)
    
    # Statistical Test Results as Report
    save_statistical_report('output/biological_analysis_report.csv')
    
    # Codon Frequency Bar Plot
    plot_codon_frequency('output/biological_analysis_report.csv')
    plot_codon_frequency_grouped('output/biological_analysis_report.csv')
    
    # Unknown Codon Pie Chart
    plot_unknown_codon_pie('output/biological_analysis_report.csv')
    
    # Compute RSCU
    compute_RSCU("output/biological_analysis_report.csv")
    compute_RSCU_grouping("output/biological_analysis_report.csv", valid=False)
    
    # Compute ENC
    compute_ENC("output/biological_analysis_report.csv")
    compute_ENC_GC3("output/biological_analysis_report.csv", valid=False)
    
    plot_enc_statistics("output/enc_scored_sequences.csv")
    plot_enc_statistics_combine("output/enc_scored_sequences.csv", valid=False)
    
    # Visual Comparison between RSCU and ENC
    plot_rscu_vs_enc("output/rscu_table.csv", "output/enc_scored_sequences.csv", output_path="output/rscu_enc_comparison.png")
    plot_rscu_vs_enc_amino_acid("output/rscu_table.csv", "output/enc_scored_sequences.csv", output_path="output/rscu_enc_comparison_amino_acid.png")
    
    # Codon Usage Entropy per Sequence
    calculate_entropy_per_sequence("output/biological_analysis_report.csv")
    calculate_entropy_per_sequence_violin("output/biological_analysis_report.csv", valid=False)
        
    # Valid Sequences
    postprocess_valid_sequences(
        input_csv='output/biological_analysis_report.csv',
        output_csv_valid='output/valid_sequences_analysis_report.csv'
    )

    # Visualization For Valid Sequences
    visualize('output/valid_sequences_analysis_report.csv', valid=True)
    
    # Statistical Test Results as Report For Valid Sequences
    save_statistical_report('output/valid_sequences_analysis_report.csv', output_file='output/statistical_report_valid.txt')
    
    # Compute RSCU
    compute_RSCU("output/valid_sequences_analysis_report.csv", valid=True)
    compute_RSCU_grouping("output/valid_sequences_analysis_report.csv", valid=True)
    
    # Compute ENC
    compute_ENC("output/valid_sequences_analysis_report.csv", valid=True)
    compute_ENC_GC3("output/valid_sequences_analysis_report.csv", valid=True)
    
    plot_enc_statistics("output/enc_scored_sequences_valid.csv", valid=True)
    plot_enc_statistics_combine("output/enc_scored_sequences_valid.csv", valid=True)

    # Visual Comparison between RSCU and ENC
    plot_rscu_vs_enc("output/rscu_table_valid.csv", "output/enc_scored_sequences_valid.csv", output_path="output/rscu_enc_comparison_valid.png")
    plot_rscu_vs_enc_amino_acid("output/rscu_table_valid.csv", "output/enc_scored_sequences_valid.csv", output_path="output/rscu_enc_comparison_valid_amino_acid.png")
    
    # Codon Usage Entropy per Sequence
    calculate_entropy_per_sequence("output/valid_sequences_analysis_report.csv", valid=True)
    calculate_entropy_per_sequence_violin("output/valid_sequences_analysis_report.csv", valid=True)
        
    # Extract Biological Results
    extract_biological_results()
    
    # Generate Amino Acid Sequences
    extract_protein_sequences()
    
    # =========================================================================
    
    # Generate Analysis Plots
    # Input paths
    ALL_CSV = "output/biological_analysis_report.csv"
    VALID_CSV = "output/valid_sequences_analysis_report.csv"
    SAVE_DIR = "output"

    # Ensure output directory exists
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load data
    df_all = pd.read_csv(ALL_CSV)
    df_valid = pd.read_csv(VALID_CSV)

    # Define codon list for standard order (64 codons)
    from itertools import product
    BASES = ['A', 'U', 'G', 'C']
    STANDARD_CODONS = [''.join(c) for c in product(BASES, repeat=3)]

    plot_codon_usage_heatmap(df_all, "All", "codon_usage_heatmap_all.png")
    plot_codon_usage_heatmap(df_valid, "Valid", "codon_usage_heatmap_valid.png")
    
    plot_full_codon_usage_heatmap(df_all, "All", "codon_usage_heatmap_all_2.png")
    plot_full_codon_usage_heatmap(df_valid, "Valid", "codon_usage_heatmap_valid_2.png")
    
    plot_codon_usage_64_heatmap(df_all, "All", "codon_usage_heatmap_all_3.png")
    plot_codon_usage_64_heatmap(df_valid, "Valid", "codon_usage_heatmap_valid_3.png")
    
    top_all = get_top_codon_table(df_all)
    top_valid = get_top_codon_table(df_valid)

    print("\nTop 10 Codons (All Sequences):")
    print(top_all)

    print("\nTop 10 Codons (Valid Sequences):")
    print(top_valid)
    
    amino_acid_usage_comparison_plot(df_all, df_valid)
    
    plot_top_unknown_codons(df_all, df_valid)

    print("\nAll plots saved in:", SAVE_DIR)


    

    
    