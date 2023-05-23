#!/usr/bin/env python
"""
Script to calculate the number of salt bridges in a given PDB file of a protein.

Usage: calculate_salt_bridge.py pdbfile [csvfile]

Author: Bingrui Yao <ybrprivate@outlook.com>

Mechanism：
    According to https://sbl.inria.fr/doc/Pointwise_interactions-user-manual.html
    To describe salt bridges, recall that atoms of interest for acidic residues are:
    · D (ASP): OD2
    · E (GLU): OE2
    Likewise, for basic residues:
    · R (ARG): NH1 and NH2
    · K (LYS): NZ
    · H (HIS): ND1 and NE2
    For two of them, if the minimum distance found is less than 3.4, and more than 2.5，the pair is termed a salt bridge.
"""

import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# Output directory for salt bridge number files
output_directory = r'/home/**/salt_bridge_number'
# Input directory for PDB files
input_directory = r'/home/**/input_pdbfiles'


def count_salt_bridges(input_file, output_file):
    """
    Count the number of salt bridges in a PDB file.

    Parameters:
    input_file (str): Path to the input PDB file.
    output_file (str): Path to the output CSV file.

    Returns:
    result_df (pd.DataFrame): DataFrame containing the salt bridge count for each PDB file.
    """

    result_dic = {}
    result_bonds = ''

    # Read the PDB file
    pdb_df = pd.read_csv(input_file, delimiter='\s+', header=None)

    # Data processing to extract relevant information
    pdb_df = pdb_df.drop(pdb_df.columns[[0, 1, 4, 5, 9, 10, 11]], axis=1)
    pdb_df.columns = ['Atom_Name', 'Amino_Acid', 'X', 'Y', 'Z']

    # Filter acidic residues in df1，basic residues in df2：
    df1 = pdb_df[((pdb_df['Amino_Acid'] == 'ASP') & (pdb_df['Atom_Name'] == 'OD2')) |
                 ((pdb_df['Amino_Acid'] == 'GLU') & (pdb_df['Atom_Name'] == 'OE2'))]
    df2_1 = pdb_df[(pdb_df['Amino_Acid'] == 'ARG') & (pdb_df['Atom_Name'].isin(['NH1', 'NH2']))]
    df2_2 = pdb_df[(pdb_df['Amino_Acid'] == 'LYS') & (pdb_df['Atom_Name'] == 'NZ')]
    df2_3 = pdb_df[(pdb_df['Amino_Acid'] == 'HIS') & (pdb_df['Atom_Name'].isin(['ND1', 'NE2']))]
    df2 = pd.concat([df2_1, df2_2, df2_3], axis=0)

    # Extract coordinates
    df1_coordinate = df1.loc[:, ['X', 'Y', 'Z']]
    df2_coordinate = df2.loc[:, ['X', 'Y', 'Z']]
    M, N = df1.shape[0], df2.shape[0]
    dist_matrix = np.zeros((M, N))

    # Compute the pairwise Euclidean distances
    for i in range(M):
        for j in range(N):
            dist_matrix[i, j] = np.linalg.norm(df1_coordinate.iloc[i] - df2_coordinate.iloc[j])

    # Filter distances based on criteria
    condition = (dist_matrix > 2.5) & (dist_matrix < 3.4)
    salt_bridge_number = np.count_nonzero(condition)

    # Get PDB ID from input file path
    pdb_id = os.path.splitext(os.path.basename(input_file))[0]

    result_dic[pdb_id] = salt_bridge_number
    result_df = pd.DataFrame.from_dict(result_dic, orient='index', columns=['Salt_Bridge_Number'])
    result_df.to_csv(output_directory + '/' + output_file.split('.')[0] + '.csv', index=True)
    return result_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate salt bridges from input files')
    parser.add_argument('-inputfile', type=str, help='Input PDB file name，e.g example_input.pdb')
    parser.add_argument('-outputfile', type=str,
                        help='Output csv file name which save the results，e.g example_output.csv')
    args = parser.parse_args()
    input_file = args.inputfile
    output_file = args.outputfile

    count_salt_bridges(input_file, output_file)