#!/usr/bin/env python

import itertools
import os
import random
import re
import warnings
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
import requests
from Bio.PDB import *
from Bio.PDB.Polypeptide import is_aa
from scipy.spatial.distance import cdist

# residue names for nucleic acid polymers, used by classify_chain()
_DNA_RESNAMES = {'DA', 'DC', 'DG', 'DT', 'DI', 'DU'}
_RNA_RESNAMES = {'A', 'C', 'G', 'U', 'I'}


def bioplex2graph(bp_PPI_df):
    '''
    Convert BioPlex PPIs into a graph.
    
    This function converts representation of BioPlex PPIs into a 
    graph data structure representation of BioPlex PPIs in a NetworkX 
    object from NetworkX. 

    Parameters
    ----------
    DataFrame of PPIs : Pandas DataFrame

    Returns
    -------
    NetworkX graph
        A NetworkX graph with Nodes = Uniprot Gene Symbols 
        and Edges = interactions.

    Examples
    --------
    # (1) Obtain the latest version of the 293T PPI network
    # (2) Turn the data into a graph
    >>> bp_293t_df = getBioPlex('293T', '3.0')
    >>> bp_293t_G = bioplex2graph(bp_293t_df) 
    >>> type(bp_293t_G)
    <class 'networkx.classes.digraph.DiGraph'>
    >>> len(bp_293t_G.edges)
    115868
    >>> len(bp_293t_G.nodes)
    13689
    '''
    # add isoform columns for Uniprot source & Uniprot target
    bp_PPI_df.loc[:,'isoformA'] = bp_PPI_df.UniprotA
    bp_PPI_df.loc[:,'isoformB'] = bp_PPI_df.UniprotB

    # reconstruct UniprotA/UniprotB columns without '-' isoform id
    UniprotA_new = []
    UniprotB_new = []
    for UniprotA, UniprotB in zip(bp_PPI_df.UniprotA, bp_PPI_df.UniprotB):

        if '-' in UniprotA:
            UniprotA_new.append(UniprotA.split('-')[0])
        else:
            UniprotA_new.append(UniprotA)

        if '-' in UniprotB:
            UniprotB_new.append(UniprotB.split('-')[0])
        else:
            UniprotB_new.append(UniprotB)

    # update columns for Uniprot source 
    # & Uniprot target to exclude isoform '-' ID
    bp_PPI_df.loc[:,'UniprotA'] = UniprotA_new
    bp_PPI_df.loc[:,'UniprotB'] = UniprotB_new
    
    # construct graph from BioPlex PPI data
    bp_G = nx.DiGraph()
    for source, target, pW, pNI, pInt in zip(bp_PPI_df.UniprotA, 
            bp_PPI_df.UniprotB, bp_PPI_df.pW, bp_PPI_df.pNI, bp_PPI_df.pInt):
        bp_G.add_edge(source, target, pW=pW, pNI=pNI, pInt=pInt)
        
    # get mapping uniprot -> entrez & store as node attribute
    uniprot_entrez_dict = {}
    for uniprot_A, entrez_A in zip(bp_PPI_df.UniprotA, bp_PPI_df.GeneA):
        uniprot_entrez_dict[uniprot_A] = entrez_A
    for uniprot_B, entrez_B in zip(bp_PPI_df.UniprotB, bp_PPI_df.GeneB):
        uniprot_entrez_dict[uniprot_B] = entrez_B

    for node_i in bp_G.nodes():
        bp_G.nodes[node_i]["entrezid"] = uniprot_entrez_dict[node_i]

    # get mapping uniprot -> symbol & store as node attribute
    uniprot_symbol_dict = {}
    for uniprot_A, symbol_A in zip(bp_PPI_df.UniprotA, bp_PPI_df.SymbolA):
        uniprot_symbol_dict[uniprot_A] = symbol_A
    for uniprot_B, symbol_B in zip(bp_PPI_df.UniprotB, bp_PPI_df.SymbolB):
        uniprot_symbol_dict[uniprot_B] = symbol_B

    for node_i in bp_G.nodes():
        bp_G.nodes[node_i]["symbol"] = uniprot_symbol_dict[node_i]

    # get mapping uniprot -> isoform & store as node attribute
    uniprot_isoform_dict = {}
    for uniprot_A, isoform_A in zip(bp_PPI_df.UniprotA, bp_PPI_df.isoformA):
        uniprot_isoform_dict[uniprot_A] = isoform_A
    for uniprot_B, isoform_B in zip(bp_PPI_df.UniprotB, bp_PPI_df.isoformB):
        uniprot_isoform_dict[uniprot_B] = isoform_B

    for node_i in bp_G.nodes():
        bp_G.nodes[node_i]["isoform"] = uniprot_isoform_dict[node_i]
        
    # get set of baits & store a bait boolean as node attribute 
    # if node is a bait True & False otherwise
    bp_i_baits = set(bp_PPI_df.UniprotA)
    for node_i in bp_G.nodes():
        if node_i in bp_i_baits:
            bp_G.nodes[node_i]["bait"] = True
        else:
            bp_G.nodes[node_i]["bait"] = False
    
    return bp_G

def get_PPI_network_for_complex(bp_PPI_G, Corum_DF, Complex_ID):
    '''
    Retrieve Network of BioPlex (AP-MS) PPIs for a CORUM complex.
    
    This function returns a subgraph of PPIs identified through AP-MS
    between the proteins in a specified CORUM complex.

    Parameters
    ----------
    Network of PPIs : NetworkX graph
    DataFrame of CORUM complexes : Pandas DataFrame
    Corum Complex ID: int

    Returns
    -------
    NetworkX Graph
        A subgraph induced by the proteins in a CORUM complex 
        from the BioPlex network used as input.

    Examples
    --------
    # (1) Obtain the latest version of the 293T PPI network
    # (2) Obtain NetworkX graph representation of 293T PPI network
    # (3) Obtain CORUM complexes
    # (4) Get AP-MS interactions as subgraph for a specified protein complex using PPI data
    >>> bp_293t_df = getBioPlex('293T', '3.0')
    >>> bp_293t_G = bioplex2graph(bp_293t_df)
    >>> Corum_DF = getCorum()
    >>> ING2_bp_293t_G = get_PPI_network_for_complex(bp_293t_G, Corum_DF, 2851)
    >>> type(ING2_bp_293t_G)
    <class 'networkx.classes.digraph.DiGraph'>
    >>> len(ING2_bp_293t_G)
    12
    '''
    # store gene UNIPROT IDs that belong to this complex in a list
    genes_in_complex_i = (Corum_DF[Corum_DF.complex_id == Complex_ID].loc[:,
                                'subunits_uniprot_id'].values[0].split(';'))
    
    # get subgraph induced by the subset of nodes in this CORUM complex
    bp_complex_i_G = bp_PPI_G.subgraph(genes_in_complex_i)
    
    return bp_complex_i_G

def get_DataFrame_from_PPI_network(bp_PPI_G):
    '''
    Convert Network of BioPlex (AP-MS) PPIs into DataFrame of BioPlex 
    interaction Network.
    
    This function returns a DataFrame of PPIs (identified through AP-MS) 
    represented as a graph.

    Parameters
    ----------
    Network of PPIs : NetworkX graph

    Returns
    -------
    Pandas DataFrame
        A DataFrame of edges (AP-MS interactions) from a network.

    Examples
    --------
    # (1) Obtain the latest version of the 293T PPI network
    # (2) Obtain NetworkX graph representation of 293T PPI network
    # (3) Obtain CORUM complexes
    # (4) Get AP-MS interactions as subgraph for a specified protein complex using PPI data row corresponding to an edge
    # (5) Convert ING2 AP-MS network into DataFrame w/ each 
    # order of rows in dataframe?
    >>> bp_293t_df = getBioPlex('293T', '3.0')
    >>> bp_293t_G = bioplex2graph(bp_293t_df)
    >>> Corum_DF = getCorum()
    >>> ING2_bp_293t_G = get_PPI_network_for_complex(bp_293t_G, Corum_DF, 2851)
    >>> ING2_bp_293t_df = get_DataFrame_from_PPI_network(ING2_bp_293t_G)
    >>> type(ING2_bp_293t_df)
    <class 'pandas.core.frame.DataFrame'>
    >>> ING2_bp_293t_df.shape
    (51, 7)
    >>> ING2_bp_293t_df.columns
    Index(['UniprotA', 'UniprotB', 'SymbolA', 'SymbolB', 'pW', 'pNI', 'pInt'], dtype='object')
    '''
    # get list of edges in network
    PPI_edge_list = list(bp_PPI_G.edges)

    # make node_A and node_B columns for both UNIPROT & SYMBOLS
    uniprotA_list = []
    uniprotB_list = []
    symbolA_list = []
    symbolB_list = []

    # make columns for calcs detected for each edge
    pW_list = []
    pNI_list = []
    pInt_list = []

    # iterate through each edge to store data for each row of DataFrame
    for edge_i in PPI_edge_list:

        nodeA, nodeB = edge_i

        # nodes are labeled with UNIPROT
        uniprotA_list.append(nodeA)
        uniprotB_list.append(nodeB)

        # get gene SYMBOL
        symbolA_list.append(bp_PPI_G.nodes[nodeA]['symbol'])
        symbolB_list.append(bp_PPI_G.nodes[nodeB]['symbol'])

        # get AP-MS calculations
        pW_list.append(bp_PPI_G.edges[edge_i]['pW'])
        pNI_list.append(bp_PPI_G.edges[edge_i]['pNI'])
        pInt_list.append(bp_PPI_G.edges[edge_i]['pInt'])

    # convert lists into cols of DataFrame
    bp_complex_i_df = pd.DataFrame()
    bp_complex_i_df.loc[:,'UniprotA'] = uniprotA_list
    bp_complex_i_df.loc[:,'UniprotB'] = uniprotB_list
    bp_complex_i_df.loc[:,'SymbolA'] = symbolA_list
    bp_complex_i_df.loc[:,'SymbolB'] = symbolB_list
    bp_complex_i_df.loc[:,'pW'] = pW_list
    bp_complex_i_df.loc[:,'pNI'] = pNI_list
    bp_complex_i_df.loc[:,'pInt'] = pInt_list
    
    return bp_complex_i_df
    
def get_prop_edges_in_complex_identified(bp_PPI_G, Corum_DF, Complex_ID):
    '''
    Calculates proportion of all possible edges identified from BioPlex (AP-MS) 
    PPIs for a CORUM complex.
    
    This function returns the proportion of all possible PPIs identified 
    through AP-MS between the proteins in a specified CORUM complex.

    Parameters
    ----------
    DataFrame of PPIs : Pandas DataFrame
    DataFrame of CORUM complexes : Pandas DataFrame
    Corum Complex ID: int

    Returns
    -------
    Float
        The proportion of interactions between all proteins in CORUM complex 
        identified through AP-MS PPI data

    Examples
    --------
    # (1) Obtain the latest version of the 293T PPI network
    # (2) Obtain NetworkX graph representation of 293T PPI network
    # (3) Obtain CORUM complexes
    # (4) Get proportion of interactions identified for a specified CORUM complex 
    # using PPI data

    >>> bp_293t_df = getBioPlex('293T', '3.0')
    >>> bp_293t_G = bioplex2graph(bp_293t_df)
    >>> Corum_DF = getCorum()
    >>> get_prop_edges_in_complex_identified(bp_293t_G, Corum_DF, 2851)
    '''
    # store gene UNIPROT IDs that belong to this complex in a list
    genes_in_complex_i = (Corum_DF[Corum_DF.complex_id == Complex_ID].loc[:,
                                'subunits_uniprot_id'].values[0].split(';'))
    
    # get subgraph induced by the subset of nodes in this CORUM complex
    bp_complex_i_G = bp_PPI_G.subgraph(genes_in_complex_i)
        
    # create a complete graph from the nodes of complex graph 
    # (all possible interactions between proteins)
    bp_complex_i_G_complete = nx.Graph()
    bp_complex_i_G_complete.add_nodes_from(genes_in_complex_i)
    bp_complex_i_G_complete.add_edges_from(
                    itertools.combinations(genes_in_complex_i, 2))
    
    # calculate proportion of interactions between proteins in complex 
    # identified through AP-MS
    prop_edges_identified = (float(len(list(bp_complex_i_G.edges)))/
                             float(len(list(bp_complex_i_G_complete.edges))))
    
    # return proportion of edges ID'd through AP-MS, round to 3 decimal places
    return round(prop_edges_identified, 3)

def resampling_test_for_uniprot_list(bp_PPI_G, uniprot_list, 
                    num_resamples = 1000, preserve_node_degree = False):
    '''
    Run resampling test to check for enrichment of BioPlex (AP-MS) PPIs 
    for a given list of proteins (uniprot IDs).

    This function returns a p-value after running a resampling test by 
    1. taking the number of proteins in the specified list of uniprot IDs (N), 
    2. choosing N random proteins from the Graph generated by all of the 
       PPI data (G),
    3. calculating the number of edges in the Subgraph (S) induced by N random 
       proteins (with the same proportion of baits (+/- 10%) as the complex) 
       and storing this value (E_i),
    4. if preserve_node_degree option is invoked, then baits & preys in S 
       must have the same degree distribution as baits & preys in the
       network generated by given uniprot IDs, respectively,
    5. repeating steps 1-3 num_resamples times to create a null distribution, 
    6. calculating the number of edges between N proteins in the 
       complex (E), 
    7. returning a p-value by calculating the proportion of values 
       [E_1, E_2, ... , E_num_resamples] that are greater than or equal to E.

    Parameters
    ----------
    Network of PPIs : NetworkX graph
    list of uniprot IDs : list
    Number of resamplings: int
    option to preserve degree distribution in subgraphs : bool

    Returns
    -------
    Float
        A p-value from a resampling test to check for enrichment of PPIs 
        detected between proteins in list

    Examples
    --------
    # (1) Obtain the latest version of the 293T PPI network
    >>> bp_293t_df = getBioPlex('293T', '3.0')
    # (2) Obtain NetworkX graph representation of 293T PPI network
    >>> bp_293t_G = bioplex2graph(bp_293t_df)
    # (3) Obtain CORUM complexes
    >>> Corum_DF = getCorum()
    # (4) Get list of uniprots for Arp2/3 complex
    >>> UniProts_Arp_2_3 = get_UniProts_from_CORUM(Corum_DF, Complex_ID = 27)
    # (5) Calculate p-value to check for enrichment of edges in 
    #     Arp2/3 protein complex
    >>> resampling_test_for_CORUM_complex(bp_293t_G, UniProts_Arp_2_3, 1000)
    '''
    # get subgraph induced by the subset of nodes in this protein list
    bp_uniprots_i_G = bp_PPI_G.subgraph(uniprot_list)

    # number of edges detected between proteins in this CORUM 
    # complex among PPI data
    num_edges_identified_uniprot_list = float(len(list(bp_uniprots_i_G.edges)))

    # if no edges detected in this uniprot list, return an ERROR message
    if num_edges_identified_uniprot_list == 0.0:
        print('ERROR: no edges detected in PPI data for this protein list, '
              'p-value could not be computed.')
        return

    # if at least 1 edge in complex, estimate p-value 
    # using resampling test

    # number of genes in complex (N genes)
    num_genes_in_uniprot_list = len(list(bp_uniprots_i_G.nodes))    

    # list of nodes in large network generated from PPI data
    nodes_in_overall_PPI_network = list(bp_PPI_G.nodes)
    
    # bait degrees in large PPI network
    ########################################
    # create a filter for uniprot IDs in large network that are baits
    G_baits_filter = np.array(
        [bp_PPI_G.nodes[node_i]['bait'] for node_i in bp_PPI_G.nodes])

    # get array of baits in large network
    G_baits = np.array(nodes_in_overall_PPI_network)[G_baits_filter]

    # get degree of each bait
    G_baits_degrees = bp_PPI_G.degree(G_baits)

    # prey degrees in large PPI network
    ########################################
    # create a filter for uniprot IDs in complex that are preys
    G_preys_filter = np.array(
        [not bp_PPI_G.nodes[node_i]['bait'] for node_i in bp_PPI_G.nodes])

    # get array of preys in large network
    G_preys = np.array(nodes_in_overall_PPI_network)[G_preys_filter]

    # get degree of each prey
    G_preys_degrees = bp_PPI_G.degree(G_preys)

    # bait degrees in complex
    ########################################
    # get array of uniprot IDs in complex
    bp_uniprots_i_G_nodes = np.array(bp_uniprots_i_G.nodes)

    # create a filter for uniprot IDs in complex that are baits
    bp_uniprots_i_G_baits_filter = np.array(
        [bp_uniprots_i_G.nodes[node_i]['bait'] for 
             node_i in bp_uniprots_i_G.nodes])

    # get list of baits in complex
    bp_uniprots_i_G_baits = bp_uniprots_i_G_nodes[
        bp_uniprots_i_G_baits_filter]

    # get degree of each bait
    bp_uniprots_i_G_baits_degrees = bp_uniprots_i_G.degree(
                                    bp_uniprots_i_G_baits)

    # get degree distribution of baits
    bp_uniprots_i_G_baits_degree_distr = Counter(
        [bp_uniprots_i_G_baits_degrees[uniprot_i] for 
         uniprot_i in bp_uniprots_i_G_baits])
    
    # remove count for any "0" degree baits
    # would not contribute to edges in S
    if 0 in bp_uniprots_i_G_baits_degree_distr.keys():
        del bp_uniprots_i_G_baits_degree_distr[0]
    
    # find number of baits in complex
    
    # if preserve node degree TRUE
    # exclude baits w/ degree 0 from bait count
    # b/c random subgraphs won't have these
    if preserve_node_degree == True:
        bp_complex_i_baits_num = (np.sum(list(
                bp_uniprots_i_G_baits_degree_distr.values())))
    
    # else count all baits in complex
    else:
        bp_complex_i_baits_num = (np.sum(bp_uniprots_i_G_baits_filter))
    
    # find proportion of baits in complex
    bp_complex_i_baits_prop = (float(bp_complex_i_baits_num) / 
                               float(num_genes_in_uniprot_list))

    # prey degrees in complex
    ########################################
    # get array of uniprot IDs in complex
    bp_uniprots_i_G_nodes = np.array(bp_uniprots_i_G.nodes)

    # create a filter for uniprot IDs in complex that are preys
    bp_uniprots_i_G_preys_filter = np.array(
        [not bp_uniprots_i_G.nodes[node_i]['bait'] for 
         node_i in bp_uniprots_i_G.nodes])

    # get list of prey in complex
    bp_uniprots_i_G_preys = bp_uniprots_i_G_nodes[
        bp_uniprots_i_G_preys_filter]

    # get degree of each prey
    bp_uniprots_i_G_preys_degrees = bp_uniprots_i_G.degree(
        bp_uniprots_i_G_preys)

    # get degree distribution of preys
    bp_uniprots_i_G_preys_degree_distr = Counter(
        [bp_uniprots_i_G_preys_degrees[uniprot_i] for 
         uniprot_i in bp_uniprots_i_G_preys])
    
    # remove count for any "0" degree preys
    # would not contribute to edges in S
    if 0 in bp_uniprots_i_G_preys_degree_distr.keys():
        del bp_uniprots_i_G_preys_degree_distr[0]

    # list that will store number of edges detected in each subgraph
    num_edges_random_subgraphs = []

    # iterate through num_resamples random subgraphs induced by N nodes
    S_i = 0
    while S_i < num_resamples:
        
        # if degree distribution option is invoked
        if preserve_node_degree == True:

            # baits in S
            ########################################
            # preserve degrees by randomly pulling baits
            # with same degree for each uniprot in complex
            S_rando_baits = []
            for deg_i, uniprot_count_i in zip(
                bp_uniprots_i_G_baits_degree_distr.keys(), 
                bp_uniprots_i_G_baits_degree_distr.values()):

                # get baits in G with same degree
                G_baits_with_deg_i = list(
                    G_baits[np.array(
                        [G_baits_degrees[bait_i] == deg_i for 
                         bait_i in G_baits])])
                
                # choose N baits w/ same degree at random w/o replacement
                N_rando_baits_from_PPI_network = random.sample(
                    G_baits_with_deg_i, uniprot_count_i)

                # add to list of random baits
                S_rando_baits = (S_rando_baits + 
                                 N_rando_baits_from_PPI_network)

            # preys in S
            ########################################
            # preserve degrees by randomly pulling preys
            # with same degree for each uniprot in complex
            S_rando_preys = []
            for deg_i, uniprot_count_i in zip(
                bp_uniprots_i_G_preys_degree_distr.keys(), 
                bp_uniprots_i_G_preys_degree_distr.values()):

                # get preys in G with same degree
                G_preys_with_deg_i = list(
                    G_preys[np.array([G_preys_degrees[prey_i] == deg_i for 
                                      prey_i in G_preys])])

                # choose N preys w/ same degree at random w/o replacement
                N_rando_preys_from_PPI_network = random.sample(
                    G_preys_with_deg_i, uniprot_count_i)

                # add to list of random preys
                S_rando_preys = (S_rando_preys + 
                                 N_rando_preys_from_PPI_network)

            # combine random Preys & Baits while preserving degrees
            N_rando_nodes_from_PPI_network = S_rando_baits + S_rando_preys

        # if degree distribution not invoked
        elif preserve_node_degree == False:

            # choose N genes at random without replacement
            N_rando_nodes_from_PPI_network = random.sample(
                nodes_in_overall_PPI_network, num_genes_in_uniprot_list)

        # get subgraph induced by random subset of nodes
        bp_PPI_S = bp_PPI_G.subgraph(N_rando_nodes_from_PPI_network)

        # check to see if nodes in subgraph have the same proportion of 
        # baits as the subgraph induced by the CORUM complex
        # excluding baits that had degree 0
        bp_S_baits_num = np.sum(
            [bp_PPI_S.nodes[node_i]['bait'] for node_i in bp_PPI_S.nodes])
        bp_S_baits_prop = (float(bp_S_baits_num) / 
                           float(num_genes_in_uniprot_list))

        # proportion of baits in CORUM complex & S are the same (+/- 10%)
        if abs(bp_complex_i_baits_prop - bp_S_baits_prop) <= 0.1:

            # calculate the number of edges detected within 
            # subgraph induced by random nodes
            num_edges_S = float(len(list(bp_PPI_S.edges)))

            # store in list that contains resamplings
            num_edges_random_subgraphs.append(num_edges_S)

            # count this as a resampling
            S_i += 1

    # convert list to numpy array
    num_edges_random_subgraphs = np.array(num_edges_random_subgraphs)

    # calculate proportion of subgraphs that had more edges than edges 
    # detected in complex (p-val from resampling test)
    p_val = (float(np.sum(num_edges_random_subgraphs 
                            >= num_edges_identified_uniprot_list) + 1.0) / 
                        (float(num_resamples) + 1.0))
    return p_val

def classify_chain(chain):
    '''
    Classify a Bio.PDB Chain as 'protein', 'dna', 'rna', or 'other'.

    Classification is based on the majority residue type among the
    chain's polymer residues (residues with a blank hetero-flag, i.e.
    excluding waters, ions, and other heteroatom/ligand records).

    Parameters
    ----------
    chain: Bio.PDB.Chain.Chain

    Returns
    -------
    str
        One of 'protein', 'dna', 'rna', 'other'. 'other' is returned for
        chains with no polymer residues (e.g. a chain consisting only of
        a bound ligand or crystallographic waters) or for chains whose
        polymer residues are not majority amino acid/DNA/RNA.

    Examples
    --------
    # (1) Classify each chain in the CDC45-MCM-GINS helicase structure,
    #     which includes a DNA chain (chain 'M') -- see classify_pdb_chains()
    #     for a runnable example using this function under the hood.
    '''
    n_aa = n_dna = n_rna = n_total = 0
    for residue in chain:
        if residue.id[0] != ' ':
            continue  # skip waters/ions/ligands (heteroatom records)
        n_total += 1
        resname = residue.get_resname().strip()
        if is_aa(resname, standard=True):
            n_aa += 1
        elif resname in _DNA_RESNAMES:
            n_dna += 1
        elif resname in _RNA_RESNAMES:
            n_rna += 1

    if n_total == 0:
        return 'other'
    if n_aa / n_total > 0.5:
        return 'protein'
    if n_dna / n_total > 0.5:
        return 'dna'
    if n_rna / n_total > 0.5:
        return 'rna'
    return 'other'


_LOCAL_STRUCTURE_FORMATS = {'.pdb': 'pdb', '.ent': 'pdb',
                             '.cif': 'mmCif', '.mmcif': 'mmCif'}


def is_local_structure_file(structure):
    '''
    Return True if `structure` is a path to an existing local structure
    file (.pdb/.ent/.cif/.mmcif, case-insensitive) rather than a PDB ID.

    Every structure-taking function in BioPlexPy accepts either form: a
    PDB ID is downloaded from RCSB, a local file (e.g. a user's own
    experimental model or an AlphaFold/Boltz/ColabFold prediction) is
    read in place.

    Examples
    --------
    >>> is_local_structure_file('6NMI')
    False
    '''
    if not isinstance(structure, (str, os.PathLike)):
        return False
    return os.path.isfile(structure)


def find_structure_files(path, extract_dir=None):
    '''
    Find the model files in a structure predictor's output, as written by
    the tool -- a folder, or the .zip the AlphaFold3 server downloads --
    or in any directory of structure files.

    A directory is searched at its top level (.pdb/.ent/.cif/.mmcif),
    which covers the AlphaFold3 server (model_0..4.cif at the top, with
    template hits in a templates/ subfolder that must NOT be counted as
    models) and ColabFold (ranked PDBs at the top). Known nested layouts
    are recognized explicitly rather than by searching every subfolder,
    since a stray template or reference file counted as a "model" would
    silently skew any across-model summary:

    - Boltz: a boltz_results_<name>/ folder (or a folder containing one)
      keeps its models in predictions/<input>/<input>_model_N.cif.

    ColabFold writes both relaxed and unrelaxed copies of each ranked
    model when relaxation is on; only the unrelaxed one is kept, so the
    same model isn't counted twice and every tool is compared on
    unrelaxed coordinates.

    If nothing is found, any structure files in subfolders are listed in
    the returned notes, so the right folder can be given instead.

    A .zip archive is handled like the folder it contains: only its
    structure files are extracted (not MSAs, JSON, or template hits), into
    extract_dir, and the same layout rules are applied there.

    Parameters
    ----------
    path to a structure file, a directory, or a .zip archive: str
    extract_dir: str (optional)
        Where to extract a .zip's structure files. Defaults to a new
        temporary directory (named in the returned notes), which is not
        deleted automatically.

    Returns
    -------
    list of str
        Model file paths, sorted.
    list of str
        Notes on what was detected or skipped, for display.
    list of str
        Prediction groups found (Boltz predictions/<input> folders). More
        than one means the output holds several different inputs, which
        shouldn't be summarized together.

    Examples
    --------
    >>> files, notes, groups = find_structure_files('.')
    '''
    exts = tuple(_LOCAL_STRUCTURE_FORMATS)

    def structure_files_in(directory):
        return sorted(os.path.join(directory, f) for f in os.listdir(directory)
                      if f.lower().endswith(exts)
                      and os.path.isfile(os.path.join(directory, f)))

    if str(path).lower().endswith('.zip') and os.path.isfile(path):
        return _find_structure_files_in_zip(path, extract_dir)
    if os.path.isfile(path):
        return [str(path)], [], []
    if not os.path.isdir(path):
        raise FileNotFoundError(f"No such file or directory: '{path}'")

    files, notes, groups = [], [], []

    # Boltz: `path` is a boltz_results_* folder (has predictions/), or
    # contains one or more boltz_results_* folders
    boltz_roots = [os.path.join(path, d) for d in sorted(os.listdir(path))
                   if d.startswith('boltz_results_')
                   and os.path.isdir(os.path.join(path, d))]
    if os.path.isdir(os.path.join(path, 'predictions')):
        boltz_roots.insert(0, str(path))
    for root in boltz_roots:
        predictions = os.path.join(root, 'predictions')
        if not os.path.isdir(predictions):
            continue
        for group in sorted(os.listdir(predictions)):
            group_dir = os.path.join(predictions, group)
            if not os.path.isdir(group_dir):
                continue
            models = [f for f in structure_files_in(group_dir)
                      if re.search(r'_model_\d+\.', os.path.basename(f))]
            if models:
                groups.append(group_dir)
                files.extend(models)
                notes.append(f'Boltz output: {len(models)} model(s) in {group_dir}')

    top_level = structure_files_in(path)

    # ColabFold: drop the relaxed duplicate of each ranked unrelaxed model
    colabfold = {}
    for f in top_level:
        m = re.match(r'(.*)_(relaxed|unrelaxed)_(rank_\d+.*)$', os.path.basename(f))
        if m:
            colabfold.setdefault((m.group(1), m.group(3)), {})[m.group(2)] = f
    dropped = {kinds['relaxed'] for kinds in colabfold.values()
               if 'relaxed' in kinds and 'unrelaxed' in kinds}
    if dropped:
        notes.append(f'ColabFold output: skipped {len(dropped)} relaxed duplicate(s) '
                     'of unrelaxed models')
    files.extend(f for f in top_level if f not in dropped)

    if not files:
        nested = [os.path.join(d, f) for d, _, fs in os.walk(path) for f in fs
                  if f.lower().endswith(exts)]
        if nested:
            shown = ', '.join(nested[:5]) + (' ...' if len(nested) > 5 else '')
            notes.append(f'No model files at the top of {path}, but {len(nested)} '
                         f'structure file(s) in subfolders ({shown}) -- point at '
                         'the folder that holds the models themselves.')
        else:
            notes.append(f'No structure files found in {path}')
        zips = [f for f in os.listdir(path) if f.lower().endswith('.zip')]
        if zips:
            notes.append(f'{path} holds .zip archive(s) ({", ".join(zips[:3])}) -- '
                         'pass the .zip itself to read the models inside it.')
    return sorted(files), notes, groups


def _find_structure_files_in_zip(zip_path, extract_dir=None):
    '''
    Internal helper for find_structure_files(): extract just the
    structure files from a .zip (skipping template hits and anything
    whose path would land outside extract_dir), then apply the same
    layout rules to the extracted tree.
    '''
    import tempfile
    import zipfile

    exts = tuple(_LOCAL_STRUCTURE_FORMATS)
    if extract_dir is None:
        extract_dir = tempfile.mkdtemp(prefix='bioplexpy_zip_')
    root = os.path.realpath(extract_dir)
    notes, n_extracted = [], 0
    with zipfile.ZipFile(zip_path) as archive:
        members = [m for m in archive.infolist()
                   if not m.is_dir() and m.filename.lower().endswith(exts)
                   and 'templates' not in m.filename.split('/')[:-1]]
        for member in members:
            target = os.path.realpath(os.path.join(root, member.filename))
            if not target.startswith(root + os.sep):
                notes.append(f'Skipped unsafe path in archive: {member.filename}')
                continue
            archive.extract(member, root)
            n_extracted += 1
    notes.append(f'Extracted {n_extracted} structure file(s) from {zip_path} '
                 f'into {root}')

    # an archive of a single folder: look inside that folder
    target_dir = root
    entries = os.listdir(target_dir)
    while (len(entries) == 1 and os.path.isdir(os.path.join(target_dir, entries[0]))
           and not entries[0].startswith('boltz_results_')):
        target_dir = os.path.join(target_dir, entries[0])
        entries = os.listdir(target_dir)

    files, more_notes, groups = find_structure_files(target_dir)
    return files, notes + more_notes, groups


def structure_label(structure):
    '''
    Short, filesystem-safe name for a structure: the PDB ID itself, or a
    local file's name without its extension. Used for figure titles and
    temporary file names, where a full path would not work.

    Examples
    --------
    >>> structure_label('6NMI')
    '6NMI'
    '''
    if is_local_structure_file(structure):
        name = os.path.splitext(os.path.basename(structure))[0]
    else:
        name = str(structure)
    return re.sub(r'[^A-Za-z0-9_.-]', '_', name)


def fetch_pdb_structure_file(PDB_ID_structure_i, protein_structure_dir):
    '''
    Download a PDB structure, preferring the legacy PDB format but falling
    back to mmCIF if RCSB doesn't have a legacy file for it. If given the
    path to a local structure file instead of a PDB ID, nothing is
    downloaded: the file is used in place, with its format taken from the
    extension (.pdb/.ent -> 'pdb', .cif/.mmcif -> 'mmCif').

    RCSB no longer generates legacy .pdb files for a growing share of new
    depositions (typically larger/newer cryo-EM structures) -- some don't
    fit the legacy format's hard limits (62 chains, 99999 atoms, 4-digit
    residue numbers) at all. Every structure on RCSB has an mmCIF file
    though, so falling back to it (rather than failing outright) means
    BioPlexPy's structure-based functions keep working on those newer
    structures. Biopython's MMCIFParser uses the same (author) chain IDs as
    the legacy PDB format by default, so chain-ID-keyed logic elsewhere
    (e.g. the SIFTS-based UniProt mappings in list_uniprot_pdb_mappings())
    is unaffected by which format was actually used.

    Parameters
    ----------
    PDB ID: str
    directory to store PDB file: str

    Returns
    -------
    (str, str)
        (file path, format), where format is 'pdb' or 'mmCif'.

    Examples
    --------
    >>> file_path, file_format = fetch_pdb_structure_file('6NMI', '.')
    Downloading PDB structure '6nmi'...
    >>> file_format
    'pdb'
    '''
    if is_local_structure_file(PDB_ID_structure_i):
        ext = os.path.splitext(str(PDB_ID_structure_i))[1].lower()
        if ext not in _LOCAL_STRUCTURE_FORMATS:
            raise ValueError(
                f"Unrecognized structure file extension '{ext}' for "
                f"'{PDB_ID_structure_i}'; expected one of "
                f"{sorted(_LOCAL_STRUCTURE_FORMATS)} (decompress .gz files first).")
        return str(PDB_ID_structure_i), _LOCAL_STRUCTURE_FORMATS[ext]

    pdbl = PDBList()
    pdb_file_path = pdbl.retrieve_pdb_file(PDB_ID_structure_i,
                                           pdir=protein_structure_dir,
                                           file_format='pdb',
                                           overwrite=True)
    if pdb_file_path and os.path.exists(pdb_file_path):
        return pdb_file_path, 'pdb'

    cif_file_path = pdbl.retrieve_pdb_file(PDB_ID_structure_i,
                                           pdir=protein_structure_dir,
                                           file_format='mmCif',
                                           overwrite=True)
    if cif_file_path and os.path.exists(cif_file_path):
        return cif_file_path, 'mmCif'

    raise FileNotFoundError(
        f"Could not download structure '{PDB_ID_structure_i}' from RCSB "
        "in either legacy PDB or mmCIF format.")


def _load_pdb_model(PDB_ID_structure_i, protein_structure_dir):
    '''
    Download (if needed) and parse a PDB structure, returning its first model.

    Internal helper shared by get_interacting_chains_from_PDB(),
    classify_pdb_chains(), and get_chain_centroids() so the structure is
    downloaded/parsed with consistent behavior in all three.
    '''
    file_path, file_format = fetch_pdb_structure_file(PDB_ID_structure_i,
                                                       protein_structure_dir)
    parser = MMCIFParser(QUIET=True) if file_format == 'mmCif' else PDBParser(QUIET=True)
    structure = parser.get_structure(structure_label(PDB_ID_structure_i), file_path)
    if len(structure) == 0:
        raise ValueError(f"No atoms could be read from '{file_path}' -- is it a "
                         'valid PDB/mmCIF structure file?')
    return structure[0]


def classify_pdb_chains(PDB_ID_structure_i, protein_structure_dir):
    '''
    Classify every chain in a PDB structure as 'protein', 'dna', 'rna',
    or 'other'.

    Parameters
    ----------
    PDB ID: str
    directory to store PDB file: str

    Returns
    -------
    dict
        Mapping of chain ID -> classification ('protein', 'dna', 'rna',
        or 'other'), via classify_chain().

    Examples
    --------
    # (1) Classify chains in the Ribonuclease P structure, which
    #     includes an RNA chain ('A', the H1 RNA)
    >>> chain_types = classify_pdb_chains('6AHR', '.')
    Downloading PDB structure '6ahr'...
    >>> chain_types['A']
    'rna'
    '''
    model = _load_pdb_model(PDB_ID_structure_i, protein_structure_dir)
    return _classify_chains(model)


def _classify_chains(model):
    '''
    Internal helper: classify every chain in an already-loaded Bio.PDB
    model. Factored out of classify_pdb_chains() so
    PDB_to_interacting_chains_uniprot_maps() can reuse a single
    downloaded/parsed structure instead of re-fetching it.
    '''
    return {chain.get_id(): classify_chain(chain) for chain in model}


def get_chain_centroids(PDB_ID_structure_i, protein_structure_dir):
    '''
    Compute the 3D centroid (mean polymer-atom coordinate) of every
    protein/DNA/RNA chain in a PDB structure.

    Used to derive a network layout from the structure's real geometry
    (see get_structure_based_layout() in visualization_funcs.py), rather
    than an arbitrary layout algorithm, so the network diagram's spatial
    arrangement reflects how the chains are actually arranged in 3D.

    Parameters
    ----------
    PDB ID: str
    directory to store PDB file: str

    Returns
    -------
    dict
        Mapping of chain ID -> numpy array of shape (3,) (the mean x/y/z
        coordinate of that chain's polymer atoms). Chains classified as
        'other' (no polymer residues, e.g. a lone ligand) are omitted.

    Examples
    --------
    >>> centroids = get_chain_centroids('6NMI', '.')
    Downloading PDB structure '6nmi'...
    >>> sorted(centroids.keys())
    ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    >>> centroids['A'].shape
    (3,)
    '''
    model = _load_pdb_model(PDB_ID_structure_i, protein_structure_dir)
    centroids = {}
    for chain in model:
        if classify_chain(chain) == 'other':
            continue
        coords = [atom.coord for residue in chain if residue.id[0] == ' '
                  for atom in residue]
        if coords:
            centroids[chain.get_id()] = np.mean(np.vstack(coords), axis=0)
    return centroids


def _plddt_scale_factor(model):
    '''
    Internal helper: work out how to read a model's B-factor column as
    pLDDT on a 0-100 scale, for predicted structures (AlphaFold2/3,
    ColabFold, Boltz and ESMFold all store per-atom pLDDT there, some on
    0-100 and some on 0-1).

    Returns 100.0 if every polymer-atom value is within 0-1, 1.0 if they
    are within 0-100, or None (with a warning) if the column can't be
    pLDDT -- all zeros, or anything outside 0-100 (e.g. a real
    experimental B-factor) -- in which case the caller should skip
    filtering rather than silently drop every atom.
    '''
    bfactors = np.array([atom.get_bfactor() for atom in model.get_atoms()
                         if atom.get_parent().id[0] == ' '])
    if bfactors.size == 0 or not np.any(bfactors):
        warnings.warn('min_plddt given, but the B-factor column is empty/all '
                      'zero, so it holds no pLDDT values -- skipping the pLDDT '
                      'filter.')
        return None
    if bfactors.min() < 0 or bfactors.max() > 100:
        warnings.warn('min_plddt given, but B-factor values fall outside 0-100, '
                      'so they are not pLDDT (is this an experimental '
                      'structure?) -- skipping the pLDDT filter.')
        return None
    return 100.0 if bfactors.max() <= 1.0 else 1.0


def get_interacting_chains_from_PDB(PDB_ID_structure_i, protein_structure_dir, dist_threshold):
    '''
    Retreive chain pairs that are physically close to eachother from
    PDB structure.

    This function downloads the PDB structure that is specified from the input
    PDB ID into the input directory, then computes the pairwise distances
    between all atoms for each pair of chains in the structure. A list of
    chain pairs that are interacting (have at least a pair of
    atoms < dist_threshold angstroms apart) is returned.

    Only polymer atoms (standard amino acid or nucleic acid residues) are
    considered; atoms from waters, ions, and other bound heteroatoms/ligands
    are excluded from the distance calculation, and chains with no polymer
    residues (e.g. a lone ligand or water "chain") are skipped entirely.
    This means protein and nucleic acid (DNA/RNA) chains are both eligible
    to be reported as interacting.

    Parameters
    ----------
    PDB ID: str
    directory to store PDB file: str
    distance threshold: int

    Returns
    -------
    Interacting Chains
        List of chain pairs from PDB structure that have at least
        one pair of polymer atoms located < distance threshold apart.

    Examples
    --------
    # (1) Obtain list of interacting chains from 6YW7 structure
    >>> interacting_chains_list = get_interacting_chains_from_PDB('6YW7', '.', 6)
    Downloading PDB structure '6yw7'...
    >>> interacting_chains_list
    [['A', 'D'], ['A', 'E'], ['A', 'B'], ['D', 'F'], ['B', 'F'], ['B', 'G'], ['F', 'G'], ['F', 'C'], ['G', 'C']]
    '''
    model = _load_pdb_model(PDB_ID_structure_i, protein_structure_dir)
    return _direct_interaction_chain_pairs(model, dist_threshold)


def _direct_interaction_chain_pairs(model, dist_threshold, min_plddt=None):
    '''
    Internal helper: compute directly-interacting chain pairs for an
    already-loaded Bio.PDB model. Factored out of
    get_interacting_chains_from_PDB() so PDB_to_interacting_chains_uniprot_maps()
    can reuse a single downloaded/parsed structure instead of re-fetching it.

    If min_plddt is given (predicted structures only), atoms whose pLDDT
    (read from the B-factor column, see _plddt_scale_factor()) is below
    it are left out of the contact search, so low-confidence regions
    (e.g. disordered loops placed arbitrarily by the predictor) can't
    create spurious chain-chain contacts.
    '''
    plddt_scale = _plddt_scale_factor(model) if min_plddt is not None else None

    def keep(atom):
        if atom.get_parent().id[0] != ' ':
            return False
        return plddt_scale is None or atom.get_bfactor() * plddt_scale >= min_plddt

    # only consider chains that are (at least in part) protein or nucleic
    # acid polymers; pure ligand/water "chains" are not meaningful subunits
    chain_IDs = [chain.get_id() for chain in model
                 if classify_chain(chain) != 'other']

    # we want to test every pair of chains to see if they have any atoms
    # that are < 6 angstroms in distance
    possible_chain_pairs = list(itertools.combinations(chain_IDs, 2))

    chain_pairs_direct_interaction = []
    # iterate through all chain pairs and check to see if any atoms are close
    for chain_i_id, chain_j_id in possible_chain_pairs:

        # get chain objects from models
        chain_i = model[chain_i_id]
        chain_j = model[chain_j_id]

        # get polymer atoms from each chain (excludes waters/ions/ligands,
        # 'A' stands for ATOM in unfold_entities' entity-level codes)
        atom_list_i = [atom for atom in Selection.unfold_entities(chain_i, "A")
                       if keep(atom)]
        atom_list_j = [atom for atom in Selection.unfold_entities(chain_j, "A")
                       if keep(atom)]

        if not atom_list_i or not atom_list_j:
            continue

        # get the coordinates for the atom in each chain
        atom_coords_i = np.vstack([atom.coord for atom in atom_list_i])
        atom_coords_j = np.vstack([atom.coord for atom in atom_list_j])

        # compute pairwise distances betweeen all atoms from different chains
        dists = cdist(atom_coords_i, atom_coords_j)

        # if a pair of atoms < dist_threshold angstroms apart, store as interacting chains
        if np.sum(dists < dist_threshold) >= 1:
            chain_pairs_direct_interaction.append([chain_i_id, chain_j_id])

    return chain_pairs_direct_interaction

def make_request(url, mode, pdb_id):
    '''
    Make requests to PDBe API.
    
    This function can make GET and POST requests to the PDBe API.
    
    Parameters
    ----------
    url: str
    mode: str
    pdb_id: str
    
    Returns
    -------
        JSON or None
    '''
    if mode == "get":
        response = requests.get(url=url+pdb_id)
    elif mode == "post":
        response = requests.post(url, data=pdb_id)

    if response.status_code == 200:
        return response.json()
    else:
        print("[No data retrieved - %s] %s" 
              % (response.status_code, response.text))
    
    return None

def get_mappings_data(pdb_id):
    '''
    Get mappings data for PDB ID.
    
    This function will retreive the mappings data from
    the PDBe API using the make_request() function.
    
    Parameters
    ----------
    pdb_id: str
    
    Returns
    -------
        JSON of mappings or None
    '''
    # specify URL
    base_url = "https://www.ebi.ac.uk/pdbe/"
    api_base = base_url + "api/"
    uniprot_mapping_url = api_base + 'mappings/uniprot/'
    
    # Check if the provided PDB id is valid
    # There is no point in making an API call
    # with bad PDB ids
    if not re.match("[0-9][A-Za-z][A-Za-z0-9]{2}", pdb_id):
        print("Invalid PDB id")
        return None
    
    # GET the mappings data
    mappings_data = make_request(uniprot_mapping_url, "get", pdb_id)
    
    # Check if there is data
    if not mappings_data:
        print("No data found")
        return None
    
    return mappings_data

def list_uniprot_pdb_mappings(pdb_id):
    '''
    Get PDB chain to UniProt mappings.
    
    This function retrieves PDB > UniProt mappings using the 
    get_mappings_data() function, the parses the resulting 
    JSON to construct a dictionary where each key is a chain
    from the PDB structure, and the corresponding value for
    each is a list of UniProt IDs that map to the chain from 
    the SIFTS project.
    
    Parameters
    ----------
    pdb_id: str
    
    Returns
    -------
    Chain to UniProt Map
        Dictionary of PDB ID chain to UniProt ID mappings
        
    Examples
    --------
    # (1) Obtain a mapping of PDB ID 6YW7 chains to UniProt IDs
    >>> chain_to_UniProt_mapping_dict = list_uniprot_pdb_mappings('6YW7')
    >>> len(chain_to_UniProt_mapping_dict)
    7
    >>> sorted(chain_to_UniProt_mapping_dict.keys())
    ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    >>> sorted(chain_to_UniProt_mapping_dict.values())
    [['O15144'], ['O15145'], ['O15511'], ['P59998'], ['P61158'], ['P61160'], ['Q92747']]
    '''
    # convert to PDB id to lower case
    pdb_id = pdb_id.lower()
    
    # Getting the mappings data
    mappings_data = get_mappings_data(pdb_id)
    
    # If there is no data, return None
    if not mappings_data:
        return None
    
    # dictionary that stores UniProt > chain id mappings
    uniprot_chain_mapping_dict = {}
    
    uniprot = mappings_data[pdb_id]["UniProt"]
    for uniprot_id in uniprot.keys():
        mappings = uniprot[uniprot_id]["mappings"]
        
        # store the chain ids that correspond to this UniProt ID
        uniprot_chain_mapping_dict[uniprot_id] = []
        
        for mapping in mappings:
            entity_id = mapping["entity_id"]
            
            chain_id = mapping["chain_id"]
            uniprot_chain_mapping_dict[uniprot_id].append(chain_id)
            
            pdb_start = mapping["start"]["residue_number"]
            pdb_end = mapping["end"]["residue_number"]
            uniprot_start = mapping["unp_start"]
            uniprot_end = mapping["unp_end"]
        
    # "flip" the uniprot > pdb chain mapping
    # get all unique chain IDs
    chain_IDs = (list(set([item for sublist in 
                           uniprot_chain_mapping_dict.values() for item 
                           in sublist])))

    chain_uniprot_mapping_dict = {}
    # iterate through every chain ID
    for chain_i in chain_IDs:

        # iterate through every uniprot ID and check to 
        # see if chain ID is mapped
        chain_uniprot_mapping_dict[chain_i] = []
        for uniprot_i in uniprot_chain_mapping_dict.keys():
            if chain_i in uniprot_chain_mapping_dict[uniprot_i]:
                chain_uniprot_mapping_dict[chain_i].append(uniprot_i)
                
    return chain_uniprot_mapping_dict

def PDB_chains_to_uniprot(interacting_chains_list, 
                          chain_to_UniProt_mapping_dict):
    '''
    Get interacting chains from PDB structure mapped to UniProt IDs.
    
    This function takes the list of interacting chains from function
    get_interacting_chains_from_PDB() and the chain to UniProt mappings
    from function list_uniprot_pdb_mappings() and returns a list of
    interacting chains using UniProt IDs.

    Because the returned pairs are UniProt-labeled (one node per protein,
    not per chain), two chains of a homo-oligomer that map to the same
    UniProt ID produce a self-pair (e.g. a contact between chain I and
    chain J of the same protein) rather than a distinct-protein edge;
    such self-pairs are dropped, and duplicate pairs arising when two
    different chain-pairs resolve to the same UniProt-UniProt pair are
    collapsed to one. This means a direct contact that exists via only
    one of several chains mapping to the same UniProt ID is
    indistinguishable from one that exists via all of them.

    Parameters
    ----------
    Interacting Chains: list
    Chain to UniProt Map: dict

    Returns
    -------
    Interacting Chains
        List of unique, non-self interacting UniProt ID pairs.

    Examples
    --------
    # (1) Obtain list of interacting chains from 6YW7 structure
    # (2) Obtain a mapping of PDB ID 6YW7 chains to UniProt IDs
    # (3) Obtain list of interacting chains from 6YW7
    #     structure using UniProt IDs
    >>> interacting_chains_list = get_interacting_chains_from_PDB('6YW7', '.', 6)
    Downloading PDB structure '6yw7'...
    >>> chain_to_UniProt_mapping_dict = list_uniprot_pdb_mappings('6YW7')
    >>> interacting_UniProt_IDs = PDB_chains_to_uniprot(interacting_chains_list, chain_to_UniProt_mapping_dict)
    >>> sorted(interacting_UniProt_IDs)
    [['O15144', 'P59998'], ['O15511', 'Q92747'], ['P59998', 'O15511'], ['P59998', 'Q92747'], ['P61158', 'O15144'], ['P61158', 'O15145'], ['P61158', 'P61160'], ['P61160', 'O15511'], ['P61160', 'P59998']]
    '''
    interacting_UniProt_IDs = []
    seen_pairs = set()
    for interacting_chain_pair_i in interacting_chains_list:

        # get UniProt IDs that map to each chain ID
        chain_i_UniProts = (
            chain_to_UniProt_mapping_dict[interacting_chain_pair_i[0]])
        chain_j_UniProts = (
            chain_to_UniProt_mapping_dict[interacting_chain_pair_i[1]])

        # store every pair of UniProt IDs that correspond
        # to the interacting chains
        for chain_i_UniProt_ID in chain_i_UniProts:
            for chain_j_UniProt_ID in chain_j_UniProts:

                # skip self-pairs (contact between two chains of the same
                # homo-oligomeric protein -- not representable as a
                # distinct-protein edge) and duplicate pairs
                if chain_i_UniProt_ID == chain_j_UniProt_ID:
                    continue
                pair_key = frozenset((chain_i_UniProt_ID, chain_j_UniProt_ID))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                interacting_UniProt_IDs.append(
                    [chain_i_UniProt_ID,chain_j_UniProt_ID])

    return interacting_UniProt_IDs

def PDB_to_interacting_chains_uniprot_maps(PDB_ID,
                                           protein_structure_dir,
                                           interact_dist_threshold,
                                           chain_to_uniprot=None,
                                           min_plddt=None):
    '''
    Get interacting chains from PDB structure mapped to UniProt IDs and
    PDB chain to UniProt mappings.

    This is a wrapper function for functions
    (1) get_interacting_chains_from_PDB(),
    (2) list_uniprot_pdb_mappings(), and
    (3) PDB_chains_to_uniprot()
    to get a list of interacting chains from PDB structure using UniProt labels
    and the chain-to-UniProt mapping for this PDB structure.

    Nucleic acid chains (DNA/RNA) have no UniProt ID: SIFTS only maps
    protein chains. Such chains are instead labeled with a synthetic ID
    of the form 'RNA:<chain>' / 'DNA:<chain>' (e.g. 'RNA:A') so that they
    still appear as nodes in the returned interaction list rather than
    raising a KeyError or being silently dropped.

    Protein chains with no UniProt mapping (e.g. a chain SIFTS doesn't
    cover, or one left out of a user-supplied chain_to_uniprot) are
    labeled 'UNMAPPED:<chain>' in the same way, and a warning is issued.

    Parameters
    ----------
    PDB ID or path to a local structure file: str
    directory to store PDB file: str (ignored for a local file)
    distance threshold: int
    chain_to_uniprot: dict (optional)
        Chain ID -> UniProt ID (or list of IDs, or any free-text label,
        e.g. for a non-human or designed chain). Replaces the SIFTS
        lookup, which only exists for RCSB entries -- required for a
        local structure file (see map_chains_to_uniprot() to build one
        automatically by sequence). Isoform suffixes ('-2') are dropped,
        since BioPlex nodes are canonical accessions.
    min_plddt: float (optional)
        For predicted structures: ignore atoms with pLDDT below this
        (0-100 scale) when finding direct contacts. See
        _plddt_scale_factor(); skipped with a warning if the B-factor
        column doesn't hold pLDDT values.

    Returns
    -------
    Chain to UniProt Map
        Dictionary of PDB ID chain to UniProt/synthetic ID mappings
    Interacting Chains
        List of interacting chains using UniProt/synthetic IDs
    Chain Types
        Dictionary of PDB ID chain to classification
        ('protein', 'dna', 'rna', or 'other'), from classify_pdb_chains()

    Examples
    --------
    >>> (chain_to_UniProt_mapping_dict, interacting_UniProt_IDs, chain_types) = PDB_to_interacting_chains_uniprot_maps('6NMI', '.', 6)
    Downloading PDB structure '6nmi'...
    '''
    # load the structure once and reuse it for both the direct-interaction
    # search and the chain classification (avoids downloading it twice)
    model = _load_pdb_model(PDB_ID, protein_structure_dir)
    interacting_chains_list = _direct_interaction_chain_pairs(
        model, interact_dist_threshold, min_plddt=min_plddt)
    chain_types = _classify_chains(model)

    # get chain > UniProt ID mappings: user-supplied if given, otherwise
    # SIFTS (protein chains only). A local file has no SIFTS entry, so
    # it must come with a mapping.
    if chain_to_uniprot is not None:
        chain_to_UniProt_mapping_dict = _normalize_chain_mapping(
            chain_to_uniprot, chain_types)
    elif is_local_structure_file(PDB_ID):
        raise ValueError(
            f"'{PDB_ID}' is a local structure file, which has no SIFTS "
            "chain-to-UniProt mapping -- pass chain_to_uniprot (see "
            "map_chains_to_uniprot() to build one by sequence).")
    else:
        chain_to_UniProt_mapping_dict = list_uniprot_pdb_mappings(PDB_ID) or {}

    # nucleic acid chains have no UniProt ID (SIFTS is protein-only), and
    # a protein chain may be unmapped; give both a synthetic, still-unique
    # label so downstream lookups don't KeyError
    unmapped = []
    for chain_id, chain_type in chain_types.items():
        if chain_id in chain_to_UniProt_mapping_dict or chain_type == 'other':
            continue
        if chain_type in ('dna', 'rna'):
            chain_to_UniProt_mapping_dict[chain_id] = [f'{chain_type.upper()}:{chain_id}']
        else:
            chain_to_UniProt_mapping_dict[chain_id] = [f'UNMAPPED:{chain_id}']
            unmapped.append(chain_id)
    if unmapped:
        warnings.warn(f'No UniProt ID for protein chain(s) {unmapped}; '
                      "labeled 'UNMAPPED:<chain>' (no BioPlex data can match them).")

    # get list of chains pairs that interact in PDB structure using UniProts
    interacting_UniProt_IDs = PDB_chains_to_uniprot(interacting_chains_list,
                                                chain_to_UniProt_mapping_dict)

    return [chain_to_UniProt_mapping_dict, interacting_UniProt_IDs, chain_types]
def _normalize_chain_mapping(chain_to_uniprot, chain_types):
    '''
    Internal helper: validate and normalize a user-supplied chain ->
    UniProt mapping into the {chain: [ID, ...]} form list_uniprot_pdb_mappings()
    returns. Values may be a single ID or a list; isoform suffixes are
    dropped (BioPlex nodes are canonical accessions, so 'Q92747-2' would
    otherwise never match). Keys that aren't chains in the structure are
    an error (typically a typo, e.g. 'a' for 'A').
    '''
    unknown = [chain_id for chain_id in chain_to_uniprot if chain_id not in chain_types]
    if unknown:
        raise ValueError(f'chain_to_uniprot has chain IDs not in the structure: '
                         f'{unknown} (structure chains: {list(chain_types)})')
    normalized = {}
    for chain_id, ids in chain_to_uniprot.items():
        if isinstance(ids, str):
            ids = [ids]
        normalized[chain_id] = [_strip_isoform(id_i) for id_i in ids]
    return normalized


def _strip_isoform(id_i):
    '''
    Internal helper: 'Q92747-2' -> 'Q92747'. Leaves free-text labels
    (anything that isn't a UniProt accession with an isoform suffix)
    untouched.
    '''
    match = re.fullmatch(r'([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})-\d+', id_i)
    return match.group(1) if match else id_i


def get_chain_sequences(PDB_ID_structure_i, protein_structure_dir='.'):
    '''
    Get the amino acid sequence of every protein chain in a structure,
    read from its modeled residues (in chain order, so any unmodeled
    stretch is simply absent). Modified residues are converted to their
    parent amino acid (e.g. selenomethionine MSE -> M); anything
    unrecognized becomes 'X'.

    Parameters
    ----------
    PDB ID or path to a local structure file: str
    directory to store PDB file: str (optional, ignored for a local file)

    Returns
    -------
    dict
        Mapping of protein chain ID -> one-letter sequence.

    Examples
    --------
    >>> chain_seqs = get_chain_sequences('6YW7', '.')
    Downloading PDB structure '6yw7'...
    >>> sorted(chain_seqs.keys())
    ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    '''
    from Bio.Data.PDBData import protein_letters_3to1_extended

    model = _load_pdb_model(PDB_ID_structure_i, protein_structure_dir)
    chain_seqs = {}
    for chain in model:
        if classify_chain(chain) != 'protein':
            continue
        chain_seqs[chain.get_id()] = ''.join(
            protein_letters_3to1_extended.get(residue.get_resname().strip().upper(), 'X')
            for residue in chain if residue.id[0] == ' ')
    return chain_seqs


def fetch_uniprot_sequences(uniprot_IDs_list):
    '''
    Download the sequences of the given UniProt accessions from the
    UniProt REST API (rest.uniprot.org). Isoform accessions (e.g.
    'Q92747-2') fetch that isoform's sequence. Only public accessions are
    sent; no structure data leaves the machine.

    Parameters
    ----------
    UniProt IDs: list

    Returns
    -------
    dict
        Mapping of accession -> sequence.

    Examples
    --------
    >>> seqs = fetch_uniprot_sequences(['P61160'])
    >>> len(seqs['P61160'])
    394
    '''
    sequences = {}
    for uniprot_id in uniprot_IDs_list:
        response = requests.get(f'https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta')
        if response.status_code != 200 or not response.text.startswith('>'):
            raise ValueError(f"Could not fetch UniProt sequence for '{uniprot_id}' "
                             f'(HTTP {response.status_code}).')
        sequences[uniprot_id] = ''.join(response.text.strip().split('\n')[1:])
    return sequences


def map_chains_to_uniprot(PDB_ID_structure_i, uniprot_IDs_list,
                          protein_structure_dir='.', min_identity=0.9,
                          min_coverage=0.5, uniprot_sequences=None):
    '''
    Work out which of a given set of UniProt proteins each protein chain
    of a structure is, by sequence -- for structures that have no SIFTS
    mapping, e.g. a user's own experimental model or an AlphaFold3/Boltz/
    ColabFold prediction. The result can be passed straight to
    PDB_to_interacting_chains_uniprot_maps(chain_to_uniprot=...) or the
    Figure 2 render functions.

    Each chain is compared with every candidate: an exact substring match
    first (the usual case for predictions, which are folded from the
    UniProt sequence), then a local alignment (BLOSUM62) for experimental
    constructs with truncations, tags, mutations, or a non-human ortholog.
    Unknown residues (UNK/'X', e.g. backbone traced in a cryo-EM map
    without a sequence assignment) carry no identity information and are
    left out of the comparison; their count is reported.
    The best-scoring candidate (identity x coverage) is accepted if it
    passes both thresholds. Chains are matched independently, so several
    chains (a homo-oligomer) can map to the same protein.

    Parameters
    ----------
    PDB ID or path to a local structure file: str
    UniProt IDs of the candidate proteins: list
    directory to store PDB file: str (optional, ignored for a local file)
    min_identity: float (optional)
        Minimum fraction of aligned chain residues identical to the
        UniProt sequence.
    min_coverage: float (optional)
        Minimum fraction of the chain's modeled (non-UNK) residues that align to
        the UniProt sequence (a long tag or fusion lowers this).
    uniprot_sequences: dict (optional)
        Accession -> sequence, to skip downloading them (e.g. offline).

    Returns
    -------
    dict
        Chain ID -> [UniProt ID] for every chain that passed.
    Pandas DataFrame
        One row per protein chain: the best match and runner-up, with
        identity, chain coverage, UniProt coverage, whether it was
        accepted, and how the match was made ('exact'/'alignment').
        Check this for anything unexpected before trusting the mapping.

    Examples
    --------
    >>> chain_map, report = map_chains_to_uniprot('6YW7', ['P61158', 'P61160', 'Q92747', 'O15144', 'O15145', 'P59998', 'O15511'], '.')
    Downloading PDB structure '6yw7'...
    >>> chain_map['C']
    ['Q92747']
    '''
    from Bio.Align import PairwiseAligner, substitution_matrices

    chain_seqs = get_chain_sequences(PDB_ID_structure_i, protein_structure_dir)
    if uniprot_sequences is None:
        uniprot_sequences = fetch_uniprot_sequences(uniprot_IDs_list)

    aligner = PairwiseAligner(mode='local',
                              substitution_matrix=substitution_matrices.load('BLOSUM62'),
                              open_gap_score=-10, extend_gap_score=-0.5)

    def compare(chain_seq, uniprot_seq):
        if not chain_seq:  # chain is entirely unknown residues
            return 0.0, 0.0, 0.0, None
        if chain_seq in uniprot_seq:
            return 1.0, 1.0, len(chain_seq) / len(uniprot_seq), 'exact'
        alignment = next(iter(aligner.align(chain_seq, uniprot_seq)))
        counts = alignment.counts()
        n_aligned = counts.identities + counts.mismatches
        if n_aligned == 0:
            return 0.0, 0.0, 0.0, 'alignment'
        uniprot_aligned = sum(end - start for start, end in alignment.aligned[1])
        return (counts.identities / n_aligned, n_aligned / len(chain_seq),
                uniprot_aligned / len(uniprot_seq), 'alignment')

    chain_map = {}
    report_rows = []
    for chain_id, chain_seq_raw in chain_seqs.items():
        chain_seq = chain_seq_raw.replace('X', '')
        scored = []
        for uniprot_id in uniprot_IDs_list:
            identity, chain_cov, uniprot_cov, method = compare(
                chain_seq, uniprot_sequences[uniprot_id])
            scored.append((identity * chain_cov, uniprot_id, identity, chain_cov,
                           uniprot_cov, method))
        scored.sort(reverse=True)
        best = scored[0] if scored else (0.0, None, 0.0, 0.0, 0.0, None)
        runner_up = scored[1] if len(scored) > 1 else (0.0, None, 0.0, 0.0, 0.0, None)
        accepted = (best[1] is not None and best[2] >= min_identity
                    and best[3] >= min_coverage)
        if accepted:
            chain_map[chain_id] = [_strip_isoform(best[1])]
        report_rows.append(dict(
            chain=chain_id, chain_length=len(chain_seq_raw),
            unknown_residues=len(chain_seq_raw) - len(chain_seq), uniprot=best[1],
            identity=round(best[2], 3), chain_coverage=round(best[3], 3),
            uniprot_coverage=round(best[4], 3), method=best[5], accepted=accepted,
            runner_up=runner_up[1], runner_up_identity=round(runner_up[2], 3),
            runner_up_chain_coverage=round(runner_up[3], 3)))

    return chain_map, pd.DataFrame(report_rows)


def _bioplex_edges_and_roles(bp_PPI_df, restrict_to=None):
    '''
    Internal helper: the undirected set of BioPlex-detected pairs
    (isoform suffixes stripped, as frozensets), plus the sets of baits
    (UniprotA) and preys (UniprotB). If restrict_to is given, only pairs
    with both proteins in it are kept. Shared by the Figure 2 BioPlex
    panels and compare_structure_contacts_to_BioPlex().
    '''
    restrict_to = set(restrict_to) if restrict_to is not None else None
    edges, baits, preys = set(), set(), set()
    for uniprot_A, uniprot_B in zip(bp_PPI_df.UniprotA, bp_PPI_df.UniprotB):
        uniprot_A = uniprot_A.split('-')[0]
        uniprot_B = uniprot_B.split('-')[0]
        if restrict_to is not None and not (uniprot_A in restrict_to
                                            and uniprot_B in restrict_to):
            continue
        edges.add(frozenset((uniprot_A, uniprot_B)))
        baits.add(uniprot_A)
        preys.add(uniprot_B)
    return edges, baits, preys


def _bioplex_symbol_lookup(*bp_PPI_dfs):
    '''
    Internal helper: UniProt accession (isoform suffix stripped) -> gene
    symbol, taken from BioPlex DataFrames.
    '''
    symbol_lookup = {}
    for df in bp_PPI_dfs:
        for uniprot_A, symbol_A in zip(df.UniprotA, df.SymbolA):
            symbol_lookup[uniprot_A.split('-')[0]] = symbol_A
        for uniprot_B, symbol_B in zip(df.UniprotB, df.SymbolB):
            symbol_lookup[uniprot_B.split('-')[0]] = symbol_B
    return symbol_lookup


def compare_structure_contacts_to_BioPlex(chain_to_UniProt_mapping_dict,
                                          interacting_UniProt_IDs, chain_types,
                                          bp_293t_df, bp_hct116_df):
    '''
    Tabulate how a structure's direct protein-protein contacts line up
    with BioPlex AP-MS interactions in both cell lines -- the numbers
    behind the Figure 2-style panels, for scripted/batch use.

    Takes the three outputs of PDB_to_interacting_chains_uniprot_maps()
    (which accepts either a PDB ID or a local structure file).

    Parameters
    ----------
    Chain to UniProt Map: dict
    Interacting UniProt/synthetic IDs: list
    Chain Types: dict
    DataFrame of 293T PPIs: Pandas DataFrame (from getBioPlex('293T', ...))
    DataFrame of HCT116 PPIs: Pandas DataFrame (from getBioPlex('HCT116', ...))

    Returns
    -------
    Pandas DataFrame
        One row per pair of proteins in the structure that is a direct
        contact in the structure, a BioPlex interaction in either cell
        line, or both. Columns: UniprotA, UniprotB, SymbolA, SymbolB,
        structure_contact, bioplex_293T, bioplex_HCT116. Nucleic acid and
        unmapped chains are left out (AP-MS can't detect them).

    Examples
    --------
    >>> from bioplexpy import getBioPlex
    >>> bp_293t_df = getBioPlex('293T', '3.0')
    >>> bp_hct116_df = getBioPlex('HCT116', '1.0')
    >>> maps = PDB_to_interacting_chains_uniprot_maps('6YW7', '.', 6)
    Downloading PDB structure '6yw7'...
    >>> contacts_df = compare_structure_contacts_to_BioPlex(*maps, bp_293t_df, bp_hct116_df)
    >>> int(contacts_df.structure_contact.sum())
    9
    '''
    protein_ids = sorted({id_i for chain_id, ids in chain_to_UniProt_mapping_dict.items()
                          if chain_types.get(chain_id, 'protein') == 'protein'
                          for id_i in ids if not id_i.startswith('UNMAPPED:')})
    contacts = {frozenset(pair) for pair in interacting_UniProt_IDs
                if pair[0] in protein_ids and pair[1] in protein_ids}
    edges_293t, _, _ = _bioplex_edges_and_roles(bp_293t_df, restrict_to=protein_ids)
    edges_hct116, _, _ = _bioplex_edges_and_roles(bp_hct116_df, restrict_to=protein_ids)
    symbols = _bioplex_symbol_lookup(bp_293t_df, bp_hct116_df)

    rows = []
    for pair in contacts | edges_293t | edges_hct116:
        if len(pair) != 2:
            continue
        uniprot_A, uniprot_B = sorted(pair)
        rows.append(dict(UniprotA=uniprot_A, UniprotB=uniprot_B,
                         SymbolA=symbols.get(uniprot_A, uniprot_A),
                         SymbolB=symbols.get(uniprot_B, uniprot_B),
                         structure_contact=pair in contacts,
                         bioplex_293T=pair in edges_293t,
                         bioplex_HCT116=pair in edges_hct116))
    columns = ['UniprotA', 'UniprotB', 'SymbolA', 'SymbolB', 'structure_contact',
               'bioplex_293T', 'bioplex_HCT116']
    return (pd.DataFrame(rows, columns=columns)
            .sort_values(['structure_contact', 'SymbolA', 'SymbolB'],
                         ascending=[False, True, True])
            .reset_index(drop=True))
