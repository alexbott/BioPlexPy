#!/usr/bin/env python

import itertools
import os
import re
import tempfile
import warnings

import matplotlib.colors
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import py3Dmol


def display_PPI_network_for_complex(ax, bp_PPI_df, Corum_DF, Complex_ID, 
    node_size, edge_width, node_font_size = 10, bait_node_color='xkcd:red', 
    prey_node_color='xkcd:rose pink', AP_MS_edge_color='xkcd:red',
    node_pos=False):
    '''
    Display network of BioPlex PPIs for a CORUM complex.
    
    This function displays a complete network in which nodes represent 
    the proteins in a specified CORUM complex and edges represent 
    BioPlex PPIs using NetworkX. Edges detected through AP-MS are
    colored darker.

    Parameters
    ----------
    ax object to draw on: Matplotlib Axes
    DataFrame of PPIs : Pandas DataFrame
    DataFrame of CORUM complexes : Pandas DataFrame
    Corum Complex ID: int
    Size of Nodes in Network: int
    Width of Edges in Network: float
    Size of font for Node Labels: int (optional)
    Color of Nodes targeted as baits: str (optional)
    Color of Nodes detected as preys only: str (optional)
    Color of Edges observed via AP-MS from PPI data: str (optional)
    Networkx Position of Nodes: dict (optional)

    Returns
    -------
    Node Positions
        Dictionary of Node Positions in NetworkX layout

    Examples
    --------
    # (1) Obtain the latest version of the 293T PPI network
    # (2) Obtain CORUM complexes
    # (3) create figure and axis objects to draw on
    # (4) Visualize network for specified protein complex
    #     using PPI data (ING2 complex ID: 2851)
    >>> bp_PPI_df = getBioPlex('293T', '3.0')
    >>> Corum_DF = getCorum()
    >>> fig, ax = plt.subplots()
    >>> ING2_node_layout = display_PPI_network_for_complex(ax, bp_PPI_df, Corum_DF, 2851, 2300, 3.5)
    >>> ING2_node_layout
    {'A0A024R3R1': array([ 1.00000000e+00, -2.29248628e-09]), 'O75446': array([0.88545603, 0.46472316]), 'P29374': array([0.56806476, 0.82298384]), 'Q09028': array([0.12053671, 0.99270884]), 'Q13547': array([-0.35460481,  0.93501625]), 'Q16576': array([-0.74851068,  0.66312264]), 'Q5PSV4': array([-0.9709418 ,  0.23931567]), 'Q92769': array([-0.9709418 , -0.23931561]), 'Q96ST3': array([-0.7485108 , -0.66312258]), 'Q9H0E3': array([-0.35460499, -0.9350162 ]), 'Q9H160': array([ 0.12053676, -0.99270884]), 'Q9H7L9': array([ 0.56806458, -0.82298396]), 'Q9HCU9': array([ 0.88545603, -0.46472319])}
    '''
    # store uniprot IDs & gene symbols that belong to this complex in a list
    genes_in_complex_i = (Corum_DF[Corum_DF.complex_id == Complex_ID]
                            .loc[:,'subunits_uniprot_id'].values[0]
                            .split(';')) # Uniprot
    
    gene_symbols_in_complex_i = (Corum_DF[Corum_DF.complex_id == Complex_ID]
                                 .loc[:,'subunits_gene_name'].values[0]
                                 .split(';')) # Symbol

    # filter BioPlex PPI dataframe to include only interactions 
    # where both genes are found in complex
    complex_i_PPI_filter = []
    for uniprot_A, uniprot_B in zip(bp_PPI_df.UniprotA, bp_PPI_df.UniprotB):
        
        # check for isoform IDs and adjust
        if '-' in uniprot_A:
            uniprot_A = uniprot_A.split('-')[0]
        if '-' in uniprot_B:
            uniprot_B = uniprot_B.split('-')[0]

        # check to see if both gene symbols for this 
        # interaction are genes in complex
        if ((uniprot_A in genes_in_complex_i) and 
            (uniprot_B in genes_in_complex_i)):
            complex_i_PPI_filter.append(True)
        else:
            complex_i_PPI_filter.append(False)

    complex_i_PPI_filter = np.array(complex_i_PPI_filter)
    # use filter to subset bp PPI dataframe
    bp_complex_i_df = bp_PPI_df[complex_i_PPI_filter]
    bp_complex_i_df.reset_index(inplace = True, drop = True) # reset index

    # reconstruct UniprotA/UniprotB columns without '-' isoform id
    UniprotA_new = []
    UniprotB_new = []
    for UniprotA, UniprotB in zip(bp_complex_i_df.UniprotA,
                                  bp_complex_i_df.UniprotB):

        if '-' in UniprotA:
            UniprotA_new.append(UniprotA.split('-')[0])
        else:
            UniprotA_new.append(UniprotA)

        if '-' in UniprotB:
            UniprotB_new.append(UniprotB.split('-')[0])
        else:
            UniprotB_new.append(UniprotB)
            
    # update columns for Uniprot source & 
    # Uniprot target to exclude isoform '-' ID
    bp_complex_i_df.loc[:,'UniprotA'] = UniprotA_new
    bp_complex_i_df.loc[:,'UniprotB'] = UniprotB_new
    
    # subset PPI dataframe to the cols we need to construct graph
    bp_complex_i_df = (bp_complex_i_df.loc[:,
                        ['UniprotA','UniprotB','SymbolA','SymbolB']])
    
    # create a graph from the nodes/genes of specified complex
    bp_complex_i_G = nx.Graph()
    bp_complex_i_G.add_nodes_from(genes_in_complex_i)
    
    # iterate over AP-MS interactions in PPI df and add edges
    for source, target in zip(bp_complex_i_df.UniprotA, 
                              bp_complex_i_df.UniprotB):
        bp_complex_i_G.add_edge(source, target)
        
    # get mapping uniprot -> symbol & store as node attribute 
    # from CORUM complex data
    uniprot_symbol_dict = dict([key,val] for key, val in 
                            zip(genes_in_complex_i, gene_symbols_in_complex_i))
    for node_i in bp_complex_i_G.nodes():
        bp_complex_i_G.nodes[node_i]["symbol"] = uniprot_symbol_dict[node_i]

    # get a list of genes that were identifed as "baits" 
    # and "preys" for coloring nodes
    bp_complex_i_baits = list(set(bp_complex_i_df.UniprotA))
    bp_complex_i_preys = list(set(bp_complex_i_df.UniprotB))

    labels = dict([(key,val) for key, val in zip(list(bp_complex_i_G.nodes), 
                [bp_complex_i_G.nodes[node_i]['symbol'] for 
                node_i in bp_complex_i_G.nodes])])

    # color nodes according to whether they were present among "baits" 
    # & "preys", just "baits", just "preys" or not detected in PPI data
    # for this complex
    node_color_map = []
    for node_i_uniprot in bp_complex_i_G.nodes:

        # gene is present among baits & preys for the PPIs 
        # detected in this complex
        if ((node_i_uniprot in bp_complex_i_baits) and 
            (node_i_uniprot in bp_complex_i_preys)):
            node_color_map.append(bait_node_color)

        # gene is present among baits but NOT preys for 
        # the PPIs detected in this complex
        elif ((node_i_uniprot in bp_complex_i_baits) and 
              (node_i_uniprot not in bp_complex_i_preys)):
            node_color_map.append(bait_node_color)

        # gene is NOT present among baits but is present among 
        # preys for the PPIs detected in this complex
        elif ((node_i_uniprot not in bp_complex_i_baits) and 
              (node_i_uniprot in bp_complex_i_preys)):
            node_color_map.append(prey_node_color)

        # gene is NOT present among baits and is NOT present among 
        # preys for the PPIs detected in this complex
        elif ((node_i_uniprot not in bp_complex_i_baits) and 
              (node_i_uniprot not in bp_complex_i_preys)):
            node_color_map.append('0.7')

    # create a complete graph from the nodes of complex graph to add in all 
    # "background" edges (edges detected with AP-MS will be colored over)
    # position will be the same for both graphs since nodes are the same
    bp_complex_i_G_complete = nx.Graph()
    bp_complex_i_G_complete.add_nodes_from(bp_complex_i_G.nodes)
    bp_complex_i_G_complete.add_edges_from(
        itertools.combinations(bp_complex_i_G.nodes, 2))

    # optional argument "node_pos" used here
    # check to see if node position object has been fed as an argument
    if node_pos == False:
        # DEFAULT: set position of nodes w/ circular layout
        pos = nx.circular_layout(bp_complex_i_G)
    else:
        pos = node_pos # use node positions fed into function

    # construct edges for COMPLETE graph for "background" edges
    edges_complete = nx.draw_networkx_edges(bp_complex_i_G_complete, 
                                            pos, width = edge_width, 
                                            alpha = 0.25, ax = ax)
    edges_complete.set_edgecolor("xkcd:grey")

    # construct edges
    edges = nx.draw_networkx_edges(bp_complex_i_G, pos, 
                                   width = edge_width, ax = ax)
    edges.set_edgecolor(AP_MS_edge_color)

    # construct nodes
    nodes = nx.draw_networkx_nodes(bp_complex_i_G, pos, 
                                   node_size = node_size, 
                                   node_color = node_color_map, 
                                   ax = ax)
    nodes.set_edgecolor("xkcd:black")
    nodes.set_linewidth(1.5)
    nx.draw_networkx_labels(bp_complex_i_G, pos, labels = labels, 
                            font_size = node_font_size, font_weight = 'bold', 
                            font_color = 'xkcd:white', ax = ax)

    # return node position layout
    return pos


def display_PDB_network_for_complex(ax, chain_to_UniProt_mapping_dict, 
    interacting_UniProt_IDs, node_size, edge_width, node_font_size=10):
    '''
    Display network of interacting chains.
    
    This function displays a complete network in which nodes represent the 
    proteins in a specified PDB structure, and edges represent chains in that
    structure, using NetworkX. Edges that are classified as interacting
    (are < dist_threshold angstroms apart) are colored black.

    Parameters
    ----------
    ax object to draw on: Matplotlib Axes
    Mapping of Chains to UniProt IDs: dictionary
    List of Interacting Chains: list
    Size of Nodes in Network: int
    Width of Edges in Network: float
    Size of font for Node Labels: int (optional)

    Returns
    -------
    Node Positions
        Dictionary of Node Positions in NetworkX layout
    Interacting Network Edges
        List of Edges for Interacting Nodes
    Number of Network Edges
        Float of the Number of Possible Interacting Edges
    
    # what order are they returned?

    Examples
    --------
    # (1) Obtain list of interacting chains from 6YW7 structure
    # (2) Obtain a mapping of PDB ID 6YW7 chains to UniProt IDs
    # (3) Obtain list of interacting chains from 6YW7 
    # structure using UniProt IDs
    # (4) create figure and axis objects to draw on
    # (5) Visualize interacting chains using Uniprot IDs

    >>> interacting_chains_list = get_interacting_chains_from_PDB('6YW7', '.', 6)
    Downloading PDB structure '6YW7'...
    >>> chain_to_UniProt_mapping_dict = list_uniprot_pdb_mappings('6YW7')
    >>> interacting_UniProt_IDs = PDB_chains_to_uniprot(interacting_chains_list,chain_to_UniProt_mapping_dict)
    >>> fig, ax = plt.subplots()
    >>> node_layout_pdb, edges_list_pdb, num_possible_edges_pdb = display_PDB_network_for_complex(ax,chain_to_UniProt_mapping_dict, interacting_UniProt_IDs, 1500, 3, node_font_size = 8)
    >>> len(node_layout_pdb)
    7
    >>> len(edges_list_pdb)
    9
    >>> num_possible_edges_pdb
    21.0
    '''
    # create connected graph from all uniprot IDs
    chain_uniprot_IDs = ([chain_uniprot_i[0] for chain_uniprot_i in 
                          chain_to_UniProt_mapping_dict.values()])

    # create a graph from the nodes/genes of this complex structure
    pdb_structure_i_G = nx.Graph()
    pdb_structure_i_G.add_nodes_from(chain_uniprot_IDs)

    # iterate over physical interactions and add edges
    for chain_pair_interacting in interacting_UniProt_IDs:
        pdb_structure_i_G.add_edge(chain_pair_interacting[0], 
                                   chain_pair_interacting[1])

    # create a complete graph from the nodes of complex graph to add in all 
    # "background" edges (edges detected with AP-MS will be colored over)
    # position will be the same for both graphs since nodes are the same
    pdb_structure_i_G_complete = nx.Graph()
    pdb_structure_i_G_complete.add_nodes_from(pdb_structure_i_G.nodes)
    pdb_structure_i_G_complete.add_edges_from(
        itertools.combinations(pdb_structure_i_G.nodes, 2))

    # set position of nodes w/ circular layout
    pos = nx.circular_layout(pdb_structure_i_G)

    # construct edges for COMPLETE graph for "background" edges
    edges_complete = nx.draw_networkx_edges(pdb_structure_i_G_complete, 
                                pos, width = edge_width, alpha = 0.25, ax = ax)
    edges_complete.set_edgecolor("xkcd:grey")

    # construct edges
    edges = nx.draw_networkx_edges(pdb_structure_i_G, pos, 
                                   width = edge_width, ax = ax)
    edges.set_edgecolor('xkcd:blue')

    # construct nodes
    nodes = nx.draw_networkx_nodes(pdb_structure_i_G, pos, 
                                   node_size = node_size, 
                                   node_color = 'xkcd:black', ax = ax)
    nodes.set_edgecolor("xkcd:black")
    nodes.set_linewidth(1.5)
    nx.draw_networkx_labels(pdb_structure_i_G, pos, font_size = node_font_size,
                            font_weight = 'bold', font_color = 'xkcd:white', 
                            ax = ax)

    # return node position layout, list of edges detected, 
    # and number of possible edges
    return [pos, pdb_structure_i_G.edges, 
            float(len(pdb_structure_i_G_complete.edges))]

def display_PPI_network_match_PDB(ax, chain_to_UniProt_mapping_dict, 
    interacting_UniProt_IDs, bp_PPI_df, node_pos, node_size, edge_width, 
    node_font_size=10, bait_node_color='xkcd:red', 
    prey_node_color='xkcd:rose pink', AP_MS_edge_color='xkcd:red'):
    '''
    Display network of BioPlex PPIs for a set of interacting UniProt IDs.
    
    This function displays a complete network in which nodes represent the 
    proteins in a specified PDB structure, and edges represent chains in that
    structure, using NetworkX. Edges that are classified as interacting from
    BioPlex PPI data (detected through AP-MS) are colored darker.

    Parameters
    ----------
    ax object to draw on: Matplotlib Axes
    Mapping of Chains to UniProt IDs: dictionary
    List of Interacting Chains: list
    DataFrame of PPIs : Pandas DataFrame
    Networkx Position of Nodes: dict
    Size of Nodes in Network: int
    Width of Edges in Network: float
    Size of font for Node Labels: int (optional)
    Color of Nodes targeted as baits: str (optional)
    Color of Nodes detected as preys only: str (optional)
    Color of Edges observed via AP-MS from PPI data: str (optional)

    Returns
    -------
    Interacting Network Edges
        List of Edges for Interacting Nodes
    Number of Network Edges
        Float of the Number of Possible Interacting Edges
    

    Examples
    --------
    # (1) Obtain list of interacting chains from 6YW7 structure
    # (2) Obtain a mapping of PDB ID 6YW7 chains to UniProt IDs
    # (3) Obtain list of interacting chains from 6YW7 
    #     structure using UniProt IDs
    # (4) create figure and axis objects to draw on
    # (5) Visualize interacting chains using Uniprot IDs
    # (6) Get BioPlex PPI data
    # (7) create figure and axis objects to draw on
    # (8) Visualize BioPlex PPI interactions using layout 
    #     from interacting chains
    >>> interacting_chains_list = get_interacting_chains_from_PDB('6YW7', '.', 6)
    Downloading PDB structure '6YW7'...
    >>> chain_to_UniProt_mapping_dict = list_uniprot_pdb_mappings('6YW7')
    >>> interacting_UniProt_IDs = PDB_chains_to_uniprot(interacting_chains_list,chain_to_UniProt_mapping_dict)
    >>> fig, ax1 = plt.subplots()
    >>> node_layout_pdb, edges_list_pdb, num_possible_edges_pdb = display_PDB_network_for_complex(ax1, chain_to_UniProt_mapping_dict, interacting_UniProt_IDs, 1500, 3, node_font_size = 8)
    >>> bp_PPI_df = getBioPlex('293T', '3.0')
    >>> fig, ax2 = plt.subplots()
    >>> edges_list_bp, num_possible_edges_bp = display_PPI_network_match_PDB(ax2, chain_to_UniProt_mapping_dict, interacting_UniProt_IDs, bp_PPI_df, node_layout_pdb, 1500, 3, node_font_size = 8)
    >>> len(edges_list_bp)
    13
    >>> num_possible_edges_bp
    21.0
    '''
    # create connected graph from all uniprot IDs
    chain_uniprot_IDs = ([chain_uniprot_i[0] for chain_uniprot_i 
                          in chain_to_UniProt_mapping_dict.values()])
    
    # filter BioPlex PPI dataframe to include only interactions 
    # where both genes are found in complex
    structure_uniprots_i_PPI_filter = []
    for uniprot_A, uniprot_B in zip(bp_PPI_df.UniprotA, bp_PPI_df.UniprotB):

        # check for isoform IDs and adjust
        if '-' in uniprot_A:
            uniprot_A = uniprot_A.split('-')[0]
        if '-' in uniprot_B:
            uniprot_B = uniprot_B.split('-')[0]

        # check to see if both gene symbols for this 
        # interaction are genes in complex
        if ((uniprot_A in chain_uniprot_IDs) and 
            (uniprot_B in chain_uniprot_IDs)):
            structure_uniprots_i_PPI_filter.append(True)
        else:
            structure_uniprots_i_PPI_filter.append(False)

    structure_uniprots_i_PPI_filter = np.array(structure_uniprots_i_PPI_filter)
    # use filter to subset bp PPI dataframe
    bp_structure_i_df = bp_PPI_df[structure_uniprots_i_PPI_filter]
    bp_structure_i_df.reset_index(inplace = True, drop = True) # reset index

    # reconstruct UniprotA/UniprotB columns without '-' isoform id
    UniprotA_new = []
    UniprotB_new = []
    for UniprotA, UniprotB in zip(bp_structure_i_df.UniprotA, 
                                  bp_structure_i_df.UniprotB):
        if '-' in UniprotA:
            UniprotA_new.append(UniprotA.split('-')[0])
        else:
            UniprotA_new.append(UniprotA)

        if '-' in UniprotB:
            UniprotB_new.append(UniprotB.split('-')[0])
        else:
            UniprotB_new.append(UniprotB)

    # update columns for Uniprot source & 
    # Uniprot target to exclude isoform '-' ID
    bp_structure_i_df.loc[:,'UniprotA'] = UniprotA_new
    bp_structure_i_df.loc[:,'UniprotB'] = UniprotB_new

    # subset PPI dataframe to the cols we need to construct graph
    bp_structure_i_df = (bp_structure_i_df.loc[:,
                        ['UniprotA','UniprotB','SymbolA','SymbolB']])

    # create a graph from the nodes/genes of specified complex
    bp_structure_i_G = nx.Graph()
    bp_structure_i_G.add_nodes_from(chain_uniprot_IDs)

    # iterate over AP-MS interactions in PPI df and add edges
    for source, target in zip(bp_structure_i_df.UniprotA, 
                              bp_structure_i_df.UniprotB):
        bp_structure_i_G.add_edge(source, target)

    # get a list of genes that were identifed as "baits" 
    # and "preys" for coloring nodes
    bp_structure_i_baits = list(set(bp_structure_i_df.UniprotA))
    bp_structure_i_preys = list(set(bp_structure_i_df.UniprotB))

    # color nodes according to whether they were present among "baits"
    # & "preys", just "baits", just "preys" or not detected 
    # in PPI data for this structure
    node_color_map = []
    for node_i_uniprot in bp_structure_i_G.nodes:

        # gene is present among baits & preys for the PPIs 
        # detected in this structure
        if ((node_i_uniprot in bp_structure_i_baits) and 
            (node_i_uniprot in bp_structure_i_preys)):
            node_color_map.append(bait_node_color)

        # gene is present among baits but NOT preys for the PPIs 
        # detected in this structure
        elif ((node_i_uniprot in bp_structure_i_baits) and 
              (node_i_uniprot not in bp_structure_i_preys)):
            node_color_map.append(bait_node_color)

        # gene is NOT present among baits but is present among preys for 
        # the PPIs detected in this structure
        elif ((node_i_uniprot not in bp_structure_i_baits) and 
              (node_i_uniprot in bp_structure_i_preys)):
            node_color_map.append(prey_node_color)

        # gene is NOT present among baits and is NOT present among preys for 
        # the PPIs detected in this complex
        elif ((node_i_uniprot not in bp_structure_i_baits) and 
              (node_i_uniprot not in bp_structure_i_preys)):
            node_color_map.append('0.7')
            
    # can use the complete graph from the pdb direct 
    # interaction graph since they have the same nodes
    # create a complete graph from the nodes of complex graph to 
    # add in all "background" edges (edges 
    # detected with AP-MS will be colored over)
    # position will be the same for both graphs since nodes are the same
    bp_structure_i_G_complete = nx.Graph()
    bp_structure_i_G_complete.add_nodes_from(bp_structure_i_G.nodes)
    bp_structure_i_G_complete.add_edges_from(
        itertools.combinations(bp_structure_i_G.nodes, 2))

    # construct edges for COMPLETE graph for "background" edges
    edges_complete = nx.draw_networkx_edges(bp_structure_i_G_complete, 
                                            node_pos, width = edge_width,
                                            alpha = 0.25, ax = ax)
    edges_complete.set_edgecolor("xkcd:grey")

    # construct edges
    edges = nx.draw_networkx_edges(bp_structure_i_G, node_pos, 
                                   width = edge_width, ax = ax)
    edges.set_edgecolor(AP_MS_edge_color)

    # construct nodes
    nodes = nx.draw_networkx_nodes(bp_structure_i_G, node_pos, 
                                   node_size = node_size, 
                                   node_color = node_color_map, ax = ax)
    nodes.set_edgecolor("xkcd:black")
    nodes.set_linewidth(1.5)
    nx.draw_networkx_labels(bp_structure_i_G, node_pos, 
                            font_size = node_font_size, 
                            font_weight = 'bold', font_color = 'xkcd:white', 
                            ax = ax)

    # return node position layout, list of edges detected,
    # and number of possible edges
    return [bp_structure_i_G.edges, float(len(bp_structure_i_G_complete.edges))]


# ---------------------------------------------------------------------------
# Functions below reproduce Figure 2F-H of Huttlin et al. 2021 (Cell
# 184:3022-3040): for a given PDB structure, a chain-colored 3D render
# (get_chain_color_palette + render_pdb_structure_py3Dmol), the PDB-derived
# direct interaction network colored to match (display_PDB_direct_interaction_network),
# that same network recolored by BioPlex AP-MS detection
# (display_BioPlex_direct_interactions), and the full BioPlex subnetwork
# across both 293T and HCT116 cell lines
# (display_All_BioPlex_interactions_two_cell_lines). render_figure2_panels()
# assembles the three network panels into one figure and returns the
# py3Dmol structure view alongside it.
# ---------------------------------------------------------------------------

# Mol*'s default "Chain ID" color theme -- the viewer RCSB uses at
# rcsb.org. A fixed 25-color qualitative palette (ColorBrewer Dark2 +
# Set1 + Set2 concatenated, Mol*'s 'many-distinct' list), assigned to
# chains by their order of first appearance in the structure (not a
# fixed per-letter table). See `chain-id.ts` and `lists.ts` in
# github.com/molstar/molstar/tree/master/src/mol-{theme,util/color}.
# Used as the canonical source of chain colors, rather than an arbitrary
# palette, so the interactive structure render and the network panels
# both show the colors a plain view of the structure on rcsb.org would.
_MOLSTAR_MANY_DISTINCT = [
    '#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666',
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999',
    '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3',
]


def get_chain_color_palette(chain_ids):
    '''
    Assign each chain in a structure the color it gets under Mol*'s
    default "Chain ID" coloring (the viewer RCSB uses at rcsb.org) --
    _MOLSTAR_MANY_DISTINCT assigned by each chain's order of first
    appearance in the structure, cycling if there are more than 25
    chains, exactly matching Mol*'s own color-assignment logic.

    Parameters
    ----------
    chain_ids: list of str
        Chain IDs in the order they appear in the structure (e.g. dict
        keys from classify_pdb_chains(), which preserves file/parse
        order). Do not pre-sort this list -- order determines color
        assignment, matching structures to what rcsb.org shows requires
        our parsed chain order to agree with theirs.

    Returns
    -------
    dict
        Mapping of chain ID -> hex color string.

    Examples
    --------
    >>> palette = get_chain_color_palette(['A', 'B'])
    >>> palette['A']
    '#1b9e77'
    >>> palette['A'] == palette['B']
    False
    '''
    return {chain_id: _MOLSTAR_MANY_DISTINCT[i % len(_MOLSTAR_MANY_DISTINCT)]
            for i, chain_id in enumerate(chain_ids)}


def get_uniprot_color_palette(chain_to_UniProt_mapping_dict, chain_color_palette):
    '''
    Propagate a chain_id -> color palette (e.g. from get_chain_color_palette())
    to a UniProt/synthetic-ID -> color palette, so structure and network
    panels can share identical colors for the same physical chain.

    Parameters
    ----------
    Chain to UniProt Map: dict
    Chain Color Palette: dict

    Returns
    -------
    dict
        Mapping of UniProt (or synthetic 'RNA:'/'DNA:') ID -> hex color.
        If a UniProt ID maps to multiple chains (e.g. the two copies of a
        homo-oligomeric subunit, which get different default chain colors
        in the 3D render since they really are different chains), the
        alphabetically-first chain ID's color is used for that protein's
        single network node, deterministically (chain_to_UniProt_mapping_dict
        is built from a set() internally, so its iteration order is not
        itself stable across runs/interpreters).
    '''
    uniprot_color = {}
    for chain_id in sorted(chain_to_UniProt_mapping_dict.keys()):
        color = chain_color_palette.get(chain_id, '#b3b3b3')
        for id_i in chain_to_UniProt_mapping_dict[chain_id]:
            uniprot_color.setdefault(id_i, color)
    return uniprot_color


def _relax_overlapping_nodes(pos, min_separation, iterations=200, step=0.5):
    '''
    Internal helper: nudge apart only the specific node pairs sitting
    closer together than min_separation, by directly displacing each
    such pair along their connecting vector. Unlike a full force-directed
    layout (e.g. nx.spring_layout with no edges), this applies no global
    repulsion between well-separated nodes -- since every node in a
    "no edges" spring layout still repels every other node each
    iteration, that approach pushes the *entire* layout toward a
    uniformly-spaced ring, destroying the real depth information the PCA
    projection captured (a node near the projection's center genuinely
    means its 3D centroid is near the structure's center of mass along
    those two axes). Here, nodes that aren't in collision are never
    touched at all.
    '''
    node_ids = list(pos.keys())
    n = len(node_ids)
    if n < 2:
        return pos

    coords = np.array([pos[node_id] for node_id in node_ids], dtype=float)
    for _ in range(iterations):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                delta = coords[i] - coords[j]
                dist = np.linalg.norm(delta)
                if dist < min_separation:
                    moved = True
                    if dist < 1e-9:
                        delta = np.array([1.0, 0.0])
                        dist = 1.0
                    push = (min_separation - dist) / 2.0 * step * (delta / dist)
                    coords[i] += push
                    coords[j] -= push
        if not moved:
            break

    return {node_id: tuple(xy) for node_id, xy in zip(node_ids, coords)}


def _separate_nodes_on_screen(ax, pos, node_size, gap_points=8, safety=1.15,
                              rounds=3):
    '''
    Internal helper: move apart nodes whose markers would overlap once
    drawn on `ax`. node_size is a marker *area in points^2*, fixed on
    screen regardless of the data scale, so a minimum separation in
    layout (data) units can't guarantee non-overlap -- whether 0.15 data
    units is enough depends on how big the axes is and how spread out the
    layout is. This works in points instead: the layout is converted to
    on-screen points using the axes' size and the data span matplotlib
    will autoscale to (the node extent plus its default 5% margins), any
    pair closer than one node diameter plus gap_points is pushed apart
    with _relax_overlapping_nodes() (which only moves colliding pairs),
    and the result is converted back. Repeated a few rounds since moving
    nodes can widen the data span. `safety` pads the separation because
    the axes can still shrink slightly when fig.tight_layout() runs after
    drawing.
    '''
    if len(pos) < 2:
        return pos
    bbox = ax.get_window_extent()
    width_pt = bbox.width * 72 / ax.figure.dpi
    height_pt = bbox.height * 72 / ax.figure.dpi
    min_sep_points = (2 * np.sqrt(node_size / np.pi) + gap_points) * safety

    node_ids = list(pos)
    coords = np.array([pos[n] for n in node_ids], dtype=float)
    for _ in range(rounds):
        span = (coords.max(axis=0) - coords.min(axis=0)) * 1.1
        span[span == 0] = 1.0
        points_per_unit = np.array([width_pt, height_pt]) / span
        in_points = {n: tuple(xy * points_per_unit) for n, xy in zip(node_ids, coords)}
        relaxed = _relax_overlapping_nodes(in_points, min_separation=min_sep_points)
        new_coords = np.array([relaxed[n] for n in node_ids]) / points_per_unit
        if np.allclose(new_coords, coords):
            break
        coords = new_coords
    return {n: tuple(xy) for n, xy in zip(node_ids, coords)}


def _protein_centroids(chain_to_UniProt_mapping_dict, chain_centroids):
    '''
    Internal helper: average chain centroids per UniProt/synthetic ID, so
    chains of a homo-oligomer collapse to the one node that represents
    them in the network. Shared by get_structure_based_layout() and
    render_figure2_panels_static() (via _prepare_figure2_inputs()) so the
    PCA rotation used to pre-orient the structure render is computed from
    the exact same point set as the network layout.

    Returns
    -------
    tuple
        (node_ids, centroids) -- node_ids sorted list of str, centroids
        an (N, 3) array in the same order.
    '''
    id_coords = {}
    for chain_id, ids in chain_to_UniProt_mapping_dict.items():
        if chain_id not in chain_centroids:
            continue
        for id_i in ids:
            id_coords.setdefault(id_i, []).append(chain_centroids[chain_id])

    node_ids = sorted(id_coords.keys())
    centroids = np.vstack([np.mean(id_coords[node_id], axis=0) for node_id in node_ids])
    return node_ids, centroids


def _pca_rotation_matrix(points):
    '''
    Internal helper: a proper (right-handed, det=+1) 3x3 rotation matrix
    whose rows are points' top-3 principal component directions, plus
    the centroid mean used to center them. Applying `rotation @ (p - mean)`
    to any point `p` puts PC1 on the new x-axis, PC2 on y, PC3 on z --
    the same rotation used both to build the 2D network layout (via just
    the first two rows) and, in render_pdb_structure_static(), to
    pre-rotate the actual structure's atom coordinates before PyMOL ever
    sees them. Doing both from this one shared matrix is what keeps the
    structure panel and the network panels' orientation in agreement --
    np.linalg.svd's principal-component *signs* are otherwise arbitrary
    (unrelated runs/axes can each independently come out flipped), and
    PyMOL's own cmd.orient() picks a completely independent camera angle,
    so without sharing one matrix the two panels have no reason to agree
    and can easily end up as mirror images of each other.

    Parameters
    ----------
    points: (N, 3) array

    Returns
    -------
    tuple
        (rotation, mean) -- rotation a (3, 3) array, mean a (3,) array.
    '''
    mean = points.mean(axis=0)
    centered = points - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    rotation = vt[:3].copy()
    if rotation.shape[0] < 3:
        # fewer than 3 independent directions in the data (degenerate,
        # e.g. <3 points): complete to a full orthonormal basis
        completion, _ = np.linalg.qr(np.vstack([rotation, np.eye(3)]).T)
        rotation = completion.T[:3]
    if np.linalg.det(rotation) < 0:
        # SVD gives no handedness guarantee; flip the axis least likely
        # to matter visually (PC3, the "into the screen" depth axis, not
        # used by the 2D network layout at all) rather than PC1/PC2
        rotation[2] *= -1
    return rotation, mean


def get_structure_based_layout(chain_to_UniProt_mapping_dict, chain_centroids,
                               scale=1.0, min_separation=0.15):
    '''
    Derive a 2D network layout from a structure's real 3D geometry, via
    PCA, instead of an arbitrary layout algorithm (e.g. a circle) --
    so proteins/nucleic acids that are spatially clustered in the actual
    structure end up visually clustered in the network diagram too.

    Each node's 3D position is the average centroid of the chain(s)
    mapping to it (chains of a homo-oligomer are averaged together).
    Those 3D positions are then projected onto their two dominant axes
    of variation (the first two principal components), which is the 2D
    plane that best preserves the real relative distances between nodes.
    Any nodes left overlapping (or nearly so) by that projection -- e.g.
    two chains whose centroids happen to be close along both PCA axes --
    are then nudged apart by a local, pairwise collision-resolution pass
    (see _relax_overlapping_nodes()) that only moves the specific
    colliding nodes, so labels stay legible without discarding the real
    spatial arrangement (e.g. distance from center) of every other node.

    render_pdb_structure_static() pre-rotates the structure itself by
    this same PCA basis (see _pca_rotation_matrix()) before rendering, so
    the structure panel and this layout end up in visual agreement rather
    than each picking an independent, possibly mirrored orientation.

    Parameters
    ----------
    Chain to UniProt Map: dict
    Chain Centroids: dict (from get_chain_centroids())
    scale: float (optional)
        Layout is normalized so the furthest node from the origin is at
        this distance, to match the scale nx.circular_layout() produces
        (~1.0) so existing node_size/edge_width defaults still look right.
    min_separation: float (optional)
        Minimum node-node distance (in the same units as scale) below
        which the anti-overlap relaxation kicks in.

    Returns
    -------
    dict
        Mapping of UniProt/synthetic ID -> (x, y) tuple, usable directly
        as the node_pos argument to display_PDB_direct_interaction_network().
    '''
    node_ids, centroids = _protein_centroids(chain_to_UniProt_mapping_dict, chain_centroids)
    rotation, mean = _pca_rotation_matrix(centroids)
    projected = (centroids - mean) @ rotation[:2].T

    max_extent = np.abs(projected).max()
    if max_extent > 0:
        projected = projected / max_extent * scale

    pos = {node_id: tuple(xy) for node_id, xy in zip(node_ids, projected)}
    return _relax_overlapping_nodes(pos, min_separation=min_separation)


def _draw_outside_labels(ax, G, node_pos, labels, font_size, node_size,
                         padding_points=6):
    '''
    Internal helper: draw node labels positioned just outside each node,
    offset radially outward from the graph's centroid -- matching Figure
    2's own label style (name beside/above the node, not overlapping it),
    rather than the earlier in-node placement.

    Since labels now sit on the plot's white background rather than on
    the node's own fill color, they're drawn in plain black rather than
    the previous per-node white/black contrast choice (which only made
    sense for text drawn on top of a colored node).

    The offset is specified in points (via matplotlib's `annotate(...,
    textcoords='offset points')`) rather than as a fraction of the
    layout's data-coordinate spread. Those are unrelated quantities: node
    markers (`node_size`) are a fixed size in points^2 regardless of how
    spread out or compact the layout is, so a data-space fraction can end
    up smaller than the node's own on-screen radius for a tight layout
    with large nodes -- which put labels *inside* the circle instead of
    outside it. Points-based offset is guaranteed to clear the node by
    `padding_points` regardless of the layout's scale or density.

    Two nodes close enough together that their labels both clear their own
    node but still land on top of *each other* (e.g. two touching nodes
    whose outward directions from the centroid are nearly parallel) are
    resolved separately, by _resolve_label_collisions() -- call that
    *after* fig.tight_layout(), not here: tight_layout() can still resize
    / reposition this axes within the figure, which would invalidate any
    bounding-box measurements taken before it runs.

    Parameters
    ----------
    node_size: int
        Same value passed to nx.draw_networkx_nodes() for these nodes
        (matplotlib scatter marker area, in points^2) -- used to compute
        each label's clearance from its node's actual on-screen radius.
    padding_points: float (optional)
        Extra gap beyond the node's edge, in points.
    '''
    positions = np.array(list(node_pos.values()))
    centroid = positions.mean(axis=0)
    node_radius_points = np.sqrt(node_size / np.pi)
    offset_points = node_radius_points + padding_points

    for node_i, (x, y) in node_pos.items():
        # node_pos is shared across panels; only label nodes drawn in this
        # one (e.g. the all-BioPlex panel leaves out DNA/RNA nodes)
        if node_i not in G:
            continue
        direction = np.array([x, y]) - centroid
        norm = np.linalg.norm(direction)
        unit = direction / norm if norm > 1e-9 else np.array([0.0, 1.0])
        ax.annotate(labels.get(node_i, node_i), xy=(x, y), xycoords='data',
                   xytext=(unit[0] * offset_points, unit[1] * offset_points),
                   textcoords='offset points', ha='center', va='center',
                   fontsize=font_size, fontweight='bold', color='black')
    # node/edge collections don't include label text in matplotlib's
    # autoscaling, so without this the outward-offset labels nearest the
    # plot edge can get clipped by the axes boundary
    ax.margins(0.2)


def _resolve_label_collisions(ax, node_size, iterations=60, pixel_step=3.0,
                              min_gap_points=6.0):
    '''
    Nudge apart any of this axes' label bounding boxes that overlap each
    other, or that overlap a node's own circle, using the figure's actual
    renderer to get each label's real rendered extent (font/string-width-
    accurate, unlike a fixed data- or point-distance heuristic). Operates
    on the Annotation objects _draw_outside_labels() already added to
    `ax` (via `ax.texts`, all created with textcoords='offset points',
    anchored with xycoords='data') -- pushes are applied by directly
    adjusting each annotation's `.xyann` (its points-offset from its
    anchor), converting the pixel-space push via the figure's dpi.

    The label-vs-node check exists because clearing a node by
    `node_radius + padding` (as _draw_outside_labels() does) is only
    correct if the label has no width of its own -- in reality the text
    extends roughly its own half-width *back toward the node* whenever
    the offset direction is close to horizontal (since these labels are
    horizontal strings), which for a wide label/small padding can still
    land inside the node's circle. Checking every label against every
    node's actual on-screen circle (not just its own) is a real
    node-radius-vs-rendered-text-bbox rectangle/circle collision test,
    robust to label text length, rather than trying to precompute a
    "safe enough" offset per label ahead of time.

    A visible gap is required, not just zero overlap: text glyphs carry
    some visual weight beyond their precise bounding box (antialiasing,
    letterforms like descenders), so two boxes that are technically
    non-overlapping by a pixel or two can still read as touching.

    Must be called after fig.tight_layout() (or anything else that can
    still resize/reposition `ax` within the figure) -- bounding boxes
    measured before that wouldn't reflect the final saved figure.

    Parameters
    ----------
    node_size: int
        Same value passed to nx.draw_networkx_nodes() -- used to compute
        each node's on-screen radius for the label-vs-node check.
    '''
    annotations = list(ax.texts)
    if not annotations:
        return

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    points_per_pixel = 72.0 / fig.dpi
    min_gap_pixels = min_gap_points / points_per_pixel
    node_radius_pixels = np.sqrt(node_size / np.pi) / points_per_pixel
    # every annotation here was created via ax.annotate(..., xy=node_pos,
    # xycoords='data'), so .xy is that node's data-space anchor
    node_centers_px = [ax.transData.transform(ann.xy) for ann in annotations]

    def push(ann, direction_px):
        norm = np.linalg.norm(direction_px)
        unit = direction_px / norm if norm > 1e-6 else np.array([1.0, 0.0])
        delta_points = unit * pixel_step * points_per_pixel
        ann.xyann = (ann.xyann[0] + delta_points[0], ann.xyann[1] + delta_points[1])

    for _ in range(iterations):
        boxes = [ann.get_window_extent(renderer) for ann in annotations]
        padded = [b.padded(min_gap_pixels / 2) for b in boxes]
        moved = False

        for i in range(len(annotations)):
            for j in range(i + 1, len(annotations)):
                if not padded[i].overlaps(padded[j]):
                    continue
                moved = True
                ci = np.array([(boxes[i].x0 + boxes[i].x1) / 2, (boxes[i].y0 + boxes[i].y1) / 2])
                cj = np.array([(boxes[j].x0 + boxes[j].x1) / 2, (boxes[j].y0 + boxes[j].y1) / 2])
                push(annotations[i], ci - cj)
                push(annotations[j], cj - ci)

        for i, box in enumerate(boxes):
            for node_center in node_centers_px:
                nearest = np.array([min(max(node_center[0], box.x0), box.x1),
                                    min(max(node_center[1], box.y0), box.y1)])
                dist = np.linalg.norm(nearest - node_center)
                if dist >= node_radius_pixels + min_gap_pixels / 2:
                    continue
                moved = True
                box_center = np.array([(box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2])
                push(annotations[i], box_center - node_center)

        if not moved:
            break
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()


def _style_nucleic_acid_nodes(nodes, G, id_type, node_color_map,
                              protein_linewidth=1.5, nucleic_acid_linewidth=2.0):
    '''
    Internal helper: redraw DNA/RNA nodes as hollow (white) circles with a
    dashed outline, matching Figure 2's own convention (e.g. the H1 RNA
    node in panel G, the DNA node in panel H) of dashed styling for
    anything protein-nucleic acid -- not just the edges connecting to it,
    but the nucleic acid node itself. Protein nodes are left as solid
    filled circles.

    Parameters
    ----------
    nodes: matplotlib.collections.PathCollection (from nx.draw_networkx_nodes())
    G: networkx.Graph
    id_type: dict (from _id_type_map())
    node_color_map: list
        The fill colors originally passed to nx.draw_networkx_nodes(), in
        G.nodes order -- used here to know which entries to leave alone.
    '''
    facecolors, linestyles, linewidths = [], [], []
    for color, node_i in zip(node_color_map, G.nodes):
        if id_type.get(node_i) != 'protein':
            facecolors.append(matplotlib.colors.to_rgba('white'))
            linestyles.append('dashed')
            linewidths.append(nucleic_acid_linewidth)
        else:
            facecolors.append(matplotlib.colors.to_rgba(color))
            linestyles.append('solid')
            linewidths.append(protein_linewidth)

    nodes.set_facecolor(facecolors)
    nodes.set_linestyle(linestyles)
    nodes.set_linewidth(linewidths)


# ways display_PDB_direct_interaction_network() can show a per-interface
# score on each edge; all continuous on a fixed 0-1 scale, no cutoff
CONFIDENCE_STYLES = ('width', 'alpha', 'color')
_SCORE_NAMES = {'pair_iptm': 'chain-pair ipTM', 'ipsae': 'ipSAE',
                'pdockq': 'pDockQ', 'pdockq2': 'pDockQ2'}
_TOOL_NAMES = {'af3': 'AlphaFold3', 'boltz': 'Boltz', 'colabfold': 'ColabFold'}
_REDUCERS = {'mean': np.mean, 'min': min, 'max': max}


def get_edge_confidence_scores(interface_confidence, chain_to_UniProt_mapping_dict,
                               score='pair_iptm', reduce='mean'):
    '''
    One score per protein pair, for drawing on the network edges, from a
    predicted model's chain-pair scores (read_interface_confidence()).

    Some scores are not symmetric -- Boltz pair ipTM, and the ipSAE and
    pDockQ2 in some ColabFold pipelines' output give a different value for
    A->B than for B->A -- so the two directions are combined with
    `reduce`: 'mean' (default), 'min' (an edge only scores high if both
    directions do) or 'max'. The contact tables keep both directions
    (compare_structure_contacts_to_BioPlex()); this only affects the figure.

    Parameters
    ----------
    interface_confidence: dict (from read_interface_confidence())
    Chain to UniProt Map: dict
    score: str (optional)
        Which score to use: 'pair_iptm' (AlphaFold3, Boltz; default), or
        'ipsae'/'pdockq'/'pdockq2' where a ColabFold scores file has them.
    reduce: str (optional)
        'mean', 'min' or 'max' of the two directions.

    Returns
    -------
    dict or None
        frozenset({UniProt_i, UniProt_j}) -> score; None if the model has
        no such score.
    str or None
        Legend label naming the tool, the score, and -- when the two
        directions actually differ -- how they were combined.
    '''
    from bioplexpy.analysis_funcs import interface_confidence_by_uniprot

    if reduce not in _REDUCERS:
        raise ValueError(f"reduce must be one of {list(_REDUCERS)}, got '{reduce}'")
    if interface_confidence is None or score not in interface_confidence['scores']:
        return None, None
    directed = interface_confidence_by_uniprot(
        interface_confidence, chain_to_UniProt_mapping_dict)[score]
    by_pair = {}
    for (id_i, id_j), value in directed.items():
        by_pair.setdefault(frozenset((id_i, id_j)), []).append(value)
    asymmetric = any(len(set(values)) > 1 for values in by_pair.values())
    edge_scores = {pair: float(_REDUCERS[reduce](values))
                   for pair, values in by_pair.items()}
    label = (f"{_TOOL_NAMES.get(interface_confidence['tool'], interface_confidence['tool'])} "
             f"{_SCORE_NAMES.get(score, score)}")
    if asymmetric:
        label += f' ({reduce} of both directions)'
    return edge_scores, label


def _confidence_edge_properties(edges, edge_scores, confidence_style, edge_width,
                                edge_color):
    '''
    Internal helper: per-edge widths and RGBA colors that encode each
    edge's score in the given style. Edges without a score keep the plain
    width and color.
    '''
    base = matplotlib.colors.to_rgba(edge_color)
    widths, colors = [], []
    for a, b in edges:
        value = edge_scores.get(frozenset((a, b)))
        if value is None:
            widths.append(edge_width)
            colors.append(base)
            continue
        value = min(max(value, 0.0), 1.0)
        if confidence_style == 'width':
            widths.append(edge_width * (0.2 + 1.8 * value))
            colors.append(base)
        elif confidence_style == 'alpha':
            widths.append(edge_width)
            colors.append(base[:3] + (max(value, 0.05),))
        else:
            widths.append(edge_width)
            colors.append(plt.cm.viridis(value))
    return widths, colors


def _draw_confidence_legend(ax, confidence_style, confidence_label, edge_width,
                            edge_color):
    '''
    Internal helper: a key for the score encoding, below the panel -- a
    colorbar for 'color', sample lines for 'width'/'alpha'.
    '''
    if confidence_style == 'color':
        cax = ax.inset_axes([0.2, -0.06, 0.6, 0.03])
        mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                         norm=matplotlib.colors.Normalize(0, 1))
        colorbar = ax.figure.colorbar(mappable, cax=cax, orientation='horizontal')
        colorbar.set_label(confidence_label, fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
        return
    from matplotlib.lines import Line2D
    samples = [0.1, 0.4, 0.7, 1.0]
    widths, colors = _confidence_edge_properties(
        [(s, s) for s in samples], {frozenset((s, s)): s for s in samples},
        confidence_style, edge_width, edge_color)
    handles = [Line2D([], [], linewidth=w, color=c) for w, c in zip(widths, colors)]
    ax.legend(handles, [f'{s:g}' for s in samples], title=confidence_label,
              loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=len(samples),
              frameon=False, fontsize=8, title_fontsize=9, handlelength=2.5)


def _id_type_map(chain_to_UniProt_mapping_dict, chain_types):
    '''
    Internal helper: build a UniProt/synthetic-ID -> chain type
    ('protein'/'dna'/'rna') mapping from a chain-to-UniProt map and a
    chain-to-type map (e.g. from classify_pdb_chains()).
    '''
    id_type = {}
    for chain_id, ids in chain_to_UniProt_mapping_dict.items():
        chain_type = chain_types.get(chain_id, 'protein')
        for id_i in ids:
            id_type[id_i] = chain_type
    return id_type


def render_pdb_structure_py3Dmol(PDB_ID, protein_structure_dir, chain_color_palette,
                                 width=400, height=400, cartoon_style='cartoon',
                                 rotation=None, center=None):
    '''
    Render a PDB structure interactively with py3Dmol, colored by chain.

    This is Figure 2's column 1 (structure colored by chain). Pass it the
    same chain_color_palette (from get_chain_color_palette()) used to
    color the network panels, so the 3D render and the network node
    colors match. Downloads the structure into protein_structure_dir if
    it isn't already there.

    Parameters
    ----------
    PDB ID or path to a local structure file: str
    directory to store PDB file: str
    Chain Color Palette: dict
    width: int (optional)
    height: int (optional)
    cartoon_style: str (optional)
        py3Dmol style keyword, e.g. 'cartoon' or 'stick'.
    rotation: (3, 3) array (optional)
        If given (with `center`), every atom is pre-rotated by
        `rotation @ (coord - center)` before the model is loaded, and
        3Dmol.js's own default camera (looking down -Z, no explicit
        rotate() call) is relied on -- the same approach
        render_pdb_structure_static() uses for the PyMOL path, so the two
        renderers share one orientation instead of each independently
        picking one (which can come out as a mirror image of the other).
        Pass the (rotation, mean) from _pca_rotation_matrix() applied to
        the same chain centroids used for get_structure_based_layout()
        (see _prepare_figure2_inputs()). If omitted, falls back to
        3Dmol.js's own auto-fit via zoomTo() with no pre-rotation.
    center: (3,) array (optional)
        See `rotation`.

    Returns
    -------
    py3Dmol.view
        Call .show() on this in a Jupyter notebook to render it.
    '''
    from bioplexpy.analysis_funcs import fetch_pdb_structure_file, structure_label
    pdb_file_path, file_format = fetch_pdb_structure_file(PDB_ID, protein_structure_dir)
    label = structure_label(PDB_ID)

    if rotation is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rotated_ext = 'cif' if file_format == 'mmCif' else 'pdb'
            rotated_path = os.path.join(tmpdir, f'{label}_rotated.{rotated_ext}')
            _write_rotated_structure(pdb_file_path, file_format, rotation, center,
                                     rotated_path)
            with open(rotated_path) as pdb_file:
                pdb_data = pdb_file.read()
    else:
        with open(pdb_file_path) as pdb_file:
            pdb_data = pdb_file.read()

    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_data, 'cif' if file_format == 'mmCif' else 'pdb')
    # clear 3Dmol.js's default per-atom style so any chain missing from
    # chain_color_palette renders invisible rather than in a default style
    view.setStyle({}, {})
    for chain_id, color in chain_color_palette.items():
        view.setStyle({'chain': chain_id},
                      {cartoon_style: {'color': matplotlib.colors.to_hex(color)}})
    view.zoomTo()
    return view


def display_PDB_direct_interaction_network(ax, chain_to_UniProt_mapping_dict,
    interacting_UniProt_IDs, chain_types, node_color_palette, node_size,
    edge_width, node_font_size=10, node_pos=None, labels=None,
    edge_color='0.3', edge_scores=None, confidence_style='width',
    confidence_label=None):
    '''
    Display the PDB-derived direct interaction network for a structure
    (Figure 2, column 2): nodes colored to match the structure's chain
    colors; solid edges for protein-protein direct contacts (<threshold
    Angstroms apart, per get_interacting_chains_from_PDB()); dashed edges
    for protein-nucleic acid direct contacts. DNA/RNA nodes themselves are
    drawn as hollow circles with a dashed outline (matching the dashed-line
    convention used for their edges), rather than solid filled circles.

    For a predicted model, each edge can also show the predictor's own
    confidence in that interface (edge_scores, from
    get_edge_confidence_scores()), on a continuous 0-1 scale with a key
    below the panel. Nothing is hidden or cut off by score.

    Parameters
    ----------
    ax object to draw on: Matplotlib Axes
    Chain to UniProt Map: dict
    Interacting UniProt/synthetic IDs: list (from PDB_chains_to_uniprot())
    Chain Types: dict (from classify_pdb_chains())
    Node Color Palette: dict (e.g. from get_uniprot_color_palette())
    Size of Nodes in Network: int
    Width of Edges in Network: float
    Size of font for Node Labels: int (optional)
    Networkx Position of Nodes: dict (optional, computed if not given)
    Node Labels: dict (optional, defaults to the node IDs themselves)
    Color of edges: str (optional)
    edge_scores: dict (optional)
        frozenset({id_i, id_j}) -> score in 0-1, e.g. from
        get_edge_confidence_scores(). Edges without a score are drawn plain.
    confidence_style: str (optional)
        How edge_scores are shown: 'width' (line width grows with the
        score; default), 'alpha' (low scores fade) or 'color' (viridis
        colormap with a colorbar).
    confidence_label: str (optional)
        Title for the score key, e.g. 'AlphaFold3 chain-pair ipTM'.

    Returns
    -------
    Node Positions
        Dictionary of Node Positions in NetworkX layout, for reuse by
        display_BioPlex_direct_interactions() so both panels share a layout.
    '''
    if edge_scores and confidence_style not in CONFIDENCE_STYLES:
        raise ValueError(f'confidence_style must be one of {CONFIDENCE_STYLES}, '
                         f"got '{confidence_style}'")
    id_type = _id_type_map(chain_to_UniProt_mapping_dict, chain_types)
    all_ids = sorted({id_i for ids in chain_to_UniProt_mapping_dict.values() for id_i in ids})

    G = nx.Graph()
    G.add_nodes_from(all_ids)
    G.add_edges_from(interacting_UniProt_IDs)

    if node_pos is None:
        node_pos = nx.circular_layout(G)

    solid_edges = [(a, b) for a, b in G.edges
                   if id_type.get(a) == 'protein' and id_type.get(b) == 'protein']
    dashed_edges = [(a, b) for a, b in G.edges if (a, b) not in solid_edges]

    for edgelist, style in ((solid_edges, 'solid'), (dashed_edges, 'dashed')):
        if not edgelist:
            continue
        if edge_scores:
            widths, colors = _confidence_edge_properties(
                edgelist, edge_scores, confidence_style, edge_width, edge_color)
            nx.draw_networkx_edges(G, node_pos, edgelist=edgelist, width=widths,
                                   edge_color=colors, style=style, ax=ax)
        else:
            edges = nx.draw_networkx_edges(G, node_pos, edgelist=edgelist,
                                           width=edge_width, style=style, ax=ax)
            edges.set_edgecolor(edge_color)
    if edge_scores:
        _draw_confidence_legend(ax, confidence_style,
                                confidence_label or 'interface score', edge_width,
                                edge_color)

    node_color_map = [node_color_palette.get(n, (0.7, 0.7, 0.7, 1.0)) for n in G.nodes]
    nodes = nx.draw_networkx_nodes(G, node_pos, node_size=node_size,
                                   node_color=node_color_map, ax=ax)
    nodes.set_edgecolor('xkcd:black')
    _style_nucleic_acid_nodes(nodes, G, id_type, node_color_map)

    if labels is None:
        labels = {n: n for n in G.nodes}
    _draw_outside_labels(ax, G, node_pos, labels, node_font_size, node_size)

    return node_pos


def display_BioPlex_direct_interactions(ax, chain_to_UniProt_mapping_dict,
    interacting_UniProt_IDs, chain_types, bp_PPI_df, node_pos, node_size,
    edge_width, node_font_size=10, labels=None,
    detected_edge_color='xkcd:green', not_detected_edge_color='xkcd:grey',
    bait_node_color='xkcd:green', prey_node_color='xkcd:pale green',
    other_node_color='0.7'):
    '''
    Display the same direct-interaction topology as
    display_PDB_direct_interaction_network() (Figure 2, column 3), but
    recolor each edge by whether that pair was also detected by BioPlex
    AP-MS in the given cell line (green = detected, gray = not detected).
    Solid/dashed edge style still reflects protein-protein vs
    protein-nucleic acid, as in column 2. Protein nodes are colored by
    bait/prey status in bp_PPI_df; nucleic acid nodes (not profiled by
    AP-MS) are drawn as hollow, dashed-outline circles, as in column 2.

    Parameters
    ----------
    ax object to draw on: Matplotlib Axes
    Chain to UniProt Map: dict
    Interacting UniProt/synthetic IDs: list
    Chain Types: dict
    DataFrame of PPIs for one cell line: Pandas DataFrame (from getBioPlex())
    Networkx Position of Nodes: dict (from display_PDB_direct_interaction_network())
    Size of Nodes in Network: int
    Width of Edges in Network: float
    Size of font for Node Labels: int (optional)
    Node Labels: dict (optional)
    Color of Edges Detected via AP-MS: str (optional)
    Color of Edges Not Detected via AP-MS: str (optional)
    Color of Nodes targeted as baits: str (optional)
    Color of Nodes detected as preys only: str (optional)
    Color of Nucleic Acid / Undetected Nodes: str (optional)

    Returns
    -------
    None
    '''
    id_type = _id_type_map(chain_to_UniProt_mapping_dict, chain_types)
    all_ids = sorted({id_i for ids in chain_to_UniProt_mapping_dict.values() for id_i in ids})

    # strip isoform suffixes and build an undirected set of BioPlex-detected pairs
    from bioplexpy.analysis_funcs import _bioplex_edges_and_roles
    bp_edges, baits, preys = _bioplex_edges_and_roles(bp_PPI_df)

    G = nx.Graph()
    G.add_nodes_from(all_ids)
    G.add_edges_from(interacting_UniProt_IDs)

    for a, b in G.edges:
        detected = frozenset((a, b)) in bp_edges
        color = detected_edge_color if detected else not_detected_edge_color
        style = ('solid' if (id_type.get(a) == 'protein' and id_type.get(b) == 'protein')
                 else 'dashed')
        edges = nx.draw_networkx_edges(G, node_pos, edgelist=[(a, b)],
                                       width=edge_width, style=style, ax=ax)
        edges.set_edgecolor(color)

    node_color_map = []
    for node_i in G.nodes:
        if id_type.get(node_i) != 'protein':
            node_color_map.append(other_node_color)
        elif node_i in baits:
            node_color_map.append(bait_node_color)
        elif node_i in preys:
            node_color_map.append(prey_node_color)
        else:
            node_color_map.append(other_node_color)

    nodes = nx.draw_networkx_nodes(G, node_pos, node_size=node_size,
                                   node_color=node_color_map, ax=ax)
    nodes.set_edgecolor('xkcd:black')
    _style_nucleic_acid_nodes(nodes, G, id_type, node_color_map)

    if labels is None:
        labels = {n: n for n in G.nodes}
    _draw_outside_labels(ax, G, node_pos, labels, node_font_size, node_size)


def display_All_BioPlex_interactions_two_cell_lines(ax, protein_ids,
    direct_pairs, bp_293t_df, bp_hct116_df, node_pos, node_size, edge_width,
    node_font_size=10, labels=None,
    cell_293t_color='xkcd:red', cell_hct116_color='xkcd:blue',
    cell_both_color='xkcd:grey', not_detected_node_color='0.85',
    direct_width=None, indirect_width=None):
    '''
    Display the full BioPlex subnetwork among a set of proteins (Figure 2,
    column 4): edges colored by which cell line(s) detected them (293T
    red / HCT116 blue / both grey); nodes colored by bait/prey status and
    by cell line. Per the paper's own legend for this panel, every edge
    is solid -- direct (also a PDB direct contact, in direct_pairs) is
    drawn bold/thick, indirect is drawn thin. There is no dashed styling
    in this panel (dashed is reserved for protein-nucleic acid edges in
    the PDB-direct / BioPlex-direct panels, which this panel excludes
    entirely since nucleic acids aren't AP-MS baits/preys).

    Parameters
    ----------
    ax object to draw on: Matplotlib Axes
    UniProt IDs to include as nodes: list (typically the protein-only IDs
        from a structure's chain_to_UniProt_mapping_dict)
    Direct (PDB-contact) pairs: set of frozenset({UniprotA, UniprotB})
    DataFrame of 293T PPIs: Pandas DataFrame (from getBioPlex('293T', ...))
    DataFrame of HCT116 PPIs: Pandas DataFrame (from getBioPlex('HCT116', ...))
    Networkx Position of Nodes: dict
    Size of Nodes in Network: int
    Width of Edges in Network: float
        Used as the bold/direct line width unless direct_width is given.
    Size of font for Node Labels: int (optional)
    Node Labels: dict (optional)
    Color of 293T-only Edges/Baits: str (optional)
    Color of HCT116-only Edges/Baits: str (optional)
    Color of Shared (Both) Edges/Baits: str (optional)
    Color of Nodes Not Detected in Either Cell Line: str (optional)
    Width of Direct (bold) Edges: float (optional, defaults to edge_width)
    Width of Indirect (thin) Edges: float (optional, defaults to 0.4 * edge_width)

    Returns
    -------
    None
    '''
    if direct_width is None:
        direct_width = edge_width
    if indirect_width is None:
        indirect_width = edge_width * 0.4
    from bioplexpy.analysis_funcs import _bioplex_edges_and_roles
    edges_293t, baits_293t, preys_293t = _bioplex_edges_and_roles(
        bp_293t_df, restrict_to=protein_ids)
    edges_hct116, baits_hct116, preys_hct116 = _bioplex_edges_and_roles(
        bp_hct116_df, restrict_to=protein_ids)
    all_edges = edges_293t | edges_hct116

    G = nx.Graph()
    G.add_nodes_from(protein_ids)
    G.add_edges_from(tuple(edge) for edge in all_edges if len(edge) == 2)

    for edge in all_edges:
        if len(edge) != 2:
            continue
        a, b = tuple(edge)
        in_293t, in_hct116 = edge in edges_293t, edge in edges_hct116
        if in_293t and in_hct116:
            color = cell_both_color
        elif in_293t:
            color = cell_293t_color
        else:
            color = cell_hct116_color
        width = direct_width if edge in direct_pairs else indirect_width
        edges = nx.draw_networkx_edges(G, node_pos, edgelist=[(a, b)],
                                       width=width, style='solid', ax=ax)
        edges.set_edgecolor(color)

    node_color_map = []
    for node_i in protein_ids:
        is_bait_293t = node_i in baits_293t
        is_bait_hct116 = node_i in baits_hct116
        is_prey_293t = node_i in preys_293t
        is_prey_hct116 = node_i in preys_hct116
        if is_bait_293t and is_bait_hct116:
            node_color_map.append(cell_both_color)
        elif is_bait_293t:
            node_color_map.append(cell_293t_color)
        elif is_bait_hct116:
            node_color_map.append(cell_hct116_color)
        elif is_prey_293t and is_prey_hct116:
            node_color_map.append(matplotlib.colors.to_rgba(cell_both_color, alpha=0.4))
        elif is_prey_293t:
            node_color_map.append(matplotlib.colors.to_rgba(cell_293t_color, alpha=0.4))
        elif is_prey_hct116:
            node_color_map.append(matplotlib.colors.to_rgba(cell_hct116_color, alpha=0.4))
        else:
            node_color_map.append(not_detected_node_color)

    nodes = nx.draw_networkx_nodes(G, node_pos, node_size=node_size,
                                   node_color=node_color_map, ax=ax)
    nodes.set_edgecolor('xkcd:black')
    nodes.set_linewidth(1.5)

    if labels is None:
        labels = {n: n for n in G.nodes}
    _draw_outside_labels(ax, G, node_pos, labels, node_font_size, node_size)


def _prepare_figure2_inputs(PDB_ID, protein_structure_dir, bp_293t_df, bp_hct116_df,
                            interact_dist_threshold, chain_to_uniprot=None,
                            min_plddt=None, confidence_score='pair_iptm',
                            confidence_reduce='mean'):
    '''
    Internal helper: everything render_figure2_panels() and
    render_figure2_panels_static() both need -- the PDB-direct/UniProt
    mappings, chain color palette, structure-based node layout, and
    gene-symbol labels -- computed once so the two entry points can't
    drift out of sync with each other.
    '''
    from bioplexpy.analysis_funcs import (PDB_to_interacting_chains_uniprot_maps,
                                          _bioplex_symbol_lookup,
                                          get_chain_centroids,
                                          _read_interface_confidence_or_warn,
                                          is_local_structure_file)

    chain_to_uniprot, interacting_uniprot_ids, chain_types = (
        PDB_to_interacting_chains_uniprot_maps(PDB_ID, protein_structure_dir,
                                               interact_dist_threshold,
                                               chain_to_uniprot=chain_to_uniprot,
                                               min_plddt=min_plddt))

    chain_color_palette = get_chain_color_palette(list(chain_types.keys()))
    node_color_palette = get_uniprot_color_palette(chain_to_uniprot, chain_color_palette)

    chain_centroids = get_chain_centroids(PDB_ID, protein_structure_dir)
    structure_layout = get_structure_based_layout(chain_to_uniprot, chain_centroids)

    # same PCA basis get_structure_based_layout() just used, exposed here
    # so render_pdb_structure_static() can pre-rotate the structure into
    # it too (see render_figure2_panels_static()) instead of the two
    # panels each picking an independent (possibly mirrored) orientation
    _, protein_centroid_points = _protein_centroids(chain_to_uniprot, chain_centroids)
    structure_rotation, structure_center = _pca_rotation_matrix(protein_centroid_points)

    id_type = _id_type_map(chain_to_uniprot, chain_types)
    all_ids = sorted({id_i for ids in chain_to_uniprot.values() for id_i in ids})

    # gene symbol labels where available, straight from the BioPlex dataframes
    symbol_lookup = _bioplex_symbol_lookup(bp_293t_df, bp_hct116_df)
    labels = {id_i: symbol_lookup.get(id_i, id_i) for id_i in all_ids}

    # a predicted model's own per-interface scores, if the predictor wrote
    # them next to the model file
    edge_scores, confidence_label = None, None
    if is_local_structure_file(PDB_ID):
        interface_confidence = _read_interface_confidence_or_warn(
            PDB_ID, chain_ids=list(chain_types))
        edge_scores, confidence_label = get_edge_confidence_scores(
            interface_confidence, chain_to_uniprot, score=confidence_score,
            reduce=confidence_reduce)
        if interface_confidence is not None and edge_scores is None:
            warnings.warn(f"No '{confidence_score}' score in "
                          f"{interface_confidence['source']} (it has: "
                          f"{', '.join(interface_confidence['scores'])}); edges "
                          'drawn without confidence.')

    return dict(
        chain_to_uniprot=chain_to_uniprot,
        interacting_uniprot_ids=interacting_uniprot_ids,
        chain_types=chain_types,
        chain_color_palette=chain_color_palette,
        node_color_palette=node_color_palette,
        structure_layout=structure_layout,
        structure_rotation=structure_rotation,
        structure_center=structure_center,
        id_type=id_type,
        all_ids=all_ids,
        labels=labels,
        edge_scores=edge_scores,
        confidence_label=confidence_label,
    )


def _draw_figure2_network_panels(axes, PDB_ID, protein_structure_dir, bp_293t_df,
                                 bp_hct116_df, prepared, node_size, edge_width,
                                 node_font_size, confidence_style='width'):
    from bioplexpy.analysis_funcs import is_local_structure_file
    '''
    Internal helper: draw the three network panels (PDB direct / BioPlex
    direct / all BioPlex) onto the given 3 axes, using inputs already
    computed by _prepare_figure2_inputs(). Shared by render_figure2_panels()
    and render_figure2_panels_static() so the network-panel logic lives
    in exactly one place.
    '''
    # all three panels share one layout and have the same size, so the
    # screen-space separation is worked out once, on the first panel
    node_pos = _separate_nodes_on_screen(axes[0], prepared['structure_layout'],
                                         node_size)
    node_pos = display_PDB_direct_interaction_network(
        axes[0], prepared['chain_to_uniprot'], prepared['interacting_uniprot_ids'],
        prepared['chain_types'], prepared['node_color_palette'], node_size, edge_width,
        node_font_size, labels=prepared['labels'], node_pos=node_pos,
        edge_scores=prepared['edge_scores'] if confidence_style else None,
        confidence_style=confidence_style,
        confidence_label=prepared['confidence_label'])
    # a user's own file may be a prediction, not a PDB entry
    source = 'Model' if is_local_structure_file(PDB_ID) else 'PDB'
    axes[0].set_title(f'{source} Direct Interaction Network')
    axes[0].axis('off')

    display_BioPlex_direct_interactions(
        axes[1], prepared['chain_to_uniprot'], prepared['interacting_uniprot_ids'],
        prepared['chain_types'], bp_293t_df, node_pos, node_size, edge_width,
        node_font_size, labels=prepared['labels'])
    axes[1].set_title('BioPlex Direct Interactions (293T)')
    axes[1].axis('off')

    protein_ids = [id_i for id_i in prepared['all_ids']
                  if prepared['id_type'].get(id_i) == 'protein']
    direct_pairs = {frozenset((a, b)) for a, b in prepared['interacting_uniprot_ids']
                    if prepared['id_type'].get(a) == 'protein'
                    and prepared['id_type'].get(b) == 'protein'}
    display_All_BioPlex_interactions_two_cell_lines(
        axes[2], protein_ids, direct_pairs, bp_293t_df, bp_hct116_df,
        node_pos, node_size, edge_width, node_font_size, labels=prepared['labels'])
    axes[2].set_title('All BioPlex Interactions (293T + HCT116)')
    axes[2].axis('off')


def render_figure2_panels(PDB_ID, protein_structure_dir, bp_293t_df, bp_hct116_df,
    interact_dist_threshold=6, figsize=(16, 5.5), node_size=1400,
    edge_width=2.5, node_font_size=9, chain_to_uniprot=None, min_plddt=None,
    confidence_style='width', confidence_score='pair_iptm', confidence_reduce='mean'):
    '''
    Reproduce Figure 2F-H of Huttlin et al. 2021 for a given PDB structure:
    finds direct interactions from the structure, overlays BioPlex AP-MS
    data from both cell lines, and assembles the three network panels
    (PDB direct / BioPlex direct / all BioPlex) into one matplotlib
    figure sharing a chain-colored layout and consistent node colors. A
    separate interactive py3Dmol view is also returned for the structure
    itself (column 1); call .show() on it in a notebook.

    For a single static image with the structure panel included (e.g.
    when there's no browser/Jupyter available to view the interactive
    py3Dmol view), see render_figure2_panels_static() instead.

    Parameters
    ----------
    PDB ID or path to a local structure file: str
    directory to store PDB file: str
    DataFrame of 293T PPIs: Pandas DataFrame (from getBioPlex('293T', ...))
    DataFrame of HCT116 PPIs: Pandas DataFrame (from getBioPlex('HCT116', ...))
    Direct-contact distance threshold (Angstroms): int (optional)
    figsize: tuple (optional)
    Size of Nodes in Network: int (optional)
    Width of Edges in Network: float (optional)
    Size of font for Node Labels: int (optional)
    chain_to_uniprot: dict (optional)
        Chain ID -> UniProt ID(s). Required when PDB_ID is a local
        structure file (which has no SIFTS mapping); see
        PDB_to_interacting_chains_uniprot_maps().
    min_plddt: float (optional)
        For predicted structures: ignore atoms below this pLDDT when
        finding direct contacts (see PDB_to_interacting_chains_uniprot_maps()).
    confidence_style: str or None (optional)
        For a predicted model whose predictor wrote per-interface scores
        next to it (see read_interface_confidence()): how the model
        network's edges show them -- 'width' (default), 'alpha' or
        'color'; None draws plain edges. No effect on experimental
        structures. Annotation only: no edge is removed.
    confidence_score: str (optional)
        Which score: 'pair_iptm' (AlphaFold3/Boltz chain-pair ipTM;
        default), or 'ipsae'/'pdockq'/'pdockq2' if the model's ColabFold
        scores file has them.
    confidence_reduce: str (optional)
        How the two directions of an asymmetric score (Boltz ipTM, ipSAE,
        pDockQ2) become one edge value: 'mean' (default), 'min' or 'max'.
        The legend says which, when it matters.

    Returns
    -------
    Figure
        Matplotlib Figure with the three network panels.
    py3Dmol.view
        Interactive, chain-colored 3D render of the structure (column 1).
    '''
    prepared = _prepare_figure2_inputs(PDB_ID, protein_structure_dir, bp_293t_df,
                                       bp_hct116_df, interact_dist_threshold,
                                       chain_to_uniprot=chain_to_uniprot,
                                       min_plddt=min_plddt,
                                       confidence_score=confidence_score,
                                       confidence_reduce=confidence_reduce)
    structure_view = render_pdb_structure_py3Dmol(
        PDB_ID, protein_structure_dir, prepared['chain_color_palette'],
        rotation=prepared['structure_rotation'], center=prepared['structure_center'])

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    _draw_figure2_network_panels(axes, PDB_ID, protein_structure_dir, bp_293t_df,
                                 bp_hct116_df, prepared, node_size, edge_width,
                                 node_font_size, confidence_style=confidence_style)

    fig.tight_layout()
    for ax in axes:
        _resolve_label_collisions(ax, node_size)
    return fig, structure_view


def _write_rotated_pdb(pdb_file_path, rotation, mean, out_path):
    '''
    Internal helper: write a copy of a legacy-format PDB file with every
    atom coordinate transformed by `rotation @ (coord - mean)`. Used by
    render_pdb_structure_static() to pre-rotate the structure into the
    same PCA frame as get_structure_based_layout() before PyMOL ever sees
    it, rather than asking PyMOL to match that frame after the fact.

    Rewrites the ATOM/HETATM coordinate columns directly at the text level
    rather than round-tripping through Biopython's Structure/Atom objects.
    Round-tripping is not safe here: for residues with alternate
    conformations (altloc), mutating `atom.coord` on the Structure object
    only updates the "selected" conformer, but PDBIO.save() unpacks *all*
    altloc conformers when writing -- so unselected conformers get written
    with their original, unrotated coordinates while their neighbors are
    rotated, producing spurious multi-hundred-angstrom "bonds" in the
    output structure (and a corresponding visual artifact in PyMOL's
    cartoon rendering, e.g. long stray lines fanning off the structure).
    Transforming every coordinate-bearing line directly avoids this.
    '''
    with open(pdb_file_path) as f:
        lines = f.readlines()

    with open(out_path, 'w') as out:
        for line in lines:
            if line.startswith(('ATOM', 'HETATM')):
                line = line.rstrip('\n').ljust(80) + '\n'
                coord = rotation @ (np.array([
                    float(line[30:38]), float(line[38:46]), float(line[46:54])
                ]) - mean)
                line = f'{line[:30]}{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}{line[54:]}'
            out.write(line)


def _write_rotated_mmcif(cif_file_path, rotation, mean, out_path):
    '''
    mmCIF counterpart to _write_rotated_pdb() -- same rationale (text-level
    rewrite to avoid Biopython's altloc-unpacking mismatch between mutated
    and written atoms), adapted to mmCIF's `_atom_site` loop layout.

    Unlike the legacy PDB format's fixed-width columns, mmCIF's `_atom_site`
    loop is whitespace-token-delimited, with column order given by the
    preceding `_atom_site.<field>` header lines -- so this locates the
    Cartn_x/y/z token positions from that header, then rewrites only those
    tokens on each ATOM/HETATM data line, leaving every other token
    (including quoted ones, e.g. atom names like "O5'") untouched.
    '''
    with open(cif_file_path) as f:
        lines = f.readlines()

    field_names = []
    in_atom_site_header = False
    cartn_idx = None
    out_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('_atom_site.'):
            in_atom_site_header = True
            field_names.append(stripped[len('_atom_site.'):])
            out_lines.append(line)
            continue
        if in_atom_site_header and not stripped.startswith('_atom_site.'):
            in_atom_site_header = False
            cartn_idx = (field_names.index('Cartn_x'),
                        field_names.index('Cartn_y'),
                        field_names.index('Cartn_z'))

        if cartn_idx is not None and line.startswith(('ATOM', 'HETATM')):
            tokens = line.split()
            xi, yi, zi = cartn_idx
            coord = rotation @ (np.array([
                float(tokens[xi]), float(tokens[yi]), float(tokens[zi])
            ]) - mean)
            tokens[xi], tokens[yi], tokens[zi] = (f'{coord[0]:.3f}',
                                                  f'{coord[1]:.3f}',
                                                  f'{coord[2]:.3f}')
            line = ' '.join(tokens) + '\n'
        out_lines.append(line)

    with open(out_path, 'w') as out:
        out.writelines(out_lines)


def _write_rotated_structure(file_path, file_format, rotation, mean, out_path):
    '''
    Dispatch to _write_rotated_pdb() or _write_rotated_mmcif() based on
    which format the structure was downloaded in (see
    fetch_pdb_structure_file() in analysis_funcs.py).
    '''
    if file_format == 'mmCif':
        _write_rotated_mmcif(file_path, rotation, mean, out_path)
    else:
        _write_rotated_pdb(file_path, rotation, mean, out_path)


def render_pdb_structure_static(PDB_ID, protein_structure_dir, chain_color_palette,
                                width=800, height=800, dpi=150,
                                rotation=None, center=None):
    '''
    Render a static, chain-colored cartoon image of a PDB structure with
    PyMOL, for embedding in a single static figure (see
    render_figure2_panels_static()) -- e.g. for headless/no-browser use
    where the interactive py3Dmol view (render_pdb_structure_py3Dmol())
    isn't practical to view.

    Requires the optional `pymol-open-source` package
    (`pip install pymol-open-source`) -- not a core dependency of
    bioplexpy, since it's a large (~30MB), compiled package only needed
    for this static-rendering path.

    Parameters
    ----------
    PDB ID or path to a local structure file: str
    directory to store PDB file: str
    Chain Color Palette: dict (e.g. from get_chain_color_palette())
    width, height: int (optional)
        Ray-traced image dimensions in pixels.
    dpi: int (optional)
    rotation: (3, 3) array (optional)
        If given (with `center`), every atom is pre-rotated by
        `rotation @ (coord - center)` before rendering, and PyMOL's own
        cmd.orient() is skipped in favor of its default camera (which
        looks down -Z), so the rendered image's screen x/y axes are
        exactly the frame's PC1/PC2 -- the same two axes
        get_structure_based_layout() uses for the network panels. Pass
        the (rotation, mean) from _pca_rotation_matrix() applied to the
        same chain centroids used for that layout (see
        _prepare_figure2_inputs()) so the structure and network panels
        share one orientation instead of each independently picking one
        (which can come out as a mirror image of the other). If omitted,
        falls back to PyMOL's own cmd.orient().
    center: (3,) array (optional)
        See `rotation`.

    Returns
    -------
    numpy.ndarray
        RGB(A) image array, ready to display via ax.imshow().
    '''
    try:
        import pymol2
    except ImportError as e:
        raise ImportError(
            "render_pdb_structure_static() requires the optional "
            "'pymol-open-source' package: pip install pymol-open-source"
        ) from e

    from bioplexpy.analysis_funcs import fetch_pdb_structure_file, structure_label
    pdb_file_path, file_format = fetch_pdb_structure_file(PDB_ID, protein_structure_dir)
    label = structure_label(PDB_ID)

    with tempfile.TemporaryDirectory() as tmpdir:
        png_path = os.path.join(tmpdir, f'{label}.png')

        if rotation is not None:
            rotated_ext = 'cif' if file_format == 'mmCif' else 'pdb'
            render_source_path = os.path.join(tmpdir, f'{label}_rotated.{rotated_ext}')
            _write_rotated_structure(pdb_file_path, file_format, rotation, center,
                                     render_source_path)
        else:
            render_source_path = pdb_file_path

        session = pymol2.PyMOL()
        session.start()
        try:
            cmd = session.cmd
            cmd.load(render_source_path, label)
            # files with no secondary-structure records (common for
            # predictions, e.g. AlphaFold3/Boltz mmCIF, and some RCSB
            # mmCIFs) would otherwise draw as bare loops; let PyMOL
            # assign it. Files that do carry it are left as deposited.
            if cmd.count_atoms(f'{label} and name CA and ss H+S') == 0:
                cmd.dss(label)
            cmd.hide('everything')
            cmd.show('cartoon')
            cmd.bg_color('white')
            cmd.set('ray_opaque_background', 1)
            for chain_id, color in chain_color_palette.items():
                hex_color = matplotlib.colors.to_hex(color).replace('#', '0x')
                cmd.color(hex_color, f'chain {chain_id}')
            if rotation is not None:
                cmd.zoom()
            else:
                cmd.orient()
            cmd.ray(width, height)
            cmd.png(png_path, dpi=dpi)
        finally:
            session.stop()

        return mpimg.imread(png_path)


def _wrap_title(title, width=40):
    '''
    Internal helper: break a long panel title (e.g. a predictor's output
    file name, which has no spaces) onto several lines at '_'/'-'/' '
    boundaries, so it stays within its own panel instead of running into
    the next one.
    '''
    lines, line = [], ''
    for token in re.findall(r'[^_\- ]+[_\- ]*|[_\- ]+', title):
        if line and len(line) + len(token) > width:
            lines.append(line)
            line = ''
        line += token
    lines.append(line)
    return '\n'.join(lines)


def render_figure2_panels_static(PDB_ID, protein_structure_dir, bp_293t_df, bp_hct116_df,
    interact_dist_threshold=6, figsize=(20, 5.5), node_size=1400, edge_width=2.5,
    node_font_size=9, structure_width=800, structure_height=800,
    chain_to_uniprot=None, min_plddt=None, confidence_style='width',
    confidence_score='pair_iptm', confidence_reduce='mean'):
    '''
    Like render_figure2_panels(), but produces a single static, 4-panel
    matplotlib Figure -- the PDB structure (via PyMOL,
    see render_pdb_structure_static()) plus the three network panels --
    matching Figure 2's F/G/H layout in one file, for cases where an
    interactive py3Dmol view isn't practical to view (e.g. no browser
    access). Requires the optional `pymol-open-source` package; see
    render_pdb_structure_static().

    Parameters
    ----------
    PDB ID or path to a local structure file: str
    directory to store PDB file: str
    DataFrame of 293T PPIs: Pandas DataFrame (from getBioPlex('293T', ...))
    DataFrame of HCT116 PPIs: Pandas DataFrame (from getBioPlex('HCT116', ...))
    Direct-contact distance threshold (Angstroms): int (optional)
    figsize: tuple (optional)
    Size of Nodes in Network: int (optional)
    Width of Edges in Network: float (optional)
    Size of font for Node Labels: int (optional)
    structure_width, structure_height: int (optional)
        Ray-traced structure image dimensions in pixels.
    chain_to_uniprot: dict (optional)
        Chain ID -> UniProt ID(s). Required when PDB_ID is a local
        structure file (which has no SIFTS mapping); see
        PDB_to_interacting_chains_uniprot_maps().
    min_plddt: float (optional)
        For predicted structures: ignore atoms below this pLDDT when
        finding direct contacts (see PDB_to_interacting_chains_uniprot_maps()).
    confidence_style: str or None (optional)
        For a predicted model whose predictor wrote per-interface scores
        next to it (see read_interface_confidence()): how the model
        network's edges show them -- 'width' (default), 'alpha' or
        'color'; None draws plain edges. No effect on experimental
        structures. Annotation only: no edge is removed.
    confidence_score: str (optional)
        Which score: 'pair_iptm' (AlphaFold3/Boltz chain-pair ipTM;
        default), or 'ipsae'/'pdockq'/'pdockq2' if the model's ColabFold
        scores file has them.
    confidence_reduce: str (optional)
        How the two directions of an asymmetric score (Boltz ipTM, ipSAE,
        pDockQ2) become one edge value: 'mean' (default), 'min' or 'max'.
        The legend says which, when it matters.

    Returns
    -------
    Figure
        Matplotlib Figure with all four panels.
    '''
    prepared = _prepare_figure2_inputs(PDB_ID, protein_structure_dir, bp_293t_df,
                                       bp_hct116_df, interact_dist_threshold,
                                       chain_to_uniprot=chain_to_uniprot,
                                       min_plddt=min_plddt,
                                       confidence_score=confidence_score,
                                       confidence_reduce=confidence_reduce)
    structure_image = render_pdb_structure_static(
        PDB_ID, protein_structure_dir, prepared['chain_color_palette'],
        width=structure_width, height=structure_height,
        rotation=prepared['structure_rotation'], center=prepared['structure_center'])

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    axes[0].imshow(structure_image)
    from bioplexpy.analysis_funcs import structure_label
    axes[0].set_title(_wrap_title(f'{structure_label(PDB_ID)} Structure'))
    axes[0].axis('off')

    _draw_figure2_network_panels(axes[1:], PDB_ID, protein_structure_dir, bp_293t_df,
                                 bp_hct116_df, prepared, node_size, edge_width,
                                 node_font_size, confidence_style=confidence_style)

    fig.tight_layout()
    for ax in axes[1:]:
        _resolve_label_collisions(ax, node_size)
    return fig


def render_figure2_panels_for_uniprots(uniprot_IDs_list, protein_structure_dir,
                                       bp_293t_df, bp_hct116_df, pdb_id=None,
                                       static=True, **kwargs):
    '''
    TODO: provisional name, flagged for rename -- not yet decided (2026-08-20).

    Convenience wrapper: given a complex's UniProt ID list -- from any
    source, e.g. get_UniProts_from_CORUM() or get_UniProts_from_ComplexPortal()
    -- picks a matching PDB structure and renders the Figure 2-style panels,
    without having to manually call get_PDB_from_UniProts() and read a
    candidate PDB ID out of the returned DataFrame first.

    The auto-picked structure is whichever candidate from
    get_PDB_from_UniProts() actually has the most of the *specific* input
    UniProt IDs mapped to it (via that DataFrame's own
    'UniProts_mapped_to_PDB' column), not just get_PDB_from_UniProts()'s
    own default row order (closest protein *count*, then most recent
    deposit). Matching on protein count alone is misleading whenever two
    different complexes happen to be the same size and share some subunits
    -- e.g. RNA Pol I and Pol II are both often modeled with 13 chains and
    share 5 literal subunits, so the default order can rank a same-sized
    Pol II structure above the real Pol I ones purely for being newer, even
    though the Pol I structures cover all 13 of Pol I's own subunits and
    the Pol II one covers only the 5 shared ones (a real case that surfaced
    this exact bug). Ties in overlap count fall back to
    get_PDB_from_UniProts()'s own ranking (protein-count closeness, then
    deposit date). This still isn't a guarantee of the "right" structure --
    e.g. it can't distinguish two equally-good candidates by quality/
    resolution -- so if the auto-pick looks wrong, inspect
    get_PDB_from_UniProts(uniprot_IDs_list) yourself (its
    'UniProts_mapped_to_PDB' column shows exactly which input IDs each
    candidate covers) and pass whichever PDB ID you prefer via `pdb_id`.

    Parameters
    ----------
    UniProt IDs for the complex: list
    directory to store PDB file: str
    DataFrame of 293T PPIs: Pandas DataFrame (from getBioPlex('293T', ...))
    DataFrame of HCT116 PPIs: Pandas DataFrame (from getBioPlex('HCT116', ...))
    pdb_id: str (optional)
        Skip auto-picking and use this PDB ID instead.
    static: bool (optional, default True)
        If True (default), renders via render_figure2_panels_static() (a
        single static Figure, e.g. for no-browser use). If False, renders
        via render_figure2_panels() (an interactive py3Dmol view for the
        structure panel, plus a Figure for the three network panels).
    **kwargs
        Passed through to whichever of the two render functions is used
        (e.g. figsize, node_size, interact_dist_threshold).

    Returns
    -------
    Whatever the chosen render function returns:
    render_figure2_panels_static() -> Figure
    render_figure2_panels() -> (Figure, py3Dmol.view)
    '''
    from bioplexpy.data_import_funcs import get_PDB_from_UniProts

    if pdb_id is None:
        candidates = get_PDB_from_UniProts(uniprot_IDs_list)
        if candidates is None or len(candidates) == 0:
            raise ValueError(
                'No PDB structure found for this UniProt ID list -- pass an '
                'explicit pdb_id, or check the IDs with get_PDB_from_UniProts() '
                'directly.')
        # rank by genuine overlap with the input UniProt IDs first (not just
        # protein-count coincidence), falling back to get_PDB_from_UniProts()'s
        # own ranking to break ties -- see docstring for why this matters
        overlap_count = candidates['UniProts_mapped_to_PDB'].apply(len)
        ranked = candidates.assign(_overlap_count=overlap_count).sort_values(
            by=['_overlap_count', 'num_proteins_diff_btwn_PDB_and_UniProts_input',
               'deposit_date'],
            ascending=[False, True, False])
        pdb_id = ranked.index[0]

    render_fn = render_figure2_panels_static if static else render_figure2_panels
    return render_fn(pdb_id, protein_structure_dir, bp_293t_df, bp_hct116_df, **kwargs)

def render_figure2_panels_from_file(structure_file, bp_293t_df, bp_hct116_df,
                                    chain_to_uniprot=None, uniprot_IDs_list=None,
                                    min_plddt=None, static=True, **kwargs):
    '''
    Render the Figure 2-style panels for a user's own structure file --
    an experimental model, or a prediction from AlphaFold3, Boltz,
    ColabFold/AF2-Multimer, etc. -- rather than an RCSB entry.

    A local file has no SIFTS chain-to-UniProt mapping, so give either:
    - chain_to_uniprot: the mapping itself, e.g. {'A': 'P61158', ...}
      (values may also be free-text labels for chains that aren't human
      proteins), or
    - uniprot_IDs_list: the proteins in the complex, and each chain is
      matched to one of them by sequence (see map_chains_to_uniprot();
      call that directly to inspect the per-chain match report first).
    Chains left unmapped are drawn with an 'UNMAPPED:<chain>' label.

    Parameters
    ----------
    path to a .pdb/.ent/.cif/.mmcif file: str
    DataFrame of 293T PPIs: Pandas DataFrame (from getBioPlex('293T', ...))
    DataFrame of HCT116 PPIs: Pandas DataFrame (from getBioPlex('HCT116', ...))
    chain_to_uniprot: dict (optional)
    uniprot_IDs_list: list (optional)
    min_plddt: float (optional)
        For predicted structures: ignore atoms below this pLDDT (0-100)
        when finding direct contacts, so low-confidence regions can't
        create spurious contacts. Leave unset for experimental structures.
    static: bool (optional, default True)
        As in render_figure2_panels_for_uniprots(): True renders one
        static Figure via render_figure2_panels_static() (needs
        pymol-open-source); False uses render_figure2_panels().
    **kwargs
        Passed through to the render function (e.g. figsize,
        interact_dist_threshold).

    Returns
    -------
    Whatever the chosen render function returns:
    render_figure2_panels_static() -> Figure
    render_figure2_panels() -> (Figure, py3Dmol.view)
    '''
    from bioplexpy.analysis_funcs import is_local_structure_file, map_chains_to_uniprot

    if not is_local_structure_file(structure_file):
        raise FileNotFoundError(f"No such structure file: '{structure_file}'")
    if chain_to_uniprot is None:
        if uniprot_IDs_list is None:
            raise ValueError('Give either chain_to_uniprot or uniprot_IDs_list, so '
                             'the chains can be matched to BioPlex proteins.')
        chain_to_uniprot, report = map_chains_to_uniprot(structure_file,
                                                         uniprot_IDs_list)
        rejected = report[~report.accepted]
        if len(rejected):
            warnings.warn('Chain(s) not confidently matched by sequence, left '
                          f'unmapped: {list(rejected.chain)}. Inspect '
                          'map_chains_to_uniprot() output for details.')

    render_fn = render_figure2_panels_static if static else render_figure2_panels
    return render_fn(structure_file, None, bp_293t_df, bp_hct116_df,
                     chain_to_uniprot=chain_to_uniprot, min_plddt=min_plddt, **kwargs)
