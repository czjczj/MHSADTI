from collections import defaultdict
import os
import pickle
import sys
import numpy as np
from rdkit import Chem

def create_atoms(mol):
    """
    ['C', 'C', 'C', 'C', 'C', ('C', 'aromatic'), ('C', 'aromatic'), ('C', 'aromatic'), ('C', 'aromatic'), ('C', 'aromatic'), ('C', 'aromatic'), 'C', 'C', 'N', 'C', 'C', 'C', 'C', 'C', 'C', 'O', 'C', 'O', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H']
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 0, 3,
       0, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
       4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    一个分子中的不同 c 也可能是不同结构的。
    :param mol:
    :return:
    """
    """Create a list of atom (e.g., hydrogen and oxygen) IDs
    considering the aromaticity."""
    atoms = [a.GetSymbol() for a in mol.GetAtoms()] #如果是a.GetIdx() 那么是从0开始单独对于原子编号的，就算原子的Symbol相同
    for a in mol.GetAromaticAtoms(): #得到其中的 芳香厅，区别对待这几个原子符号，得到芳香厅从 0 开始单独编号的位置，然后进行单独的计算其中的值
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol):
    """Create a dictionary, which each key is a node ID
    and each value is the tuples of its neighboring node
    and bond (e.g., single and double) IDs."""
    #对于每一个节点，添加（） 一个字典，双向的， （边的另一个节点，边的类型） 需要确定每个节点的标号载不同的分子中是不是相同的
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds(): #Idx 都是从0开始顺序编号的的
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict

#atoms type:list len:分子中原子个数 val:每个分子的symbol 顺序编号（考虑芳烃）
#i_jbond_dict  type:dict  len:分子中原子个数  key:0....原子个数-1  val: [顺序编号，键的类型]
def extract_fingerprints(atoms, i_jbond_dict, radius):
    """Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using Weisfeiler-Lehman algorithm."""

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges
            (i.e., r-radius subgraphs or fingerprints)."""
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint]) #fingerprint 中键是  [原子编号，(原子编号，keytype),(),()] 这样的形式
            nodes = fingerprints

            """Also update each edge ID considering two nodes
            on its both sides."""
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)

def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)


def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='
    words = [word_dict[sequence[i:i+ngram]]
             for i in range(len(sequence)-ngram+1)]
    return np.array(words)


def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)

if __name__ == "__main__":
    DATASET, radius, ngram = ('drugbank',2,3)
    with open(f'./{DATASET}.txt', 'r') as f:
        data_list = f.read().strip().split('\n')
    """Exclude data contains '.' in the SMILES format."""
    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    N = len(data_list)
    dir_input = (f'./radius{radius}_ngram{ngram}1/')
    os.makedirs(dir_input, exist_ok=True)

    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))
    word_dict = defaultdict(lambda: len(word_dict))
    Smiles, compounds, adjacencies, proteins, interactions = '', [], [], [], []
    for no, data in enumerate(data_list):
        smiles, sequence, interaction = data.strip().split()
        Smiles += smiles + '\n'
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))  # Consider hydrogens.
        atoms = create_atoms(mol)#得到smiles中每一个原子的数字表示
        i_jbond_dict = create_ijbonddict(mol)
        fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius)
        compounds.append(fingerprints)
        adjacency = create_adjacency(mol)
        adjacencies.append(adjacency) #临界矩阵 nxn维度，每一个值为 1或者0
        words = split_sequence(sequence, ngram)
        proteins.append(words)
        interactions.append(np.array([float(interaction)]))

    with open(f'{dir_input}Smiles.txt', 'w') as f:
        f.write(Smiles)
    np.save(f'{dir_input}compounds', compounds)
    np.save(f'{dir_input}adjacencies', adjacencies)
    np.save(f'{dir_input}proteins', proteins)
    np.save(f'{dir_input}interactions', interactions) #label
    dump_dictionary(fingerprint_dict, dir_input + 'fingerprint_dict.pickle')
    dump_dictionary(atom_dict, dir_input + 'atom_dict.pickle')
    dump_dictionary(bond_dict, dir_input + 'bond_dict.pickle')
    dump_dictionary(edge_dict, dir_input + 'edge_dict.pickle')
    dump_dictionary(word_dict, dir_input + 'word_dict.pickle')
    print('The preprocess of ' + DATASET + ' dataset has finished!')
