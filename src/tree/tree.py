import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Optional
from networkx import DiGraph

class Tree:
    class Node:
        def __init__(self, val = None):
            self.val = val
            self.cildren: Optional[Tree.Node] = None

    root: Optional[Node]

    def __init__(self):
        self.root = None

def practice_tree():
    pass