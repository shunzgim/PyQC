from .import opt_einsum as oe


class Node:
    def __init__(self, tensor):
        self.tensor = tensor
        self.edges = []
        self.rank = tensor.ndim
        for i in range(self.rank):
            self.edges.append(0)

    @property
    def shape(self):
        return self.tensor.shape


class TensorNetwork:
    """
    """    
    def __init__(self):
        self.contract_expression = None
    
    def _get_einsum_arg(self, all_nodes):
        einsumarg = ''
        tensor_list = []
        tensor_shape = []
        lens = len(all_nodes)
        for i in range(lens):
            node = all_nodes[i]
            tensor_list.append(node.tensor)
            tensor_shape.append(node.shape)
            edges = node.edges
            for s in edges:
                einsumarg += oe.get_symbol(s)
            if i<lens-1:
                einsumarg += ','
        return einsumarg, tensor_list, tensor_shape

    def get_contract_path(self, einsumarg, tensor_list):
        path = oe.contract_path(einsumarg, *tensor_list)
        return path
            
    def find_path(self, all_nodes):
        einsumarg, _, tensor_shape = self._get_einsum_arg(all_nodes)
        self.contract_expression = oe.contract_expression(einsumarg, *tensor_shape)
        
    def contract(self, all_nodes):
        _, tensor_list, _ = self._get_einsum_arg(all_nodes) 
        return self.contract_expression(*tensor_list)
    

