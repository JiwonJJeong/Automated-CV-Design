"""AMINO: generating a minimally redundant set of order parameters through
clustering of mutual information based distances. Method by Ravindra, Smith,
and Tiwary. Code maintained by Ravindra and Smith.

This is the serial kernel density estimation version.

Read and cite the following when using this method:

https://pubs.rsc.org/--/content/articlehtml/2020/me/c9me00115h

This is the serial kernel density estimation version.
Refactored for pipeline stability:
- Fixed NoneType binning bug
- Fixed ZeroDivisionError on constant features
- Returns list of strings for DataFrame compatibility
"""

import numpy as np
from sklearn.neighbors import KernelDensity
import copy

class OrderParameter:
    """Order Parameter (OP) class - stores OP name and trajectory"""

    def __init__(self, name, traj):
        self.name = name
        t = np.array(traj).reshape([-1,1])
        std_val = np.std(t)
        
        # SAFETY FIX: Prevent division by zero for constant features
        if std_val > 1e-12:
            self.traj = t / std_val
        else:
            self.traj = np.zeros_like(t)

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return str(self.name)
        
    def __repr__(self):
        return str(self.name)

# Memoizes distance computation between OP's to prevent re-calculations
class Memoizer:
    def __init__(self, bins, bandwidth, kernel, weights=None):
        self.memo = {}
        self.bins = int(bins) # Ensure integer
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.weights = weights

    def d2_bin(self, x, y):
        KD = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
        KD.fit(np.column_stack((x,y)), sample_weight=self.weights)
        
        grid1 = np.linspace(np.min(x), np.max(x), self.bins)
        grid2 = np.linspace(np.min(y), np.max(y), self.bins)
        
        mesh = np.meshgrid(grid1, grid2)
        data = np.column_stack((mesh[0].reshape(-1,1), mesh[1].reshape(-1,1)))
        
        samp = KD.score_samples(data)
        samp = samp.reshape(self.bins, self.bins)
        p = np.exp(samp)/np.sum(np.exp(samp))

        return p

    def distance(self, OP1, OP2):
        index1 = str(OP1.name) + " " + str(OP2.name)
        index2 = str(OP2.name) + " " + str(OP1.name)

        memo_val = self.memo.get(index1, False) or self.memo.get(index2, False)
        if memo_val:
            return memo_val

        x = OP1.traj
        y = OP2.traj
        
        # Calculate P(x,y)
        p_xy = self.d2_bin(x, y)
        
        # Marginal probabilities
        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)

        p_x_times_p_y = np.tensordot(p_x, p_y, axes=0)
        
        # Mask zeros to avoid log(0)
        with np.errstate(divide='ignore', invalid='ignore'):
            # Mutual Information
            log_term = np.ma.log(np.ma.divide(p_xy, p_x_times_p_y))
            info = np.sum(p_xy * log_term)
            
            # Entropy
            log_p_xy = np.ma.log(p_xy)
            entropy = np.sum(-1 * p_xy * log_p_xy)

        # Safety against zero entropy (perfect correlation/constant)
        if entropy < 1e-12:
            return 0.0

        output = max(0.0, (1 - (info / entropy)))
        self.memo[index1] = output
        return output

class DissimilarityMatrix:
    def __init__(self, size, mut):
        self.size = size
        self.matrix = [[] for i in range(size)]
        self.mut = mut
        self.OPs = []

    def add_OP(self, OP):
        if len(self.OPs) == self.size:
            mut_info = []
            existing = []
            for i in range(len(self.OPs)):
                mut_info.append(self.mut.distance(self.OPs[i], OP))
                product = 1
                for j in range(len(self.OPs)):
                    if not i == j:
                        product = product * self.matrix[i][j]
                existing.append(product)
            
            update = False
            difference = None
            old_OP = -1
            
            for i in range(len(self.OPs)):
                candidate_info = 1
                for j in range(len(self.OPs)):
                    if not i == j:
                        candidate_info = candidate_info * mut_info[j]
                
                if candidate_info > existing[i]:
                    update = True
                    diff = candidate_info - existing[i]
                    if difference is None or diff > difference:
                        difference = diff
                        old_OP = i
            
            if update:
                mut_info[old_OP] = self.mut.distance(OP, OP)
                self.matrix[old_OP] = mut_info
                self.OPs[old_OP] = OP
                for i in range(len(self.OPs)):
                    self.matrix[i][old_OP] = mut_info[i]
        else:
            for i in range(len(self.OPs)):
                mut_info = self.mut.distance(OP, self.OPs[i])
                self.matrix[i].append(mut_info)
                self.matrix[len(self.OPs)].append(mut_info)
            self.matrix[len(self.OPs)].append(self.mut.distance(OP, OP))
            self.OPs.append(OP)

def distortion(centers, ops, mut):
    dis = 0.0
    for i in ops:
        min_val = np.inf
        for j in centers:
            tmp = mut.distance(i, j)
            if tmp < min_val:
                min_val = tmp
        dis = dis + (min_val * min_val)
    return 1 + (dis ** (0.5))

def grouping(centers, ops, mut):
    groups = [[] for i in range(len(centers))]
    for OP in ops:
        group = 0
        min_dist = mut.distance(OP, centers[0])
        for i in range(1, len(centers)):
            tmp = mut.distance(OP, centers[i])
            if tmp < min_dist:
                group = i
                min_dist = tmp
        groups[group].append(OP)
    return groups

def group_evaluation(ops, mut):
    if not ops:
        return None
    center = ops[0]
    min_distortion = distortion([ops[0]], ops, mut)
    for i in ops:
        tmp = distortion([i], ops, mut)
        if tmp < min_distortion:
            center = i
            min_distortion = tmp
    return center

def cluster(ops, seeds, mut):
    old_centers = []
    centers = copy.deepcopy(seeds)

    while (set(centers) != set(old_centers)):
        old_centers = copy.deepcopy(centers)
        centers = []
        groups = grouping(old_centers, ops, mut)

        for i in range(len(groups)):
            # Safety check for empty groups
            if len(groups[i]) == 0:
                centers.append(old_centers[i])
            else:
                result = group_evaluation(groups[i], mut)
                centers.append(result)

    return centers

def find_ops(old_ops, max_outputs=20, bins=20, bandwidth=None, kernel='epanechnikov',
             distortion_filename=None, return_memo=False, weights=None):

    # 1. FIX: Calculate bins FIRST before Memoizer needs it
    if bins is None:
        bins = int(np.ceil(np.sqrt(len(old_ops[0].traj))))
    else:
        bins = int(bins)

    if bandwidth is None:
        if kernel == 'parabolic':
            kernel = 'epanechnikov'
        if kernel == 'epanechnikov':
            bw_constant = 2.2
        else:
            bw_constant = 1
        
        if weights is None:
            n = np.shape(old_ops[0].traj)[0]
        else:
            weights = np.array(weights)
            n = np.sum(weights)**2 / np.sum(weights**2)
        bandwidth = bw_constant*n**(-1/6)
        print('Selected bandwidth: ' + str(bandwidth))

    # 2. NOW create Memoizer
    mut = Memoizer(bins, bandwidth, kernel, weights)
    
    distortion_array = []
    num_array = []
    op_dict = {}

    while (max_outputs > 0):
        print(f"Checking {max_outputs} order parameters...")
        matrix = DissimilarityMatrix(max_outputs, mut)
        for i in old_ops:
            matrix.add_OP(i)
        for i in old_ops[::-1]:
            matrix.add_OP(i)

        num_array.append(len(matrix.OPs))
        seed = list(matrix.OPs)
        
        tmp_ops = cluster(old_ops, seed, mut)
        op_dict[len(seed)] = tmp_ops
        distortion_array.append(distortion(tmp_ops, old_ops, mut))
        max_outputs = max_outputs - 1

    # Determining number of clusters
    num_ops = 1  # FIX: Start at 1
    
    # Analyze jumps in distortion
    for dim in range(1, 11):
        neg_expo = np.array(distortion_array) ** (-0.5 * dim)
        jumps = []
        for i in range(len(neg_expo) - 1):
            jumps.append(neg_expo[i] - neg_expo[i + 1])
        
        if not jumps: continue

        min_index = np.argmax(jumps)
        if num_array[min_index] > num_ops:
            num_ops = num_array[min_index]

    if distortion_filename is not None:
        np.save(distortion_filename, distortion_array[::-1])

    # Safety check: ensure num_ops exists
    if num_ops not in op_dict:
        num_ops = max(op_dict.keys())

    # --- THE RETURN TYPE FIX ---
    # Return strings, not objects
    final_names = [str(op.name) for op in op_dict[num_ops]]
    
    if return_memo:
        return final_names, mut
        
    return final_names