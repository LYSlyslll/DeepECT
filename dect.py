import collections
import math
from tqdm import tqdm

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.cluster import KMeans

from typing import Union, Tuple, Callable, Iterable, Dict, List, Any, Optional


class TreeNode:
    """
    A node in the DeepECT cluster tree.

    Attributes:
        id (int): A unique identifier for the node.
        parent (TreeNode, optional): The parent node in the tree.
        left (TreeNode, optional): The left child node.
        right (TreeNode, optional): The right child node.
        center (nn.Parameter): The learnable cluster center of the node.
        weight (torch.Tensor): The weight of the node, indicating its importance or occupancy.
        is_leaf (bool): True if the node is a leaf, False otherwise.
    """
    _next_id = 0

    def __init__(self, center: Union[torch.Tensor, nn.Parameter], parent: Union['TreeNode', None] = None, device: torch.device = 'cpu') -> None:
        self.id = TreeNode._next_id
        TreeNode._next_id += 1

        self.parent = parent
        self.left = None
        self.right = None

        if isinstance(center, torch.Tensor):
            self.center = nn.Parameter(center.clone().detach().to(device))
        else:
            self.center = center

        self.weight = torch.tensor(1.0, device=device)
        self.is_leaf = True

    def update_weight(self, assigned_ratio: float, alpha: float = 0.5) -> None:
        """
        Updates the node's weight using an exponential moving average.

        Args:
            assigned_ratio (float): The proportion of data points assigned to this node in the current batch.
            alpha (float): The smoothing factor for the moving average.
        """
        self.weight = (1 - alpha) * self.weight + alpha * assigned_ratio

    def __repr__(self) -> str:
        return f'Node(id={self.id}, leaf={self.is_leaf}, weight={self.weight:.2f})'


class DeepECT(nn.Module):
    """
    Deep Embedded Clustering Tree (DeepECT) model.
    This model combines a deep embedding model (like an autoencoder) with a dynamically
    growing and pruning binary tree structure for hierarchical clustering.
    """
    def __init__(self, embedding_model: nn.Module, latent_dim: int, embedding_model_loss: Optional[Callable] = None, device: torch.device = 'cpu') -> None:
        super().__init__()
        self.embedding_model = embedding_model.to(device)
        self.embedding_model_loss = embedding_model_loss or self.cosine_distance_loss
        self.latent_dim = latent_dim
        self.device = device

        # Initialize the tree with a single root node at the origin
        initial_center = torch.zeros(latent_dim, device=self.device)
        self.root = TreeNode(initial_center, device=self.device)

        # Store nodes in dictionaries and lists for easy access
        self.nodes = {self.root.id: self.root}
        self.leaf_nodes = [self.root]

    @staticmethod
    def cosine_distance_loss(x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        x1_norm = F.normalize(x1, p=2, dim=-1, eps=eps)
        x2_norm = F.normalize(x2, p=2, dim=-1, eps=eps)
        cos_sim = (x1_norm * x2_norm).sum(dim=-1)
        cos_distance = 1 - cos_sim
        return cos_distance.mean()

    def initialize_tree_from_embeddings(self, latent_vectors: torch.Tensor) -> None:
        """Initialize the root node center using precomputed latent vectors.

        Args:
            latent_vectors (torch.Tensor): A tensor of shape (N, latent_dim)
                containing latent representations used to set the initial
                cluster center for the root node.
        """
        if latent_vectors.ndim != 2 or latent_vectors.size(1) != self.latent_dim:
            raise ValueError(
                f"Expected latent vectors with shape (N, {self.latent_dim}), got {tuple(latent_vectors.shape)}"
            )

        latent_vectors = latent_vectors.to(self.device)
        with torch.no_grad():
            center = latent_vectors.mean(dim=0)
            if center.norm(p=2) > 0:
                center = F.normalize(center, p=2, dim=0)
            self.root.center.data.copy_(center)

    def get_tree_parameters(self) -> list:
        """
        Gathers all learnable parameters (centers of leaf nodes) from the tree.

        Returns:
            list: A list of nn.Parameter objects corresponding to the leaf node centers.
        """
        params = []
        for node in self.nodes.values():
            if node.is_leaf:
                params.append(node.center)
        return params

    def _find_closest_leaf(self, z: torch.Tensor) -> torch.Tensor:
        """
        Finds the closest leaf node for each vector in the batch `z`.

        Args:
            z (torch.Tensor): The batch of latent vectors, shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: A tensor of indices corresponding to the closest leaf node for each vector.
        """
        leaf_centers = torch.stack([node.center for node in self.leaf_nodes])
        distances = torch.cdist(z, leaf_centers)
        assignments = torch.argmin(distances, dim=1)
        return assignments

    def _update_inner_node_centers(self) -> None:
        """
        Recursively updates the centers of internal nodes from the bottom up.
        The center of an internal node is the weighted average of its children's centers.
        """
        def post_order_traversal(node):
            if node is None or node.is_leaf:
                return

            post_order_traversal(node.left)
            post_order_traversal(node.right)

            # Update center based on children's weights and centers
            total_weight = node.left.weight + node.right.weight
            if total_weight > 1e-8: # Avoid division by zero
                w_l, w_r = node.left.weight, node.right.weight
                mu_l, mu_r = node.left.center, node.right.center
                node.center.data = (w_l * mu_l + w_r * mu_r) / total_weight

        post_order_traversal(self.root)

    def _grow_tree(self, data_loader: Iterable, max_leaves: int, split_count: Union[int, float] = 1) -> bool:
        """
        The tree growing procedure. It identifies leaf nodes with the highest variance
        and splits them using 2-means clustering.

        Args:
            data_loader: The data loader to evaluate variance on.
            max_leaves (int): The maximum number of leaf nodes allowed in the tree.
            split_count (int or float): The number of nodes to split.
                - If int: The exact number of nodes to split.
                - If float (0.0, 1.0): The fraction of current leaf nodes to split.

        Returns:
            bool: True if the tree was grown, False otherwise.
        """
        self.embedding_model.eval()
        any_split_successful = False

        with torch.no_grad():
            # Get all latent representations from the data
            all_z = torch.cat([self.embedding_model(data.to(self.device))[0] for data in data_loader], dim=0)
            assignments = self._find_closest_leaf(all_z)

            # Find split candidates based on intra-cluster variance
            split_candidates = []
            for i, leaf in enumerate(self.leaf_nodes):
                assigned_z = all_z[assignments == i]
                if len(assigned_z) > 1:
                    variance = torch.sum((assigned_z - leaf.center)**2)
                    split_candidates.append({'variance': variance, 'node': leaf, 'data': assigned_z})

        if not split_candidates:
            self.embedding_model.train()
            return False

        split_candidates.sort(key=lambda x: x['variance'], reverse=True)

        # Determine the number of nodes to split
        current_leaves_count = len(self.leaf_nodes)
        if isinstance(split_count, float):
            num_to_split = max(1, int(current_leaves_count * split_count))
        else:
            num_to_split = int(split_count)

        num_to_split = min(num_to_split, len(split_candidates))
        available_slots = max_leaves - current_leaves_count
        num_to_split = min(num_to_split, available_slots)
        
        if num_to_split <= 0:
            self.embedding_model.train()
            return False

        nodes_to_split_info = split_candidates[:num_to_split]

        for split_info in nodes_to_split_info:
            node_to_split = split_info['node']
            data_for_split = split_info['data']

            # Split the node's data into two new clusters using 2-means
            kmeans = KMeans(n_clusters=2, n_init='auto', random_state=42)
            kmeans.fit(data_for_split.cpu().numpy())
            new_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=self.device)

            # Update the tree structure
            node_to_split.is_leaf = False
            left_child = TreeNode(new_centers[0], parent=node_to_split, device=self.device)
            right_child = TreeNode(new_centers[1], parent=node_to_split, device=self.device)
            node_to_split.left = left_child
            node_to_split.right = right_child

            self.nodes[left_child.id] = left_child
            self.nodes[right_child.id] = right_child

            self.leaf_nodes.remove(node_to_split)
            self.leaf_nodes.extend([left_child, right_child])
            any_split_successful = True
        
        if any_split_successful:
            self._update_inner_node_centers()
            
        self.embedding_model.train()
        return any_split_successful

    def _prune_tree(self, threshold: float = 0.1) -> bool:
        """
        The tree pruning procedure. It finds and removes 'dead' leaf nodes with weights
        below a given threshold.

        Args:
            threshold (float): The weight threshold below which a node is considered 'dead'.

        Returns:
            bool: True if the tree was pruned, False otherwise.
        """
        tree_was_pruned = False
        # Iterate over a copy of the list as we modify it
        for leaf in list(self.leaf_nodes):
            if leaf.weight < threshold and leaf.parent is not None:
                # Identify the dead node, its parent, sibling, and grandparent
                dead_node = leaf
                parent = dead_node.parent
                sibling = parent.left if parent.right == dead_node else parent.right
                grandparent = parent.parent

                if grandparent is not None:
                    # Rewire the grandparent to point directly to the sibling
                    if grandparent.left == parent:
                        grandparent.left = sibling
                    else:
                        grandparent.right = sibling
                    sibling.parent = grandparent
                else:
                    # If the parent was the root, the sibling becomes the new root
                    self.root = sibling
                    sibling.parent = None

                # Clean up the removed nodes
                self.leaf_nodes.remove(dead_node)
                del self.nodes[dead_node.id]
                del self.nodes[parent.id]

                tree_was_pruned = True

        return tree_was_pruned

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, x_hat = self.embedding_model(x)

        # Loss 1: Reconstruction Loss (L_REC)
        # This ensures the embedding retains information from the original data.
        loss_rec = self.embedding_model_loss(x, x_hat)
        if isinstance(loss_rec, (Tuple, List)):
            loss_rec, *_ = loss_rec

        if not self.leaf_nodes:
            return loss_rec, loss_rec, torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        leaf_assignments = self._find_closest_leaf(z)

        # Loss 2: Node Center Loss (L_NC)
        # Pulls leaf centers towards the mean of their assigned data points.
        loss_nc = torch.tensor(0.0, device=self.device)
        num_leaves_with_data = 0
        for i, leaf in enumerate(self.leaf_nodes):
            assigned_indices = (leaf_assignments == i).nonzero(as_tuple=True)[0]
            if len(assigned_indices) > 0:
                mean_z = z[assigned_indices].mean(dim=0).detach()
                loss_nc += self.cosine_distance_loss(
                    leaf.center.unsqueeze(0),
                    mean_z.unsqueeze(0)
                )
                num_leaves_with_data += 1
        if num_leaves_with_data > 0:
            loss_nc /= num_leaves_with_data

        # Loss 3: Node Data Compression Loss (L_DC)
        # Encourages data points within a subtree to lie along the direction
        # that separates the node from its sibling, improving cluster separation.
        loss_dc = torch.tensor(0.0, device=self.device)
        
        # Pre-compute ancestors for each leaf to speed up lookup
        leaf_to_ancestors = {}
        for leaf_idx, leaf_node in enumerate(self.leaf_nodes):
            ancestors = set()
            curr = leaf_node
            while curr is not None:
                ancestors.add(curr.id)
                curr = curr.parent
            leaf_to_ancestors[leaf_idx] = ancestors

        # Pre-compute projection vectors (direction between siblings)
        projection_vectors = {}
        nodes_to_process = [node for node in self.nodes.values() if node.parent is not None]
        for node in nodes_to_process:
            sibling = node.parent.left if node.parent.right == node else node.parent.right
            direction = node.center - sibling.center
            norm = torch.norm(direction)
            if norm > 1e-8:
                 projection_vectors[node.id] = direction / norm

        # Map each data point to all its ancestor nodes in the tree
        points_assigned_to_internal_nodes = collections.defaultdict(list)
        for point_idx, assigned_leaf_idx in enumerate(leaf_assignments):
            ancestors_ids = leaf_to_ancestors[assigned_leaf_idx.item()]
            for ancestor_id in ancestors_ids:
                points_assigned_to_internal_nodes[ancestor_id].append(point_idx)
        
        # Calculate loss for each relevant internal node
        num_nodes_for_dc = 0
        for node in nodes_to_process:
            if node.id in points_assigned_to_internal_nodes and node.id in projection_vectors:
                assigned_z_for_node = z[points_assigned_to_internal_nodes[node.id]]
                projection_vector = projection_vectors[node.id]

                diff = assigned_z_for_node - node.center.detach()
                diff_norm = F.normalize(diff, p=2, dim=-1)
                proj_norm = F.normalize(projection_vector, p=2, dim=0)
                cos_alignment = (diff_norm * proj_norm).sum(dim=-1)
                loss_dc += (1 - torch.abs(cos_alignment)).mean()
                num_nodes_for_dc += 1
        if num_nodes_for_dc > 0:
            loss_dc /= num_nodes_for_dc

        total_loss = loss_rec + loss_nc + loss_dc

        # Update node weights and internal centers after loss calculation
        with torch.no_grad():
            for i, leaf in enumerate(self.leaf_nodes):
                num_assigned = (leaf_assignments == i).sum().float() / len(leaf_assignments)
                leaf.update_weight(num_assigned, alpha=phase_params['weight_alpha'])
            self._update_inner_node_centers()

        return total_loss, loss_rec, loss_nc, loss_dc

    def _compute_leaf_purity(self, dataloader: Iterable) -> float:
        was_training = self.embedding_model.training
        self.embedding_model.eval()
        all_z: List[torch.Tensor] = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                z, _ = self.embedding_model(batch)
                all_z.append(z)

        if not all_z:
            if was_training:
                self.embedding_model.train()
            return 0.0

        all_z_tensor = torch.cat(all_z, dim=0)
        assignments = self._find_closest_leaf(all_z_tensor)
        leaf_purities: List[float] = []
        for i, leaf in enumerate(self.leaf_nodes):
            assigned_indices = (assignments == i).nonzero(as_tuple=True)[0]
            if assigned_indices.numel() == 0:
                continue
            assigned_z = all_z_tensor[assigned_indices]
            normalized_leaf_center = F.normalize(leaf.center.unsqueeze(0), p=2, dim=-1)
            normalized_assigned = F.normalize(assigned_z, p=2, dim=-1)
            similarities = torch.sum(normalized_assigned * normalized_leaf_center, dim=1)
            leaf_purities.append(similarities.mean().item())

        if was_training:
            self.embedding_model.train()

        if not leaf_purities:
            return 0.0
        return float(sum(leaf_purities) / len(leaf_purities))

    def train(self, dataloader: Iterable, iterations: int, lr: float, max_leaves: int, split_interval: int,
              pruning_threshold: float, split_count_per_growth: Union[int, float] = 1,
              evaluation_loader: Union[Iterable, None] = None):
        """
        The main training loop for the DeepECT model.

        Args:
            dataloader: The data loader for training.
            iterations (int): The total number of training iterations.
            lr (float): The learning rate for the optimizer.
            max_leaves (int): The maximum number of leaf nodes in the tree.
            split_interval (int): The number of iterations between tree growing procedures.
            pruning_threshold (float): The weight threshold for pruning dead nodes.
            split_count_per_growth (int or float): The number or fraction of nodes to split during each growth phase.
            evaluation_loader (Iterable, optional): Data loader used to evaluate leaf purity each epoch.
        """
        # Helper to re-create the optimizer when tree structure changes
        def create_optimizer(tree_params):
            params = [{'params': self.embedding_model.parameters(), 'weight_decay': 0.01}]
            if tree_params:
                params.append({'params': tree_params, 'weight_decay': 0.0})
            return optim.AdamW(params, lr=lr, betas=(0.9, 0.95))

        def create_scheduler(optimizer, last_step=-1):
            total_steps = max(1, iterations)
            warmup_steps = min(5000, total_steps)

            def lr_lambda(step: int) -> float:
                current_step = step + 1
                if current_step <= warmup_steps and warmup_steps > 0:
                    return current_step / float(max(1, warmup_steps))
                if total_steps == warmup_steps:
                    return 1.0
                progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                progress = min(max(progress, 0.0), 1.0)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_step)

        optimizer = create_optimizer(self.get_tree_parameters())
        scheduler = create_scheduler(optimizer)
        optimizer.zero_grad(set_to_none=True)
        bar = tqdm(total=iterations)
        data_iterator = iter(dataloader)
        iteration = 0
        scaler = amp.GradScaler(enabled=self.device.type == 'cuda')
        accumulation_steps = None
        loss_history: List[float] = []
        pending_losses = {'loss': 0.0, 'rec': 0.0, 'nc': 0.0, 'dc': 0.0}
        micro_step = 0
        batches_per_epoch = len(dataloader) if hasattr(dataloader, '__len__') else None
        batches_seen_in_epoch = 0
        epoch_purities: List[float] = []
        purity_loader = evaluation_loader if evaluation_loader is not None else dataloader
        aggressive_ratio = 0.6
        aggressive_phase_iters = max(1, int(iterations * aggressive_ratio))

        def get_phase_params(current_iteration: int) -> Dict[str, Any]:
            if current_iteration < aggressive_phase_iters:
                return {
                    'split_interval': 1,
                    'split_count': 0.25,
                    'pruning_threshold': 0.0,
                    'enable_pruning': False,
                    'enable_evaluation': False,
                    'weight_alpha': 0.05,
                }
            return {
                'split_interval': split_interval,
                'split_count': split_count_per_growth,
                'pruning_threshold': pruning_threshold,
                'enable_pruning': True,
                'enable_evaluation': True,
                'weight_alpha': 0.05,
            }

        active_purity_loader = purity_loader

        while iteration < iterations:
            phase_params = get_phase_params(iteration)
            try:
                data = next(data_iterator)
            except StopIteration:
                if batches_per_epoch is not None and batches_seen_in_epoch == batches_per_epoch and phase_params['enable_evaluation']:
                    epoch_purity = self._compute_leaf_purity(active_purity_loader)
                    epoch_purities.append(epoch_purity)
                data_iterator = iter(dataloader)
                data = next(data_iterator)

            batches_seen_in_epoch = 0 if batches_per_epoch is not None and batches_seen_in_epoch == batches_per_epoch else batches_seen_in_epoch
            batches_seen_in_epoch += 1
            super().train()
            data = data.to(self.device)
            structure_changed = False

            # Prune the tree to remove inactive nodes
            if phase_params['enable_pruning'] and self._prune_tree(threshold=phase_params['pruning_threshold']):
                structure_changed = True

            # Grow the tree by splitting high-variance nodes
            if iteration > 0 and iteration % phase_params['split_interval'] == 0 and len(self.leaf_nodes) < max_leaves:
                if self._grow_tree(dataloader, max_leaves=max_leaves, split_count=phase_params['split_count']):
                    structure_changed = True

            # If tree structure changed, we need a new optimizer for the new set of parameters
            if structure_changed:
                optimizer = create_optimizer(self.get_tree_parameters())
                scheduler = create_scheduler(optimizer, last_step=iteration - 1)
                optimizer.zero_grad(set_to_none=True)
                scaler = amp.GradScaler(enabled=self.device.type == 'cuda')

            if accumulation_steps is None:
                desired_global_batch = 4096 if self.device.type == 'cuda' else max(data.size(0), 1024)
                accumulation_steps = max(1, desired_global_batch // data.size(0))

            with amp.autocast(enabled=self.device.type == 'cuda'):
                loss, l_rec, l_nc, l_dc = self(data)

            pending_losses['loss'] = loss.detach().item()
            pending_losses['rec'] = l_rec.detach().item()
            pending_losses['nc'] = l_nc.detach().item()
            pending_losses['dc'] = l_dc.detach().item()

            scaled_loss = loss / accumulation_steps
            scaler.scale(scaled_loss).backward()
            micro_step += 1

            should_step = (micro_step % accumulation_steps == 0) or (iteration + 1 == iterations)
            if should_step:
                scaler.unscale_(optimizer)
                trainable_params = [
                    p for group in optimizer.param_groups for p in group['params'] if p.requires_grad
                ]
                if trainable_params:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                loss_history.append(pending_losses['loss'])

                bar.update(1)
                bar.set_description(
                    f'Iter {iteration} | '
                    f'Loss: {pending_losses["loss"]:.3f} | '
                    f'Rec: {pending_losses["rec"]:.3f} | NC: {pending_losses["nc"]:.3f} | '
                    f'DC: {pending_losses["dc"]:.3f} | '
                    f'Leaves: {len(self.leaf_nodes)}'
                )
                iteration += 1


        bar.close()
        if batches_per_epoch is not None and batches_seen_in_epoch > 0:
            final_phase_params = get_phase_params(iteration)
            if final_phase_params['enable_evaluation']:
                epoch_purity = self._compute_leaf_purity(active_purity_loader)
                epoch_purities.append(epoch_purity)

        return loss_history, epoch_purities
    def predict(self, data_loader: Iterable) -> torch.Tensor:
        """
        Assigns each data point from the data loader to a cluster (leaf node).

        Args:
            data_loader: The data loader containing the data to predict on.

        Returns:
            torch.Tensor: A tensor of cluster assignments for the entire dataset.
        """
        self.embedding_model.eval()
        with torch.no_grad():            
            all_z = []
            for data in tqdm(data_loader, desc="Predicting"):
                data = data.to(self.device)
                z, _ = self.embedding_model(data)
                all_z.append(z)

            all_z = torch.cat(all_z, dim=0)
            assignments = self._find_closest_leaf(all_z)

        return assignments
    
    def get_state(self) -> Dict[str, Any]:
        """
        Serializes the model's state into a dictionary for saving.
        This includes the embedding model's state and the tree structure.

        Returns:
            Dict[str, Any]: A dictionary containing the model's state.
        """
        tree_nodes_data = []
        for node in self.nodes.values():
            node_data = {
                'id': node.id,
                'center': node.center.detach().cpu(),
                'weight': node.weight,
                'is_leaf': node.is_leaf,
                'parent_id': node.parent.id if node.parent else None,
                'left_id': node.left.id if node.left else None,
                'right_id': node.right.id if node.right else None,
            }
            tree_nodes_data.append(node_data)
            
        tree_state = {
            'nodes_data': tree_nodes_data,
            'root_id': self.root.id if self.root else None
        }

        state = {
            'embedding_model_state_dict': self.embedding_model.state_dict(),
            'tree_state': tree_state,
        }
        return state

    def save_model(self, path: str) -> None:
        """
        Saves the complete model state (embedding model and tree) to a file.

        Args:
            path (str): The path to the file where the model will be saved.
        """
        state = self.get_state()
        torch.save(state, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Loads the complete model state (embedding model and tree) from a file.

        Args:
            path (str): The path to the file from which to load the model.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.embedding_model.load_state_dict(checkpoint['embedding_model_state_dict'])

        tree_state = checkpoint['tree_state']
        if tree_state.get('root_id') is None:
            self.root = None
            self.nodes = {}
            self.leaf_nodes = []
            return

        self.nodes.clear()
        self.leaf_nodes.clear()
        
        # First pass: create all nodes
        for node_data in tree_state['nodes_data']:
            node = TreeNode(center=node_data['center'], device=self.device)
            node.id = node_data['id']
            node.weight = node_data['weight']
            node.is_leaf = node_data['is_leaf']
            self.nodes[node.id] = node

        # Second pass: link nodes together
        for node_data in tree_state['nodes_data']:
            node = self.nodes[node_data['id']]
            if node_data['parent_id'] is not None:
                node.parent = self.nodes[node_data['parent_id']]
            if node_data['left_id'] is not None:
                node.left = self.nodes[node_data['left_id']]
            if node_data['right_id'] is not None:
                node.right = self.nodes[node_data['right_id']]

        self.root = self.nodes[tree_state['root_id']]
        self.leaf_nodes = [node for node in self.nodes.values() if node.is_leaf]

        # Ensure new node IDs don't conflict with loaded ones
        max_id = max([d['id'] for d in tree_state['nodes_data']]) if tree_state['nodes_data'] else -1
        TreeNode._next_id = max_id + 1
        
        print(f"Model loaded from {path}. Tree has {len(self.nodes)} nodes ({len(self.leaf_nodes)} leaves).")

    def prune_subtree(self, node_id_to_prune: int) -> None:
        """
        Manually prunes all descendants of a given node, making it a leaf.

        Args:
            node_id_to_prune (int): The ID of the node whose subtree should be pruned.
        """
        if node_id_to_prune not in self.nodes:
            print(f"Node with id {node_id_to_prune} not found.")
            return

        target_node = self.nodes[node_id_to_prune]
        if target_node.is_leaf:
            print(f"Node {node_id_to_prune} is already a leaf.")
            return

        # Find all descendants using a queue (BFS)
        descendants_to_delete = []
        queue = collections.deque()
        if target_node.left:
            queue.append(target_node.left)
        if target_node.right:
            queue.append(target_node.right)
            
        while queue:
            current_node = queue.popleft()
            descendants_to_delete.append(current_node)
            if current_node.left:
                queue.append(current_node.left)
            if current_node.right:
                queue.append(current_node.right)

        print(f"Pruning subtree of node {target_node.id}. Deleting {len(descendants_to_delete)} descendants.")
        
        # Remove descendants from global lists
        for node in descendants_to_delete:
            del self.nodes[node.id]
            if not node.is_leaf:
                continue
            if node in self.leaf_nodes:
                self.leaf_nodes.remove(node)
        
        # Make the target node a leaf
        target_node.left = None
        target_node.right = None
        target_node.is_leaf = True
        if target_node not in self.leaf_nodes:
            self.leaf_nodes.append(target_node)
        
        # Update internal node centers after pruning
        self._update_inner_node_centers()