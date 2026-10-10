"""Ordered-prefix identity and four deliberately small cache models."""
from collections import OrderedDict, deque


class Tree:
    def __init__(self):
        self.edges = [{}]
        self.parent = [0]
        self.depth = [0]
        self.extent = [0]

    def path(self, hashes):
        node, path = 0, []
        for value in hashes:
            if value not in self.edges[node]:
                child = len(self.parent)
                self.edges[node][value] = child
                self.edges.append({})
                self.parent.append(node)
                self.depth.append(self.depth[node] + 1)
                self.extent.append(0)
            node = self.edges[node][value]
            path.append(node)
        return path

    def observe(self, path):
        for node in path:
            self.extent[node] = max(self.extent[node], len(path))


class Cache:
    def __init__(self, capacity, policy, tree):
        if capacity <= 0 or policy not in ('block_lru', 'prefix_lru', 'rr_tail', 'cost_tail'):
            raise ValueError('invalid cache configuration')
        self.capacity, self.policy, self.tree = capacity, policy, tree
        self.blocks = OrderedDict()
        self.children = {}
        self.leaves = set()
        self.rotation = deque()
        self.leaf_generation = {}
        self.generation = 0
        self.writes = self.evictions = self.rejected_batches = 0

    def hit(self, path):
        count = 0
        for node in path:
            if node not in self.blocks:
                break
            count += 1
        return count

    def touch(self, path):
        # Mirrors pinned native LRUCachePolicy.touch: early prefix most recent.
        for node in reversed(path):
            if node in self.blocks:
                self.blocks.move_to_end(node)

    def add(self, node):
        self.blocks[node] = None
        self.writes += 1
        if self.policy == 'block_lru':
            return
        parent = self.tree.parent[node]
        self.children[node] = 0
        self.make_leaf(node)
        if parent:
            assert parent in self.blocks
            self.children[parent] += 1
            self.leaves.discard(parent)

    def make_leaf(self, node):
        self.leaves.add(node)
        self.generation += 1
        self.leaf_generation[node] = self.generation
        self.rotation.append((node, self.generation))

    def remove(self, node):
        del self.blocks[node]
        self.evictions += 1
        if self.policy == 'block_lru':
            return
        assert node in self.leaves and self.children[node] == 0
        self.leaves.remove(node)
        del self.children[node]
        parent = self.tree.parent[node]
        if parent:
            self.children[parent] -= 1
            if not self.children[parent]:
                self.make_leaf(parent)

    def victim(self):
        if self.policy == 'prefix_lru':
            return next(node for node in self.blocks if node in self.leaves)
        if self.policy == 'rr_tail':
            while True:
                node, generation = self.rotation.popleft()
                if node in self.leaves and self.leaf_generation[node] == generation:
                    return node
        # Balance estimated lost attention work, using only lengths observed so far.
        # Quadratic length is a screening proxy, not a measured cost model.
        return min(self.leaves, key=lambda n: (self.tree.extent[n] ** 2 -
                   (self.tree.depth[n] - 1) ** 2, n))

    def store(self, path, chunk=32):
        self.tree.observe(path)
        if self.policy == 'block_lru':
            for start in range(0, len(path), chunk):
                keys = path[start:start + chunk]
                missing = [n for n in keys if n not in self.blocks]
                needed = len(missing) - (self.capacity - len(self.blocks))
                protected = set(keys)
                candidates = [n for n in self.blocks if n not in protected]
                if needed > len(candidates):
                    self.rejected_batches += 1
                    continue
                for node in candidates[:max(needed, 0)]:
                    self.remove(node)
                for node in missing:
                    self.add(node)
                self.touch(path)
            return
        for node in path:
            if node in self.blocks:
                continue
            # Never write an unusable suffix after its prefix was reclaimed.
            if self.tree.parent[node] and self.tree.parent[node] not in self.blocks:
                break
            if len(self.blocks) == self.capacity:
                self.remove(self.victim())
                if self.tree.parent[node] and self.tree.parent[node] not in self.blocks:
                    break
            self.add(node)
        self.touch(path)

    def validate(self):
        assert len(self.blocks) <= self.capacity
        if self.policy != 'block_lru':
            assert all(not self.tree.parent[n] or self.tree.parent[n] in self.blocks for n in self.blocks)
            assert self.leaves == {n for n in self.blocks if self.children[n] == 0}
