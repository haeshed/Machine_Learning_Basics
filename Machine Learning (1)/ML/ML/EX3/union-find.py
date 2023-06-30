class UnionFind:
    def __init__(self, size):
        self.rank = [0 for i in range(size)]
        self.parent = [i for i in range(size)]
    
    def Find(self, x):
        parent = self.parent[x]
        while parent != x:
            x = parent
            parent = self.parent[x]
        return parent
    
    def Union(self, x, y):
        root_x = self.Find(x)
        root_y = self.Find(y)
        if root_x == root_y:
            return
        else:
            if self.rank[root_x] >= self.rank[root_y]:
                self.parent[root_y] = root_x
                self.rank[root_x] = max(self.rank[root_y] + 1, self.rank[root_x])
            else:
                self.parent[root_x] = root_y
                self.rank[root_y] = max(self.rank[root_x] + 1, self.rank[root_y])
    def Same(self, x, y):
        return self.Find(x) == self.Find(y)