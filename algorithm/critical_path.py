from typing import List, Tuple
from collections import deque

class CriticalPath:

    big_num = 1e9

    class edge_node:
        def __init__(self, to, w, next):
            self.to = to
            self.w = w
            self.next = next

    def add_edge(self, u: int, v: int, w: float):
        en = self.edge_node(v, w, self.head[u])
        self.edges.append(en)
        self.head[u] = self.edge_idx
        self.edge_idx += 1

    # Construct the graph
    # n: number of node in graph
    # edge_list: [(u, v, w), ...], 0 <= u < n, 0 <= v < n, w >= 0
    def __init__(self, n: int, edge_list: List[Tuple[int, int, float]]):
        self.n = n
        self.head = [-1 for i in range(n)]
        self.edges = []
        self.edge_idx = 0
        for it in edge_list:
            self.add_edge(it[0], it[1], it[2])

    # Compute the longest path by SPFA 
    # src: source node
    # dst: destination node
    # return: (length, path_list)
    # If there is no path between src and dst, or there is negative cycle in the graph, then return (-1, []) 
    def SPFA(self, src: int, dst: int) -> Tuple[int, List[int]]:
        dist = [self.big_num for i in range(self.n)]
        visit = [False for i in range(self.n)]
        outcnt = [0 for i in range(self.n)]
        dist[src] = 0
        visit[src] = 1
        pre = [-1 for i in range(self.n)]
        q = deque()
        q.append(src)
        while len(q) > 0:
            top = q.pop()
            visit[top] = 0
            outcnt[top] += 1
            # check the negative cycle
            if (outcnt[top] > self.n): 
                return (-1, [])
            j = self.head[top]
            while j != -1:
                to = self.edges[j].to
                # use negative weight to find the longest path 
                tw = dist[top] - self.edges[j].w
                if dist[to] > tw:
                    dist[to] = tw
                    pre[to] = top
                    if not visit[to]:
                        visit[to] = True
                        q.append(to)
                j = self.edges[j].next
        if dist[dst] == self.big_num:
            return (-1, [])
        pt = []
        j = dst
        while j != -1:
            pt.append(j)
            j = pre[j]
        pt.reverse()
        return (-dist[dst], pt)

    # Compute the critical paths by Topological Sort and Dynamic Programming
    # The input graph need to be DAG 
    # src and dst can be auto determined by the degrees
    # This method can find all edges in more than one critical paths, namely critical graph
    def topo_dp(self) -> List[Tuple[int, int, float]]:
        # Determine the src and dst
        degree_in = [0 for i in range(self.n)]
        degree_out = [0 for i in range(self.n)]
        etv = [0.0 for i in range(self.n)] # Earliest Time of Vertex
        ltv = [self.big_num for i in range(self.n)] # Lastest Time of Vertex
        for u in range(self.n):
            j = self.head[u]
            while j != -1:
                v = self.edges[j].to
                degree_in[v] += 1
                degree_out[u] += 1
                j = self.edges[j].next
        src = []
        dst = []
        for i in range(self.n):
            if degree_in[i] == 0:
                src.append(i)
            if degree_out[i] == 0:
                dst.append(i)

        # Compute the Earliest Time of Vertex
        topo = src
        i = 0
        iq = len(topo)
        while i < iq:
            u = topo[i]
            j = self.head[u]
            while j != -1:
                v = self.edges[j].to
                etv[v] = max(etv[v], etv[u] + self.edges[j].w)
                degree_in[v] -= 1
                if degree_in[v] == 0:
                    topo.append(v)
                    iq += 1
                j = self.edges[j].next
            i += 1
        if iq < self.n:
            return []

        # Compute the Lastest Time of Vertex
        for d in dst:
            ltv[d] = etv[d]
        topo.reverse()
        for u in topo:
            j = self.head[u]
            while j != -1:
                v = self.edges[j].to
                ltv[u] = min(ltv[u], ltv[v] - self.edges[j].w)
                j = self.edges[j].next

        # Check the critical paths
        topo.reverse()
        path = []
        for u in topo:
            j = self.head[u]
            while j != -1:
                v = self.edges[j].to
                w = self.edges[j].w
                if etv[u] == ltv[v] - w:
                    path.append([u, v, w])
                j = self.edges[j].next

        return path

# AOE 网络的关键路径
# 等价于求图的最长路径
# 可以用SPFA算法直接得到
# 或者通过拓扑排序和动态规划法得到

if __name__ == "__main__":
    # 边列表输入，权重只在边上。如果有点权需要拆点转点权为边权
    edge_list = [[0, 1, 3], [0, 2, 2], [1, 3, 2], [1, 4, 3], [2, 3, 4], [2, 5, 3], [3, 5, 2], [4, 5, 1]]
    critical_path = CriticalPath(6, edge_list)
    ret = critical_path.SPFA(0, 5)
    print("length of critical path:   ", ret[0])
    print("One critical path (node list): ", ret[1])

    ret = critical_path.topo_dp()
    print("All critical paths (edge list): ", ret)
