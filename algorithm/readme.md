# Offline strategy for hybrid RRAM/CMOS mappindg



```python
def layerwaver_like_cmos_schedule(DAG):
    Schdeule = []
    MT, CT <- Morax(DAG, config) # MT: memory access time, CT: computation time
    update_dag(DAG, MT, CT) # 点权重为本层执行时间，边权重为toVertex的访存时间，不考虑cluster
    while not all_chdeuled():
        Select node i with min(MT[i] + CT[i]) # selection：AIMT, LAYWERWAVER
        Schdeule <- DAG[i]
        update DAG->vertex_weight with Schdeule
    return Schdeule
```

```python
def dfs(Schdeule, DAG):
    total_time = Morax(Schdeule, DAG)
    cp = critical_path(Schdeule, DAG)
    for node in cp:
        if dataflow_prefer(node) and no_rram_jam(Schdeule, node):
            if  node.mem < mem_left:
                Schdeule.node <- RRAM
                new_Schdeule <- Schdeule
                new_total_time <- Morax(new_Schdeule, DAG)
                select[node].gain <- total_time - new_total_time
    sort(select)
    # K sub dfs
    min_cost = Schdeule.new_total_time
    for node in sort.top_k():
        if select[node].gain < 0 
            break
        new_cost = dfs(select[node].schdule)
        if min_cost > new_cost
            min_cost <- new_cost
    return min_cost
    
```
