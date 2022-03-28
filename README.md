## Morax means the mixing of CMOS and RRAM.   

### How to define morax model csv

##### csv name    
**model.csv**   
if model has BN/LN layer, use **model_norm.csv**

#### CONV Model                            
1. How to fill **[IC OC FS KS STR]**  
    **Linear**: fill IC OC, keep FS KS STR = 1 or empty
    **CONV**: fill all  
    **DWCONV**: fill all (Do make sure IC = OC)  
    **Residual** : fill IC FS, let OC = 1, keep KS STR = 1 or empty (OC must be 1 for maestro)    
                     (Note that res layers wonnt be shown in pytorch print models)  
    **Batchnorm**： fill IC OC FS, keep KS STR = 1 or empty (Do make sure IC = OC)   
    **PWCONV** : use 1x1 CONV  
    **TRCONV** : fill all  
    **NGCONV** :  fill all  

2. How to fill **[RP] [IDX] [APD]**  
          **[RP]**  
                   0: **this layer**  has no relu next  
                   1: **this layer**  has relu but no pooling next  
                   2 and above: **this layer has**  relu and pooling next, fill in the **[pooling kernel size]**    
          **[IDX]**      
                   default and usually -1, i.e. the previous layer.   
                   (the index is **just the layer order in our csv** )    
                   for multi-input layers (concat conv, residual, batchnorm, VDP, VADD, GEMM, etc), fill one index in **IDX**, the other in **APD**  
          **[APD]**  
                  for Linear:   
                    0: default (not first fc laye)        
                    1: just the (first fc layer)   
                    2 and above: (first fc layer) pooling kernel size of the pre-appending pooling layer    
                  for multi-input layers (concat conv, residual, VDP, VADD, VMUL, GEMM, etc): input index 2  
                  for TRCONV: dilation  
                  for NGCONV: group number  
                  for others: default 0 or empty

(All default blank could be： default value or NaN(blank))

| Layer                   | IC  | OC  | FS  | KS  | STR | TYP | RP  | IDX | APD                                            | Note               |
| ----------------------- | --- | --- | --- | --- | --- | --- | --- | --- | ---------------------------------------------- | ------------------ |
| Linear  [0]             | IC  | OC  |     |     |     | TYP | RP  | IDX | see [APD]                                      |                    |
| CONV    [1]             | IC  | OC  | FS  | KS  | STR | TYP | RP  | IDX | 0 (default) <br> -x (input index 2 for concat) |                    |
| DWCONV [2]              | IC  | OC  | FS  | KS  | STR | TYP | RP  | IDX | 0                                              | IC = OC            |
| Residual [3]            | IC  | 1   | FS  |     |     | TYP | RP  | IDX | input index 2                                  | OC = 1 for maestro |
| Batchnorm [4]           | IC  | OC  | FS  |     |     | TYP | RP  | IDX | 0                                              |                    |
| TRCONV [5]              | IC  | OC  | FS  | KS  | STR | TYP | RP  | IDX | dilation                                       |                    |
| NGCONV [6]              | IC  | OC  | FS  | KS  | STR | TYP | RP  | IDX | group number                                   |                    |
|                         |     |     |     |     |     |     |     |     |                                                |
| Pooling  [-1] (no nedd) | IC  | OC  | FS  | KS  | STR | TYP | RP  | IDX | 0                                              | IC = OC            |
| Softmax1d [-2]          | IC  |     |     |     |     | TYP | RP  | IDX | 0                                              | IC = OC            |

#### MLP/RNN/Attention Model  
1. **[MNK]**    
    **GEMM MK × KN**  
    **VecVec M**
    **VecMat or MatMat M * N** (M=Vinput N=Voutput)   

2. **[ACT]**
    1: this layer has activation
    0: this layer has no activation

3. **[IDX1][IDX2]**   
    All linear operators has two operands.   
    Use IDX1 IDX2 to indicate there index (-1 for previous layer).   
    IDX1 or IDX2 == 0： this operand is model parameters.

    for **CONCAT[13]**
    M: Head Number  
    IDX1: index of the beginning of head layer

| Layer               | M   | N   | K   | TYP | ACT | IDX1 | IDX2 | Note |
| ------------------- | --- | --- | --- | --- | --- | ---- | ---- | ---- |
| VDP [7]             | M   |     |     | TYP | ACT | IDX1 | IDX2 |      |
| VADD [8]            | M   |     |     | TYP | ACT | IDX1 | IDX2 |      |
| VMUL [9]            | M   |     |     | TYP | ACT | IDX1 | IDX2 |      |
| VMM [10] (= Linear) | M   | N   |     | TYP | ACT | IDX1 | IDX2 |      |
| GEMM [11]           | M   | N   | K   | TYP | ACT | IDX1 | IDX2 |      |
| MADD [12]           | M   | N   |     | TYP | ACT | IDX1 | IDX2 |      |
| Layernorm  [13]     | M   | N   |     | TYP | ACT | IDX1 |      |      |
| CONCAT  [14]        | M   |     |     | TYP |     | IDX1 |      |      |
|                     |     |     |     |     |     |      |      |      |
| Softmax1d [-2]      | M   |     |     | TYP | ACT | IDX1 |      |      |
| Softmax2d  [-3]     | M   | N   |     | TYP | ACT | IDX1 |      |      |
