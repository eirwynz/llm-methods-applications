# 11-967-LLMs-and-Applications
**Revised Executive Summary for Module 1.3 with Classical Mathematical Notation**  
*Step-by-Step Formulation for Self-Attention and Multi-Head Attention*  

---

### **Definitions**:  
Let:  
- \( n \in \mathbb{N} \): Sequence length.  
- \( d_{\text{model}} \in \mathbb{N} \): Embedding dimension (e.g., 512).  
- \( H \in \mathbb{N} \): Number of attention heads.  
- \( d_k = \frac{d_{\text{model}}}{H} \): Dimension of **queries** (\( \mathbf{q}_i \)) and **keys** (\( \mathbf{k}_i \)).  
- \( d_v = \frac{d_{\text{model}}}{H} \): Dimension of **values** (\( \mathbf{v}_i \)).  

---

### **Self-Attention Mechanism**:  
1. **Input Matrices**:  
   - Let \( \mathbf{Q} \in \mathbb{R}^{n \times d_k} \), \( \mathbf{K} \in \mathbb{R}^{n \times d_k} \), \( \mathbf{V} \in \mathbb{R}^{n \times d_v} \).  
   - Rows \( \mathbf{q}_i \in \mathbb{R}^{d_k} \), \( \mathbf{k}_i \in \mathbb{R}^{d_k} \), \( \mathbf{v}_i \in \mathbb{R}^{d_v} \) correspond to token \( i \).  

2. **Scaled Dot-Product Scores**:  
   \[
   \mathbf{S} = \mathbf{Q} \mathbf{K}^\top \in \mathbb{R}^{n \times n}, \quad s_{ij} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_k}}
   \]  

3. **Softmax Normalization**:  
   \[
   \alpha_{ij} = \frac{\exp(s_{ij})}{\sum_{m=1}^n \exp(s_{im})}, \quad \mathbf{A} = [\alpha_{ij}] \in \mathbb{R}^{n \times n}
   \]  

4. **Contextualized Output**:  
   \[
   \mathbf{C} = \mathbf{A} \mathbf{V} \in \mathbb{R}^{n \times d_v}, \quad \mathbf{c}_i = \sum_{j=1}^n \alpha_{ij} \mathbf{v}_j
   \]  

---

### **Multi-Head Attention**:  
1. **Projection Matrices per Head**:  
   For each head \( h \in \{1, \dots, H\} \):  
   - \( \mathbf{W}_h^Q \in \mathbb{R}^{d_{\text{model}} \times d_k} \): Query projection.  
   - \( \mathbf{W}_h^K \in \mathbb{R}^{d_{\text{model}} \times d_k} \): Key projection.  
   - \( \mathbf{W}_h^V \in \mathbb{R}^{d_{\text{model}} \times d_v} \): Value projection.  

2. **Compute Attention for Each Head**:  
   \[
   \mathbf{Q}_h = \mathbf{Q} \mathbf{W}_h^Q, \quad \mathbf{K}_h = \mathbf{K} \mathbf{W}_h^K, \quad \mathbf{V}_h = \mathbf{V} \mathbf{W}_h^V
   \]  
   \[
   \mathbf{C}_h = \text{softmax}\left(\frac{\mathbf{Q}_h \mathbf{K}_h^\top}{\sqrt{d_k}}\right) \mathbf{V}_h \in \mathbb{R}^{n \times d_v}
   \]  

3. **Concatenate and Project**:  
   \[
   \mathbf{M} = [\mathbf{C}_1 \,|\, \mathbf{C}_2 \,|\, \dots \,|\, \mathbf{C}_H] \in \mathbb{R}^{n \times H d_v}
   \]  
   \[
   \mathbf{O} = \mathbf{M} \mathbf{W}^O \in \mathbb{R}^{n \times d_{\text{model}}}, \quad \mathbf{W}^O \in \mathbb{R}^{H d_v \times d_{\text{model}}}
   \]  

---

### **Positional Encoding**:  
For position \( pos \in \{1, \dots, n\} \) and dimension \( i \in \{1, \dots, d_{\text{model}}\} \):  
\[
\mathbf{P} \in \mathbb{R}^{n \times d_{\text{model}}}, \quad \mathbf{P}_{pos,i} = 
\begin{cases} 
\sin\left(\frac{pos}{10000^{2j/d_{\text{model}}}}\right) & \text{if } i = 2j \\
\cos\left(\frac{pos}{10000^{2j/d_{\text{model}}}}\right) & \text{if } i = 2j + 1 
\end{cases}
\]  

---

### **Key Takeaways**:  
1. **Self-Attention**:  
   - **Input**: Token embeddings \( \mathbf{Q}, \mathbf{K}, \mathbf{V} \).  
   - **Output**: Contextualized embeddings \( \mathbf{C} \), where each \( \mathbf{c}_i \) aggregates information from all tokens weighted by relevance.  

2. **Multi-Head Attention**:  
   - Splits computation into \( H \) parallel heads to capture diverse relationships.  
   - Concatenates and projects results to preserve dimensionality.  

3. **Positional Encoding**:  
   - Injects positional information via fixed sinusoidal patterns.  

---

**Why This Notation Works**:  
- **Explicit Definitions**: Every variable is introduced with dimensions and purpose.  
- **No Abstractions**: Avoids "softmax(...)" or "Concat(...)" in favor of expanded mathematical operations.  
- **Implementation-Ready**: Directly maps to code (e.g., `Q @ K.T / np.sqrt(d_k)`).  

Let me know if you’d like further decomposition! 🧮
