# Softmax 函数（亦称归一化指数函数）

Softmax 可以把实数向量  

\[
\mathbf{z} = (z_1, z_2, \dots, z_K)
\]

映射为概率分布。对第 \(i\) 维：

\[
\operatorname{softmax}(\mathbf{z})_i
= \frac{e^{z_i}}{\displaystyle\sum_{j=1}^{K} e^{z_j}}\;.
\]

---

## 核心要点

### 输出取值范围
- 每个分量 \(\in(0,1)\)。  
- 所有分量之和恒为 1，可直接解释为类别概率。

### 特征
- **平移不变性**：\(\operatorname{softmax}(\mathbf{z}+c)=\operatorname{softmax}(\mathbf{z})\)。  
  通常用 \(\tilde{\mathbf{z}}=\mathbf{z}-\max(\mathbf{z})\) 做数值稳定化，避免 \(\exp\) 溢出。  
- **温度可调**：\(\operatorname{softmax}(\mathbf{z}/T)\)。  
  - \(T>1\)：分布更平坦。  
  - \(T<1\)：分布更尖锐。  

### 梯度 / 雅可比
- 对自身：\(\displaystyle \frac{\partial y_i}{\partial z_i} = y_i(1-y_i)\)。  
- 对互异分量：\(\displaystyle \frac{\partial y_i}{\partial z_j} = -y_i y_j \quad(i\neq j)\)。  
- 与交叉熵组合时，梯度简化为 \(y - y^{\text{true}}\)，常用于多分类。

---

## 应用场景
1. **多分类输出层**：神经网络、逻辑回归、多项式回归等。  
2. **注意力机制**：将注意力分数归一化为权重。  
3. **温度采样 / 多臂赌博机**：把评分转成概率后进行抽样。  

---

## 与 Sigmoid 的区别
- **Sigmoid** 只针对二分类（输出单值）；**Softmax** 支持 \(K>2\) 的多分类。  
- Sigmoid 不保证多类别概率和为 1；Softmax 天然满足归一化。

---

简而言之，Softmax 能把未归一化的“分数”转成总和为 1 的概率分布，可微分且易于数值稳定，是现代深度学习多分类任务的默认选择
