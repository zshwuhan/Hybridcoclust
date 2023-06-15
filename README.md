## Brief description of models

- **Structure**
```math
	\begin{align*}
		\mathbf{X} \approx \mathbf{\Theta}= \boldsymbol{\beta} \odot
			\mathbf{R} B_{\mathbf{A}} \mathbf{C}^{\top},
	\end{align*}
```
-- **Objective Function**
```math
\begin{align*}\label{equ:NMTF_Bregman}
	\underset{\mathbf{R} \geqslant 0,B_{\mathbf{A}}, \mathbf{C} \geqslant 0}{\arg\min}
	D_{F^*}(\mathbf{X};\mathbf{R} B_{\mathbf{A}} \mathbf{C}^{\top})
	-\log_{\boldsymbol{\pi}} \mathbf{R}^{\top}\mathbf{1}_{m} -
	\log_{\boldsymbol{\rho}}\mathbf{C}^{\top}\mathbf{1}_{n},
\end{algin*}
```

## Cite
Please cite the following paper in your publication if you are using [Hybridcoclust]() in your research:

```bibtex
 @article{Hybridcoclust, 
    title={One Equivalence Between Exponential Family Latent Block Model and Bregman Non-negative Matrix Tri-factorization for Co-clustering.}, 
    DOI={Preprint}, 
    journal={preprint}, 
    author={Saeid Hoseinipour, Mina Aminghafari, Adel Mohammadpour}, 
    year={2023}
} 
```
## References

[1] [Mehrdad Farahani et al, Parsbert: Transformer-based model for Persian language understanding, Neural Processing Letters (2021).](https://github.com/Saeidhoseinipour/parsbert) 

[2] [Yoo et al, Orthogonal nonnegative matrix tri-factorization for co-clustering: Multiplicative updates on Stiefel manifolds (2010), 
	Information Processing and Management.](https://www.sciencedirect.com/science/article/abs/pii/S0306457310000038)
	
[3] [Ding et al, Orthogonal nonnegative matrix tri-factorizations for clustering, Proceedings of the 12th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (2008).](https://dl.acm.org/doi/abs/10.1145/1150402.1150420)

[4] [Long et al, Co-clustering by block value decomposition, Proceedings of the Eleventh ACM SIGKDD International Conference on Knowledge Discovery in Data 	Mining (2005).](https://dl.acm.org/doi/abs/10.1145/1081870.1081949)

[5] [Li et al, Nonnegative Matrix Factorization on Orthogonal Subspace (2010), Pattern Recognition Letters.](sciencedirect.com/science/article/abs/pii/S0167865509003651)

[6] [Cichocki et al, Non-negative matrix factorization with $\alpha$-divergence (2008), Pattern Recognition Letters.](https://www.sciencedirect.com/science/article/abs/pii/S0167865508000767)

[7] [Saeid, Hoseinipour et al, Orthogonal Parametric Non-negative Matrix Tri-Factorization with $\alpha$-Divergence for Co-clustering (2023), Expert Systems With Application.](https://doi.org/10.1016/j.eswa.2023.120680)

[8] [Saeid, Hoseinipour et al, Sparse Expoential Family Latent Block Model for Co-clustering (2023), Stat (preprint).]()

[9] [Saeid, Hoseinipour et al, Orthogonal Non-negative Matrix Tri-Factorization with $\alpha$-Divergence for Persian Co-clustering (2023), Iranian Journal of Science and Technology, Transactions of Electrical Engineering (preprint).]()



