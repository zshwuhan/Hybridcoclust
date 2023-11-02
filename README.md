## Brief description of models

- **Structure**
```math
	\begin{align*}
		\mathbf{X} \approx \mathbf{\Theta}= \boldsymbol{\beta} \odot
			\mathbf{R} B_{\mathbf{A}} \mathbf{C}^{\top},
	\end{align*}
```
- **Bregman objective function**
```math
\begin{align*}
	\underset{\mathbf{R},B_{\mathbf{A}}, \mathbf{C} > 0}{\arg\min}
	D_{F^*}(\mathbf{X};\mathbf{R} B_{\mathbf{A}} \mathbf{C}^{\top})
	-\log_{\boldsymbol{\pi}} \mathbf{R}^{\top}\mathbf{1}_{m} -
	\log_{\boldsymbol{\rho}}\mathbf{C}^{\top}\mathbf{1}_{n},
\end{align*}
```

- **Complete log-likelihood function**
```math
\begin{align*}
\underset{\mathbf{R},B_{\mathbf{A}}, \mathbf{C} > 0}{\arg\max}
	L(\mathbf{R},\mathbf{C}; \boldsymbol{\pi}, \boldsymbol{\rho}, \mathbf{A})
	\propto
	\log_{\boldsymbol{\pi}} \mathbf{R}^{\top}\mathbf{1}_{m}
		+
	\log_{\boldsymbol{\rho}}\mathbf{C}^{\top}\mathbf{1}_{n}
	+
	\mathbf{1}^{\top}_{g}(\mathbf{R}^{\top}S_{\mathbf{X}}\mathbf{C} \odot  B_{\mathbf{A}} -  F_{\mathbf{A}})\mathbf{1}_{s}.
\end{align*}
```
- **Hybrid objective function**
```math
\begin{align*}
	\underset{\mathbf{R},\mathbf{C}, \mathbf{A}}{\arg\max} \;
	L(\mathbf{R},\mathbf{C};  \boldsymbol{\pi}, \boldsymbol{\rho}, \mathbf{A} )&=	\underset{\mathbf{R},\mathbf{C}, \mathbf{A}}{\arg\min} \;
	D_{F^*}(\mathbf{X};\mathbf{R} B_{\mathbf{A}} \mathbf{C}^{\top})
	-\log_{\boldsymbol{\pi}} \mathbf{R}^{\top}\mathbf{1}_{m} -
	\log_{\boldsymbol{\rho}}\mathbf{C}^{\top}\mathbf{1}_{n}\nonumber\\
	&\propto	\underset{\mathbf{R},\mathbf{C}, B_{\mathbf{A}}}{\arg\max}	\;
	Tr\left(
		(\nabla F^{*}_{\mathbf{R} B_{\mathbf{A}} \mathbf{C}^{\top}})^{\top}
		(
		\mathbf{X} - \mathbf{R}B_{\mathbf{A}} \mathbf{C}^{\top}
		)
		\right)	
		+
		\log_{\boldsymbol{\pi}} \mathbf{R}^{\top}\mathbf{1}_{m} 
		+
		\log_{\boldsymbol{\rho}}
		\mathbf{C}^{\top}\mathbf{1}_{n}
\end{align*}
```
## Cite
Please cite the following paper in your publication if you are using [`Hybridcoclust`]() in your research:

```bibtex
 @article{Hybridcoclust, 
    title={One Equivalence Between Exponential Family Latent Block Model and Bregman Non-negative Matrix Tri-factorization for Co-clustering.}, 
    DOI={Preprint}, 
    journal={preprint}, 
    author={Saeid Hoseinipour, Mina Aminghafari, Adel Mohammadpour}, 
    year={2023}
} 
```
## Visulazation 

<img alt="Screenshot: 'README.md'" src="https://github.com/Saeidhoseinipour/Hybridcoclust/blob/master/Doc/Image/WC_1_5_bold_31_32_11_22_33_v2.png?raw=true" width="100%">

## References

[1] [Yoo et al, Orthogonal nonnegative matrix tri-factorization for co-clustering: Multiplicative updates on Stiefel manifolds (2010), 
	Information Processing and Management.](https://www.sciencedirect.com/science/article/abs/pii/S0306457310000038)
	
[2] [Ding et al, Orthogonal nonnegative matrix tri-factorizations for clustering, Proceedings of the 12th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (2008).](https://dl.acm.org/doi/abs/10.1145/1150402.1150420)

[3] [Long et al, Co-clustering by block value decomposition, Proceedings of the Eleventh ACM SIGKDD International Conference on Knowledge Discovery in Data 	Mining (2005).](https://dl.acm.org/doi/abs/10.1145/1081870.1081949)

[4] [Li et al, Nonnegative Matrix Factorization on Orthogonal Subspace (2010), Pattern Recognition Letters.](sciencedirect.com/science/article/abs/pii/S0167865509003651)

[5] [Cichocki et al, Non-negative matrix factorization with $\alpha$-divergence (2008), Pattern Recognition Letters.](https://www.sciencedirect.com/science/article/abs/pii/S0167865508000767)

[6] [Saeid, Hoseinipour et al, Orthogonal Parametric Non-negative Matrix Tri-Factorization with $\alpha$-Divergence for Co-clustering (2023), Expert Systems With Application.](https://doi.org/10.1016/j.eswa.2023.120680)

[7] [Saeid, Hoseinipour et al, Sparse Expoential Family Latent Block Model for Co-clustering (2023), Computational Statistics and Data Analysis (preprint).]()

[8] [Saeid, Hoseinipour et al, Orthogonal Non-negative Matrix Tri-Factorization with $\alpha$-Divergence for Persian Co-clustering (2023), Iranian Journal of Science and Technology, Transactions of Electrical Engineering (preprint).]()



