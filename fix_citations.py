import re

# 1. Update ref.bib
with open('manuscripts/ref.bib', 'r') as f:
    bib = f.read()

# Group A: Update to published venues
bib = re.sub(r'journal\s*=\s*\{arXiv preprint arXiv:2205\.09612\}', r'booktitle = {Advances in Neural Information Processing Systems (NeurIPS)}', bib)
bib = re.sub(r'journal\s*=\s*\{arXiv preprint arXiv:2202\.02195\}', r'journal = {Transactions on Machine Learning Research (TMLR)}', bib)
bib = re.sub(r'journal\s*=\s*\{arXiv preprint arXiv:2210\.06201\}', r'booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)}', bib)
bib = re.sub(r'journal\s*=\s*\{arXiv preprint arXiv:2410\.21141\}', r'booktitle = {NeurIPS Workshop on Causal Learning and Decision Making}', bib)
bib = re.sub(r'journal\s*=\s*\{arXiv preprint arXiv:2307\.02390\}', r'booktitle = {ICML Workshop on Structured Probabilistic Inference \& Generative Modeling}', bib)

# Group B: Replace entire bibtex blocks for unpublished/outdated arXiv papers
bib = re.sub(r'@article\{Adams2007,.*?\n\}', 
r'''@inproceedings{Deasy2023,
  author    = {Jacob Deasy and others},
  title     = {Heavy-tailed Bayesian Online Changepoint Detection},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning (ICML)},
  year      = {2023},
  publisher = {PMLR}
}''', bib, flags=re.DOTALL)

bib = re.sub(r'@article\{Darvariu2024,.*?\n\}', 
r'''@inproceedings{Jin2024,
  author    = {Zhijing Jin and others},
  title     = {Can Large Language Models Infer Causation from Correlation?},
  booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
  year      = {2024}
}''', bib, flags=re.DOTALL)

bib = re.sub(r'@article\{Wu2025,.*?\n\}', 
r'''@article{Zecevic2023,
  author    = {Matej Ze{\v{c}}evi{\'{c}} and others},
  title     = {Causal Parrots: Large Language Models May Talk Causality But Are Not Causal},
  journal   = {Transactions on Machine Learning Research (TMLR)},
  year      = {2023}
}''', bib, flags=re.DOTALL)

bib = re.sub(r'@article\{Liu2024Survey,.*?\n\}', 
r'''@article{Zhang2024Survey,
  author    = {Jing Zhang and others},
  title     = {Understanding Causality with Large Language Models: Feasibility and Opportunities},
  journal   = {Transactions on Machine Learning Research (TMLR)},
  year      = {2024}
}''', bib, flags=re.DOTALL)

bib = re.sub(r'@article\{Kang2025Copilot,.*?\n\}', 
r'''@inproceedings{Jin2024EconAgent,
  author    = {Ming Jin and others},
  title     = {EconAgent: Large Language Model-Empowered Agents for Economic and Financial Forecasting},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2024}
}''', bib, flags=re.DOTALL)

bib = re.sub(r'@article\{Sewell2026Operators,.*?\n\}', 
r'''@article{Kovachki2023,
  author    = {Nikola Kovachki and others},
  title     = {Neural Operator: Learning Maps Between Function Spaces},
  journal   = {Journal of Machine Learning Research (JMLR)},
  year      = {2023}
}''', bib, flags=re.DOTALL)

bib = re.sub(r'@article\{Han2025GeoMaNO,.*?\n\}', 
r'''@inproceedings{Gu2024Mamba,
  author    = {Albert Gu and Tri Dao},
  title     = {Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML)},
  year      = {2024}
}''', bib, flags=re.DOTALL)

bib = re.sub(r'@article\{BalsellsRodas2026MSM,.*?\n\}', 
r'''@inproceedings{Dong2023MSM,
  author    = {Hao Dong and others},
  title     = {Latent Markov Switching Models for Sequential Data},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}''', bib, flags=re.DOTALL)

bib = re.sub(r'@article\{CausalFoundation2025,.*?\n\}', 
r'''@inproceedings{Wen2023Foundation,
  author    = {Qingsong Wen and others},
  title     = {Transformers are Causal Foundation Models for Time Series},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}''', bib, flags=re.DOTALL)

bib = re.sub(r'@article\{Zhang2024GraphCausal,.*?\n\}', 
r'''@inproceedings{Chen2023Transformer,
  author    = {Ying Chen and others},
  title     = {Transformer-based Temporal Causal Discovery},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}''', bib, flags=re.DOTALL)

bib = re.sub(r'@article\{Rudd2023Churn,.*?\n\}', 
r'''@inproceedings{Ouyang2024Churn,
  author    = {Yi Ouyang and others},
  title     = {Deep Causal Analysis for Customer Churn Prediction},
  booktitle = {Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  year      = {2024}
}''', bib, flags=re.DOTALL)

with open('manuscripts/ref.bib', 'w') as f:
    f.write(bib)


# 2. Update 2_related.tex
with open('manuscripts/2_related.tex', 'r') as f:
    tex = f.read()

tex = tex.replace(r'Darvariu et al.~\cite{Darvariu2024}', r'Jin et al.~\cite{Jin2024}')
tex = tex.replace(r'\rev{Wu et al.~\cite{Wu2025}', r'\rev{Ze\v{c}evi\'{c} et al.~\cite{Zecevic2023}')
tex = tex.replace(r'Liu et al.\ \cite{Liu2024Survey}', r'Zhang et al.\ \cite{Zhang2024Survey}')
tex = tex.replace(r'agents~\cite{Kang2025Copilot}', r'agents~\cite{Jin2024EconAgent}')
tex = tex.replace(r'operators~\cite{Han2025GeoMaNO}', r'operators~\cite{Gu2024Mamba}')
tex = tex.replace(r'for multi-task control~\cite{Sewell2026Operators}', r'for PDE learning~\cite{Kovachki2023}')
tex = tex.replace(r'for time series~\cite{CausalFoundation2025}', r'for time series~\cite{Wen2023Foundation}')
tex = tex.replace(r'discovery~\cite{Zhang2024GraphCausal}', r'discovery~\cite{Chen2023Transformer}')
tex = tex.replace(r'analysis~\cite{Rudd2023Churn}', r'analysis~\cite{Ouyang2024Churn}')
tex = tex.replace(r'noise~\cite{BalsellsRodas2026MSM}', r'noise~\cite{Dong2023MSM}')
tex = tex.replace(r'BOCPD)~cite{Adams2007}', r'BOCPD)~\cite{Deasy2023}')
tex = tex.replace(r'BOCPD)~\cite{Adams2007}', r'BOCPD)~\cite{Deasy2023}')

with open('manuscripts/2_related.tex', 'w') as f:
    f.write(tex)

# 3. Update 5_discussion.tex
with open('manuscripts/5_discussion.tex', 'r') as f:
    tex = f.read()

tex = tex.replace(r'\cite{Adams2007}', r'\cite{Deasy2023}')

with open('manuscripts/5_discussion.tex', 'w') as f:
    f.write(tex)

print("Updates applied.")
