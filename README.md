LSNPC: **L**atent Space **S**hift Model for **N**oisy **P**rediction **C**orrection. 

This is the repository for the paper: [Correcting Noisy Multilabel Predictions: Modeling Label Noise through Latent Space Shifts](https://arxiv.org/abs/2502.14281)

### Deprecation
In the code, **MLNLC** is equivalent to LSNPC. 


# Abstract
Noise in data appears to be inevitable in most real-world machine learning applications and would cause severe overfitting problems. Not only can data features contain noise, but labels are also prone to be noisy due to human input. In this paper, rather than noisy label learning in multiclass classifications, we instead focus on the less explored area of noisy label learning for multilabel classifications. Specifically, we investigate the post-correction of predictions generated from classifiers learned with noisy labels. The reasons are two-fold. Firstly, this approach can directly work with the trained models to save computational resources. Secondly, it could be applied on top of other noisy label correction techniques to achieve further improvements. To handle this problem, we appeal to deep generative approaches that are possible for uncertainty estimation. Our model posits that label noise arises from a stochastic shift in the latent variable, providing a more robust and beneficial means for noisy learning. We develop both unsupervised and semi-supervised learning methods for our model. The extensive empirical study presents solid evidence to that our approach is able to consistently improve the independent models and performs better than a number of existing methods across various noisy label settings. Moreover, a comprehensive empirical analysis of the proposed method is carried out to validate its robustness, including sensitivity analysis and an ablation study, among other elements.

# Folder structure
Python 3.10.0+ with other requirements specified in the ```requirements.txt```
```python
📁 data_process        # files for setting up datasets used in the experiments
📁 dnn                 # folder containing baselines
📁 nlc                 # folder containing our proposed approach
📁 shells              # folder containing shell scripts for running experiments
📁 training            # folder containing training scripts for all methods
📄 argument.py    
📄 download_vit.py
📄 metrics.py          # evaluation metrics
📄 requirements.txt    # required libraries
📄 utils.py
```

# Dataset
For [Tomato dataset](https://github.com/mamta-joshi-gehlot/Tomato-Village/tree/main/Variant-b(MultiLabel%20Classification)), the downloaded zip file needs to be unziped into `[root]/tomato` folder with following file and subfolders. The `[root]` is the root folder we set with our argument.
```python
📁 train
📁 test
📁 val
📄 anno.csv (the csv file from Variant-b(MultiLabel Classification)
```

# Citation
```latex
@misc{huang2025correctingnoisymultilabelpredictions,
      title={Correcting Noisy Multilabel Predictions: Modeling Label Noise through Latent Space Shifts}, 
      author={Weipeng Huang and Qin Li and Yang Xiao and Cheng Qiao and Tie Cai and Junwei Liang and Neil J. Hurley and Guangyuan Piao},
      year={2025},
      eprint={2502.14281},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.14281}, 
}
```

