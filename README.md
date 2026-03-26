# Marchuk: Efficient Global Weather Forecasting from Mid-Range to Sub-Seasonal Scales via Flow Matching


<a href="https://arxiv.org/abs/2603.24428"><img src="https://img.shields.io/badge/arXiv-2603.24428-b31b1b.svg" height=22.5></a>
<a href="https://v-gen-ai.github.io/Marchuk/"><img src="https://img.shields.io/badge/Project-Website-blue" height=22.5><a>
<a href="https://colab.research.google.com/drive/1f1q14U4b9fJwQRYrFGvEpQt6JOE_1nmX?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=22.5></a>
<a href="https://huggingface.co/v-gen-ai/Marchuk"><img src="https://img.shields.io/badge/%E2%80%8B-Hugging%20Face-FFD21E?logo=huggingface&logoColor=FFD21E" height=22.5></a>
<a href="https://github.com/v-gen-ai/Marchuk/blob/main/LICENSE"><img src="https://img.shields.io/github/license/AIRI-Institute/al_toolbox" height=22.5></a>

<p align="center">
  <img src="specific_humidity.gif" alt="q700-gif" width="500"><br>
  <em>Q700 6h forecast initialized 2021-12-01</em>
</p>


Accurate subseasonal weather forecasting remains a major challenge due to the inherently chaotic
nature of the atmosphere, which limits the predictive skill of conventional models beyond the mid-
range horizon (approximately 15 days). In this work, we present Marchuk, a generative latent
flow-matching model for global weather forecasting spanning mid-range to subseasonal timescales,
with prediction horizons of up to 30 days. Marchuk conditions on current-day weather maps
and autoregressively predicts subsequent days’ weather maps within the learned latent space. We
replace rotary positional encodings (RoPE) with trainable positional embeddings and extend the
temporal context window, which together enhance the model’s ability to represent and propagate
long-range temporal dependencies during latent forecasting. Marchuk offers two key advantages:
high computational efficiency and strong predictive performance. Despite its compact architecture of
only 276 million parameters, the model achieves performance comparable to LaDCast, a substantially
larger model with 1.6 billion parameters, while operating at significantly higher inference speeds.

## Installation

```
conda create -n marchuk
conda activate marchuk
git clone https://github.com/tonyzyl/ladcast.git
pip install -e ladcast
```

## Evaluation script

Run `inference.ipynb` to evaluate Marchuk model on single timestamp.

## BibTeX
```
@article{marchuk2026,
  title={Marchuk: Efficient Global Weather Forecasting from Mid-Range to Sub-Seasonal Scales via Flow Matching},
  author={Arsen Kuzhamuratov, Mikhail Zhirnov, Andrey Kuznetsov, Ivan Oseledets, Konstantin Sobolev},
  journal={arXiv preprint arXiv:},
  year={2026}
}
```