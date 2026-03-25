<p align="center">
  <img src="specific_humidity.gif" alt="q700-gif" width="500"><br>
  <em>Q700 6h forecast initialized 2021-12-01</em>
</p>


Accurate subseasonal weather forecasting remains a major challenge due to the inherently chaotic
nature of the atmosphere, which limits the predictive skill of conventional models beyond the mid-
range horizon (approximately 15 days). In this work, we present Marchuk, a generative latent
flow-matching model for global weather forecasting spanning mid-range to subseasonal timescales,
with prediction horizons of up to 30 days. Marchuk is a flow-matching latent diffusion model that
conditions on current-day weather maps and autoregressively predicts subsequent days’ weather
maps within the learned latent space. We replace rotary positional encodings (RoPE) with train-
able positional embeddings and extend the temporal context window, which together enhance the
model’s ability to represent and propagate long-range temporal dependencies during latent fore-
casting. Marchuk offers two key advantages: high computational efficiency and strong predictive
performance. Despite its compact architecture of only 276 million parameters, the model achieves
performance comparable to LaDCast, a substantially larger model with 1.6 billion parameters, while
operating at significantly higher inference speeds.


## Installation

```
conda create -n marchuk
conda activate marchuk
git clone https://github.com/tonyzyl/ladcast.git
pip install -e ladcast
```
Download Marchuk weights from [here](https://drive.google.com/file/d/15K_gRGpADh7Pcp1J1hE8tu5b0HBlYh5Y/view?usp=sharing).

## Evaluation script

See `inference.ipynb` to run Marchuk model.

## BibTeX
```
@misc{marchuk2025,
  title         = {Marchuk: Efficient Global Weather Forecasting from Mid-Range to Sub-Seasonal Scales via Flow Matching},
  author        = {},
  year          = {2026},
  eprint        = {},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/}
}
```