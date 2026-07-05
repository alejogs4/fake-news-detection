# Detecting Fake News Through Its Propagation

**Postgraduate Course: Artificial Intelligence with Deep Learning**
[Presentation slides](https://docs.google.com/presentation/d/1Ky57UEuolyhbOoCLO9icmi5UPGnK39yVt63M6m6o5PU/edit?slide=id.g3ed224869cc_0_11#slide=id.g3ed224869cc_0_11)

A study found that fake news spreads about six times faster than true news ([Vosoughi, Roy and Aral, *Science*, 2018](https://www.science.org/doi/10.1126/science.aap9559)). Most detection systems try to catch fake news by reading the text. This project asks a different question. Can we detect fake news by looking at how it spreads through people, not only by what it says?

We reproduce two published graph neural network papers on the [FakeNewsNet](https://github.com/safe-graph/GNN-FakeNews) and UPFD benchmark. We also test two new ideas of our own to see if they improve the results.

---

## Table of contents

- [Motivation](#motivation)
- [The three hypotheses](#the-three-hypotheses)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Experiments](#experiments)
  - [Experiment 1: H0, content baseline](#experiment-1-h0-content-baseline)
  - [Experiment 2: H1, confirmation bias (UPFD)](#experiment-2-h1-confirmation-bias-upfd)
  - [Experiment 3: H2, echo chamber (HGFND)](#experiment-3-h2-echo-chamber-hgfnd)
  - [Experiment 4: New hyperedges on top of HGFND](#experiment-4-new-hyperedges-on-top-of-hgfnd)
- [Overall conclusions](#overall-conclusions)
- [Project milestones](#project-milestones)
- [References](#references)

---

## Motivation

Fake news is a real problem for society. It is not new, and it is not local to one country. It affects how people vote, how they see health information, and how they trust news in general.

A study at MIT looked at more than 126,000 news stories on Twitter. It found that false news spreads about six times faster than true news, and reaches more people ([Vosoughi, Roy and Aral, *Science*, 2018](https://www.science.org/doi/10.1126/science.aap9559)). By the time a true story catches up, a false story has often already reached everyone it will ever reach.

Most current detection systems try to solve this by reading the text of an article. This has two clear limits.

- **Scale.** Hundreds of hours of content go online every minute. No group of people can read and check all of it in time.
- **Style.** Well-written fake news can copy the style of real news closely. A model that only reads words can be fooled, because fake news does not always sound fake.

Our starting idea for this project: the way news spreads through people carries information that the text alone does not. Who shares a story, when they share it, and who they share it with, can tell us more than the words in the article. This project tests that idea directly, using graph neural networks to model news as it spreads through a network of people, instead of only reading what the news says.

---

## The three hypotheses

| | Question | Signal used | Reference |
|---|---|---|---|
| **H0** | Can the text alone predict fake news? | Article content only (BERT embedding). No social graph. | Our own baseline |
| **H1** | Does a user's own history predict what they share? | Propagation tree and per-user content and profile features. | [UPFD, Dou et al., SIGIR 2021](https://arxiv.org/abs/2104.12259) |
| **H2** | Does the group a user belongs to predict what spreads? | Hypergraph that connects news sharing a user, a time window, or a topic. | [HGFND, Jeong et al., IEEE BigData 2022](https://www.cs.emory.edu/~kshu5/files/HGFND_IEEE_Bigdata22_Final.pdf) |

Each hypothesis adds one more layer of social context. First the text. Then the person. Then the group.

---

## Repository structure

```
.
├── src/                  # Model code: GCN, GAT, SAGE backbones, and HGFND hypergraph attention
├── experiments/          # One script or notebook per experiment
├── results/              # Raw output metrics for each experiment run
├── docs/                 # Extra notes on architecture, cloud setup, and learning guide
├── examples/             # Small usage examples
├── config.yaml           # Central config: dataset, GNN backbone, feature switches
├── environment.yml       # Conda environment file
├── main.py               # Entry point. Runs one experiment from config.yaml
└── README.md
```

This repository is still being cleaned up after submission. If you find an old result file or notebook, use the numbers in this README and in the presentation slides. Those are the ones we confirm as correct.

---

## Getting started

### What you need

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Python 3.10 or newer
- A GPU helps but is not required for small runs. PolitiFact runs fine on CPU. GossipCop is much faster on GPU.

### Install locally

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate fake-news-detection

# Install the project in editable mode
pip install -e .
```

### Recommended: Google Colab

We ran our experiments on Google Colab with a T4 GPU. GossipCop is large, so a GPU runtime saves a lot of time. To run this project on Colab:

1. Upload or clone this repository into a Colab session.
2. Run `pip install -r requirements.txt` (or `pip install -e .`) in the first cell.
3. Run `main.py`, or open a notebook from the `experiments/` folder.

### Running an experiment

The project uses one config file. You do not need to change model code to reproduce our runs.

```bash
python main.py
```

Open `config.yaml` to:
- Choose the dataset: `politifact` or `gossipcop`
- Choose the GNN backbone: `GCN`, `GAT`, or `SAGE`
- Turn feature branches on or off: set `use_gnn: false` or `use_text: false` to test one signal alone
- Choose which hyperedge types are active, for the H2 and Experiment 4 runs: `user`, `time`, `topic`, `verified_user`, `sentiment`

---

## Experiments

All experiments use the UPFD benchmark. PolitiFact has 314 news graphs. GossipCop has 5,464 news graphs. All experiments use BERT (768-dim) content features and PyTorch as the framework. Unless we say otherwise, accuracy is one test-set score. Experiment 4 reports the mean and standard deviation over 7 random seeds.

### Experiment 1: H0, content baseline

**Hypothesis.** If fake news can be found from writing style or content alone, a text-only classifier should already work well. It should not need any social information.

**Setup.** BERT encodes the article text. This goes into a GNN classifier (GCNFN backbone) with no social graph. There are no edges from social behavior. This setup shows what content alone can achieve.

**Results.**

| Dataset | Accuracy |
|---|---|
| PolitiFact | 71.0% |
| GossipCop | 85.6% |

**Conclusions.** Text alone scores clearly above chance. But it does not reach a high accuracy, especially on PolitiFact, which is the smaller and harder dataset. This led us to test H1: does knowing who shares the news help?

---

### Experiment 2: H1, confirmation bias (UPFD)

**Hypothesis.** People tend to share news that matches what they already believe. This is called confirmation bias. If we add each user's own posting history to the model, next to the news content, the model should detect fake news better than with content alone.

**Setup.** We reproduce [UPFD](https://arxiv.org/abs/2104.12259) ([code](https://github.com/safe-graph/GNN-FakeNews)). Each news item is the root of a propagation tree. The users who retweet it are child nodes. Each user node carries features from their own past tweets. Each news node carries the BERT-encoded article. A GraphSAGE encoder reads the whole tree. A linear classifier gives the final prediction.

**Results.**

| Dataset | Accuracy |
|---|---|
| PolitiFact | 84.6% |
| GossipCop | 97.2% |

**Conclusions.** Adding each user's history improves both datasets over H0. PolitiFact goes up by 13.6 percentage points. GossipCop goes up by 11.6 percentage points. This confirms that knowing who shares a story adds real information. The next question: does the group a user belongs to add anything on top of that?

---

### Experiment 3: H2, echo chamber (HGFND)

**Hypothesis.** Fake news spreads through echo chambers. These are closed groups of users who keep sharing the same kind of content. If we connect news items that share a user, a time window, or a topic into one hypergraph, this group signal should improve on the person-level signal from H1.

**Setup.** We reproduce [HGFND](https://www.cs.emory.edu/~kshu5/files/HGFND_IEEE_Bigdata22_Final.pdf) ([code](https://github.com/ujeong1/IEEEBigdata22_HGFND)). The propagation tree encoder from H1 gives a 128-dim vector for each news item. Then we group news items into hyperedges. Groups form around the same sharer, the same time window, or the same topic. A two-layer hypergraph attention network reads this structure. First it reads from nodes to hyperedges. Then it reads from hyperedges back to nodes. This gives the final representation for classification.

**Results.**

| Dataset | Accuracy |
|---|---|
| PolitiFact | 92.3% |
| GossipCop | 97.5% |

**Conclusions.** H2 improves on H1 for both datasets. PolitiFact goes up by 7.7 percentage points. The gain is largest on PolitiFact, the smaller and harder dataset. This suggests that group-level structure helps most when there is less content signal to begin with. This is our strongest result. It supports our main idea: social structure, not text, is the stronger signal for this task.

---

### Experiment 4: New hyperedges on top of HGFND

**Hypothesis.** HGFND already groups news by user, time, and topic. We asked if new hyperedge types could add more signal and improve results further.

We tested two new hyperedge types, and their combination.

**Verified-user hyperedge.** This groups news by whether it was shared mostly by verified accounts, or mostly by unverified accounts. The idea: fake news often spreads more through unverified accounts.

**Sentiment hyperedge.** This groups news by the emotional tone of the headline. We score each headline as alarming, neutral, or positive. The idea: fake news often uses more emotional language.

**Verified-user and Sentiment combined.** Both hyperedge types are active at the same time. This tests if the two signals add up.

We also thought about a country-of-sharer hyperedge. This would group news by the country of the people who shared it. The goal was to find coordinated groups spreading the same fake news. We could not build this hyperedge. The dataset does not include location data for users. We mention it here for completeness. Not every idea can be tested with the data we have.

**Setup.** We use the same HGFND backbone as Experiment 3. This means frozen BERT features, a GraphSAGE propagation encoder, and a two-layer hypergraph attention network. We test only on PolitiFact. We did not run GossipCop for this experiment because of time and compute limits. Larger batch sizes were not possible with our available GPU. Each configuration runs 7 times with different random seeds. We report the mean and standard deviation.

**Results.**

| Configuration | PolitiFact accuracy |
|---|---|
| HGFND, published baseline | 91.11 ± 1.89 |
| Plus verified-user hyperedge | 91.01 ± 2.29 |
| Plus sentiment hyperedge | 91.14 ± 2.19 |
| Plus verified-user and sentiment combined | 90.56 ± 2.25 |

**Conclusions.** Neither new hyperedge gave a clear improvement over the published baseline. Both single additions land close to 91.11%, inside normal noise. The combined version is clearly worse. We see this as a useful and honest result, not a failure. We tested a clear idea, and the data shows it does not help for this dataset and this backbone. One possible reason: BERT's 768-dim content features may already contain much of the same information as verified-status and sentiment. This leaves little new signal for the hypergraph to use. This is a clear, checkable finding. It helps us know where to focus next.

---

## Overall conclusions

1. **Social structure beats text alone.** Accuracy goes up at each step, from H0 at 71.0%, to H1 at 84.6%, to H2 at 92.3%, all on PolitiFact. Each new layer of social context improves detection: first the person, then the group.
2. **Group-level signal (H2) gives us the strongest result.** The gain is largest on the smaller dataset. This suggests that social structure helps most when content signal is limited.
3. **Not every new idea works, and that is still useful.** Our two new hyperedge types did not improve on HGFND's original design. Combining them made results worse. We report this clearly instead of making it sound better than it is.
4. **Next steps we would try with more time:** test the new hyperedges on GossipCop if compute allows, study why verified-user and sentiment signal seems to repeat what BERT already knows, and try the country-of-sharer hyperedge if location data becomes available.

---

## Project milestones

| Week | Milestone | What we did |
|---|---|---|
| W1 | Choose the problem | We considered Bitcoin fraud, air quality, and fake news detection. We chose fake news. |
| W2 | Understand the state of the art | We studied fake news detection with graph neural networks. We selected UPFD and HGFND as our base papers. |
| W3 | Baseline | We reproduced UPFD and HGFND. We compared our numbers to the published results. |
| W4 | Experiments I | We compared GNN backbones (GCN, GAT, SAGE) for UPFD. We selected the best one. |
| W5 | Experiments II | We designed and ran new hyperedge types and tests for HGFND. |
| W6 | Final presentation | We closed the project, collected what we learned, and delivered the final version. |

---

## References

- Dou, Y., Shu, K., Xia, C., Yu, P. S., and Sun, L. (2021). **User Preference-aware Fake News Detection.** SIGIR 2021. [Paper](https://arxiv.org/abs/2104.12259) · [Code](https://github.com/safe-graph/GNN-FakeNews)
- Jeong, U., et al. (2022). **Nothing Stands Alone: Relational Fake News Detection with Hypergraph Neural Networks.** IEEE BigData 2022. [Paper](https://www.cs.emory.edu/~kshu5/files/HGFND_IEEE_Bigdata22_Final.pdf) · [Code](https://github.com/ujeong1/IEEEBigdata22_HGFND)
- Vosoughi, S., Roy, D., and Aral, S. (2018). **The spread of true and false news online.** *Science*, 359(6380). [Paper](https://www.science.org/doi/10.1126/science.aap9559)
- Shu, K., et al. **FakeNewsNet.** [Dataset and code](https://github.com/safe-graph/GNN-FakeNews)

---

**Authors:** Alejandro García, Simón Flores, Santiago Romo
**Course:** Postgraduate, Artificial Intelligence with Deep Learning
