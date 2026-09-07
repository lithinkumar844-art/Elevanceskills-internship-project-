"""
data_loader.py — Load arXiv papers into a pandas DataFrame.

Priority:
  1. Full Kaggle JSONL  (data/arxiv-metadata-oai-snapshot.json)
  2. Pre-built sample CSV  (data/cs_papers_sample.csv)
  3. Fetch from arXiv public API  (~500 papers, needs internet)
  4. Built-in hardcoded sample  (fallback, always works offline)
"""

from __future__ import annotations
import json
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)
CS_CATS = set(config.CS_CATEGORIES.keys())


def _cat_label(cats: str) -> str:
    for c in cats.split():
        if c in config.CS_CATEGORIES:
            return config.CS_CATEGORIES[c]
    return "Computer Science"


def _is_cs(cats: str) -> bool:
    return any(c in CS_CATS for c in cats.split())


# ── Source 1: Full Kaggle JSONL ───────────────────────────────────────────────

def _load_from_jsonl(path: Path) -> pd.DataFrame:
    records, n_read = [], 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                p = json.loads(line)
            except json.JSONDecodeError:
                continue
            cats = p.get("categories", "")
            if not _is_cs(cats):
                continue
            abstract = (p.get("abstract") or "").replace("\n", " ").strip()
            if len(abstract) < 100:
                continue
            year = None
            ud = p.get("update_date", "")
            if ud:
                try:
                    year = int(ud[:4])
                except ValueError:
                    pass
            records.append({
                "id":             p.get("id", ""),
                "title":          (p.get("title") or "").replace("\n", " ").strip(),
                "abstract":       abstract,
                "authors":        p.get("authors", ""),
                "categories":     cats,
                "category_label": _cat_label(cats),
                "year":           year,
                "doi":            p.get("doi", ""),
                "url":            f"https://arxiv.org/abs/{p.get('id','')}",
            })
            n_read += 1
            if n_read >= config.MAX_PAPERS_FROM_JSONL:
                break
    df = pd.DataFrame(records).drop_duplicates("id").reset_index(drop=True)
    logger.info("Loaded %d CS papers from JSONL.", len(df))
    return df


# ── Source 2: Cached sample CSV ───────────────────────────────────────────────

def _load_from_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"id", "title", "abstract", "authors", "categories", "category_label", "year", "url"}
    if not required.issubset(df.columns) or len(df) == 0:
        raise ValueError("CSV missing required columns or empty")
    logger.info("Loaded %d papers from sample CSV.", len(df))
    return df


# ── Source 3: arXiv public API ────────────────────────────────────────────────

def _fetch_from_api(total: int = 300) -> pd.DataFrame:
    NS = "http://www.w3.org/2005/Atom"
    records, start = [], 0
    batch = 50   # smaller batch = less timeout risk
    cats_query = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV"

    logger.info("Fetching CS papers from arXiv API...")
    while len(records) < total:
        url = (
            f"https://export.arxiv.org/api/query"
            f"?search_query={cats_query}"
            f"&start={start}&max_results={batch}"
            f"&sortBy=submittedDate&sortOrder=descending"
        )
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
        except Exception as e:
            logger.warning("API fetch failed at batch %d: %s", start, e)
            break

        try:
            root = ET.fromstring(r.text)
        except ET.ParseError:
            break

        entries = root.findall(f"{{{NS}}}entry")
        if not entries:
            break

        for entry in entries:
            arxiv_id = (entry.findtext(f"{{{NS}}}id") or "").split("/abs/")[-1].strip()
            title    = (entry.findtext(f"{{{NS}}}title") or "").replace("\n", " ").strip()
            abstract = (entry.findtext(f"{{{NS}}}summary") or "").replace("\n", " ").strip()
            authors  = ", ".join(
                (a.findtext(f"{{{NS}}}name") or "")
                for a in entry.findall(f"{{{NS}}}author")
            )
            cats = " ".join(
                t.attrib.get("term", "")
                for t in entry.findall(f"{{{NS}}}category")
            )
            published = entry.findtext(f"{{{NS}}}published") or ""
            year = int(published[:4]) if len(published) >= 4 else None

            if len(abstract) < 80 or not title:
                continue

            records.append({
                "id":             arxiv_id,
                "title":          title,
                "abstract":       abstract,
                "authors":        authors,
                "categories":     cats,
                "category_label": _cat_label(cats),
                "year":           year,
                "doi":            "",
                "url":            f"https://arxiv.org/abs/{arxiv_id}",
            })

        start += batch
        time.sleep(4)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).drop_duplicates("id").reset_index(drop=True)
    config.SAMPLE_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(config.SAMPLE_CSV, index=False)
    logger.info("Fetched and cached %d papers from arXiv API.", len(df))
    return df


# ── Source 4: Built-in offline sample ────────────────────────────────────────

def _builtin_sample() -> pd.DataFrame:
    """Hardcoded landmark CS papers — always works, no internet needed."""
    papers = [
        ("1706.03762", "Attention Is All You Need",
         "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.",
         "Vaswani, Ashish; Shazeer, Noam; Parmar, Niki", "cs.CL cs.LG", "Computation & Language (NLP)", 2017),
        ("1810.04805", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
         "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.",
         "Devlin, Jacob; Chang, Ming-Wei; Lee, Kenton", "cs.CL", "Computation & Language (NLP)", 2019),
        ("1512.03385", "Deep Residual Learning for Image Recognition",
         "We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions.",
         "He, Kaiming; Zhang, Xiangyu; Ren, Shaoqing", "cs.CV", "Computer Vision", 2016),
        ("1406.2661", "Generative Adversarial Networks",
         "We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G.",
         "Goodfellow, Ian; Pouget-Abadie, Jean; Mirza, Mehdi", "cs.LG", "Machine Learning", 2014),
        ("1301.3666", "Efficient Estimation of Word Representations in Vector Space",
         "We propose two novel model architectures for computing continuous vector representations of words from very large data sets. The quality of these representations is measured in a word similarity task, and the results are compared to the previously best performing techniques based on different types of neural networks.",
         "Mikolov, Tomas; Chen, Kai; Corrado, Greg", "cs.CL cs.LG", "Computation & Language (NLP)", 2013),
        ("1502.03167", "Batch Normalization: Accelerating Deep Network Training",
         "Training Deep Neural Networks is complicated by the fact that the distribution of each layer inputs changes during training, as the parameters of the previous layers change. We refer to this phenomenon as internal covariate shift, and address the problem by normalizing layer inputs.",
         "Ioffe, Sergey; Szegedy, Christian", "cs.LG", "Machine Learning", 2015),
        ("1409.0473", "Neural Machine Translation by Jointly Learning to Align and Translate",
         "We conjecture that the use of a fixed-length vector is a bottleneck in improving the performance of this basic encoder-decoder architecture, and propose to extend this by allowing a model to automatically search for parts of a source sentence that are relevant to predicting a target word.",
         "Bahdanau, Dzmitry; Cho, Kyunghyun; Bengio, Yoshua", "cs.CL cs.LG", "Computation & Language (NLP)", 2015),
        ("1312.6114", "Auto-Encoding Variational Bayes",
         "We introduce a stochastic variational inference and learning algorithm that scales to large datasets and, under some mild differentiability conditions, even works in the case of intractable posteriors. We demonstrate that a continuous latent variable model with neural network components can be trained efficiently.",
         "Kingma, Diederik P.; Welling, Max", "cs.LG stat.ML", "Machine Learning", 2014),
        ("2005.14165", "Language Models are Few-Shot Learners",
         "We train GPT-3, an autoregressive language model with 175 billion parameters, and test its performance in the few-shot setting. GPT-3 achieves strong performance on many NLP datasets, including translation, question-answering, and cloze tasks.",
         "Brown, Tom; Mann, Benjamin; Ryder, Nick", "cs.CL", "Computation & Language (NLP)", 2020),
        ("1409.1556", "Very Deep Convolutional Networks for Large-Scale Image Recognition",
         "We investigate the effect of the convolutional network depth on its accuracy in the large-scale image recognition setting. Our main contribution is a thorough evaluation of networks of increasing depth using an architecture with very small convolution filters.",
         "Simonyan, Karen; Zisserman, Andrew", "cs.CV", "Computer Vision", 2015),
        ("2010.11929", "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
         "We show that a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to mid-sized or small image recognition benchmarks, Vision Transformer attains excellent results.",
         "Dosovitskiy, Alexey; Beyer, Lucas; Kolesnikov, Alexander", "cs.CV cs.AI cs.LG", "Computer Vision", 2021),
        ("1611.01144", "Categorical Reparameterization with Gumbel-Softmax",
         "We introduce the Gumbel-Softmax distribution, which is a continuous distribution over the simplex that can approximate samples from a categorical distribution. The Gumbel-Softmax enables gradients to flow through discrete variables.",
         "Jang, Eric; Gu, Shixiang; Poole, Ben", "cs.LG", "Machine Learning", 2017),
        ("1703.06870", "Mask R-CNN",
         "We present a conceptually simple, flexible, and general framework for object instance segmentation. Our approach efficiently detects objects in an image while simultaneously generating a high-quality segmentation mask for each instance.",
         "He, Kaiming; Gkioxari, Georgia; Dollar, Piotr", "cs.CV", "Computer Vision", 2017),
        ("1907.11692", "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
         "We present a replication study of BERT pretraining that carefully measures the impact of many key hyperparameters and training data size. We find that BERT was significantly undertrained, and can match or exceed the performance of every model published after it.",
         "Liu, Yinhan; Ott, Myle; Goyal, Naman", "cs.CL", "Computation & Language (NLP)", 2019),
        ("2103.00020", "Learning Transferable Visual Models From Natural Language Supervision",
         "We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million image-text pairs collected from the internet. CLIP achieves strong zero-shot transfer.",
         "Radford, Alec; Kim, Jong Wook; Hallacy, Chris", "cs.CV cs.LG", "Computer Vision", 2021),
        ("1706.01427", "A simple neural network module for relational reasoning",
         "We describe how to use Relation Networks as a simple plug-and-play module to solve problems that fundamentally hinge on relational reasoning. We tested RNs on text-based and visual question answering, achieving state-of-the-art results.",
         "Santoro, Adam; Raposo, David; Barrett, David", "cs.NE cs.AI cs.LG", "Neural & Evolutionary Computing", 2017),
        ("1810.12152", "Graph Neural Networks: A Review of Methods and Applications",
         "We provide a detailed review of graph neural networks and its applications in various domains. We propose a general design pipeline for GNN models and discuss the challenges and future research directions.",
         "Zhou, Jie; Cui, Ganqu; Hu, Shengding", "cs.LG cs.AI", "Machine Learning", 2019),
        ("2106.09685", "LoRA: Low-Rank Adaptation of Large Language Models",
         "We propose Low-Rank Adaptation, or LoRA, which freezes the pretrained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks.",
         "Hu, Edward J.; Shen, Yelong; Wallis, Phillip", "cs.CL cs.AI cs.LG", "Computation & Language (NLP)", 2022),
        ("2302.13971", "LLaMA: Open and Efficient Foundation Language Models",
         "We introduce LLaMA, a collection of foundation language models ranging from 7B to 65B parameters. We train our models on trillions of tokens, and show that it is possible to train state-of-the-art models using publicly available datasets exclusively.",
         "Touvron, Hugo; Lavril, Thibaut; Izacard, Gautier", "cs.CL", "Computation & Language (NLP)", 2023),
        ("2304.02643", "Segment Anything",
         "We introduce the Segment Anything Model (SAM), a promptable model for image segmentation. SAM is trained on a dataset of 11 million images and 1.1 billion masks. The model supports zero-shot transfer to new image distributions and tasks.",
         "Kirillov, Alexander; Mintun, Eric; Ravi, Nikhila", "cs.CV", "Computer Vision", 2023),
        ("2305.10601", "Voyager: An Open-Ended Embodied Agent with Large Language Models",
         "We introduce VOYAGER, the first LLM-powered embodied lifelong learning agent in Minecraft that continuously explores the world, acquires diverse skills, and makes novel discoveries without human intervention.",
         "Wang, Guanzhi; Xie, Yuqi; Jiang, Yunfan", "cs.AI cs.RO", "Artificial Intelligence", 2023),
        ("1602.01783", "Asynchronous Methods for Deep Reinforcement Learning",
         "We propose a conceptually simple and lightweight framework for deep reinforcement learning that uses asynchronous gradient descent for optimization of deep neural network controllers. We present asynchronous variants of four standard reinforcement learning algorithms.",
         "Mnih, Volodymyr; Badia, Adria Puigdomenech; Mirza, Mehdi", "cs.LG cs.AI", "Machine Learning", 2016),
        ("2010.02502", "An Introduction to Federated Learning",
         "Federated learning is a new machine learning framework that enables multiple parties to collaboratively train a machine learning model without exchanging their raw data. We survey the concept, system design, privacy mechanisms, and applications.",
         "Zhang, Chen; Xie, Yu; Bai, Hang", "cs.LG cs.DC", "Machine Learning", 2021),
        ("1805.09300", "Reinforcement Learning: An Introduction",
         "Reinforcement learning is learning what to do to maximize a numerical reward signal. The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them.",
         "Sutton, Richard S.; Barto, Andrew G.", "cs.AI cs.LG", "Artificial Intelligence", 2018),
        ("2201.11903", "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
         "We explore how generating a chain of thought, a series of intermediate reasoning steps, significantly improves the ability of large language models to perform complex reasoning. We achieve state-of-the-art results on arithmetic, commonsense, and symbolic reasoning benchmarks.",
         "Wei, Jason; Wang, Xuezhi; Schuurmans, Dale", "cs.CL cs.AI", "Computation & Language (NLP)", 2022),
    ]

    records = []
    for pid, title, abstract, authors, cats, cat_label, year in papers:
        records.append({
            "id":             pid,
            "title":          title,
            "abstract":       abstract,
            "authors":        authors,
            "categories":     cats,
            "category_label": cat_label,
            "year":           year,
            "doi":            "",
            "url":            f"https://arxiv.org/abs/{pid}",
        })

    df = pd.DataFrame(records)
    config.SAMPLE_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(config.SAMPLE_CSV, index=False)
    logger.info("Using built-in sample of %d landmark CS papers.", len(df))
    return df


# ── Public entry point ────────────────────────────────────────────────────────

def load_papers(force_api: bool = False) -> pd.DataFrame:
    """Load CS papers — tries every source in order, never crashes."""

    # 1. Full Kaggle JSONL
    if not force_api and config.ARXIV_JSONL.exists():
        try:
            return _load_from_jsonl(config.ARXIV_JSONL)
        except Exception as e:
            logger.warning("JSONL load failed: %s", e)

    # 2. Cached CSV
    if not force_api and config.SAMPLE_CSV.exists():
        try:
            return _load_from_csv(config.SAMPLE_CSV)
        except Exception as e:
            logger.warning("CSV load failed: %s — will re-fetch.", e)
            config.SAMPLE_CSV.unlink(missing_ok=True)

    # 3. arXiv API
    try:
        df = _fetch_from_api(total=300)
        if len(df) > 10:
            return df
    except Exception as e:
        logger.warning("API fetch failed: %s", e)

    # 4. Built-in offline sample (always works)
    logger.warning("Using built-in offline sample (25 landmark papers).")
    return _builtin_sample()
