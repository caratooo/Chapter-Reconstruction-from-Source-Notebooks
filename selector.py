"""
Stage 2: Semantic Selection
Uses sentence-transformers for embedding-based relevance scoring (preferred).
Selects notebooks and sections that form a coherent chapter theme.
"""

import numpy as np
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from ingest import NotebookContent, Cell
from sentence_transformers import SentenceTransformer


# Pre-defined chapter themes
CANDIDATE_THEMES = [
    {
        "name": "Supervised Learning Fundamentals: From Linear Models to Ensembles",
        "description": (
            "A comprehensive guide to supervised learning covering linear regression, "
            "logistic regression, gradient descent, support vector machines, decision trees, "
            "and ensemble methods like random forests and gradient boosting."
        ),
        "seed_keywords": [
            "linear regression", "logistic regression", "gradient descent",
            "regularization", "support vector machines", "decision trees",
            "random forests", "ensemble learning", "boosting", "classification",
            "overfitting", "cross-validation", "bias-variance tradeoff",
        ],
    },
    {
        "name": "Deep Learning with Neural Networks: Architecture to Training",
        "description": (
            "Building and training neural networks with Keras and TensorFlow, "
            "covering architectures, activation functions, optimizers, regularization, "
            "and practical training techniques."
        ),
        "seed_keywords": [
            "neural network", "deep learning", "keras", "tensorflow",
            "activation function", "backpropagation", "optimizer", "dropout",
            "batch normalization", "learning rate", "convolutional",
        ],
    },
    {
        "name": "The Complete ML Pipeline: From Data to Deployment",
        "description": (
            "End-to-end machine learning workflow including data preprocessing, "
            "feature engineering, model selection, evaluation, and deployment."
        ),
        "seed_keywords": [
            "pipeline", "preprocessing", "feature engineering", "cross-validation",
            "grid search", "model selection", "deployment", "data cleaning",
            "train test split", "evaluation metrics",
        ],
    },
    {
        "name": "Classification: Algorithms, Metrics, and Real-World Applications",
        "description": (
            "A thorough exploration of classification techniques including binary and "
            "multiclass classification, probabilistic models, decision boundaries, "
            "and evaluation strategies for imbalanced and real-world datasets."
        ),
        "seed_keywords": [
            "classification", "logistic regression", "naive bayes", "k-nearest neighbors",
            "decision boundary", "softmax", "precision", "recall", "f1 score",
            "confusion matrix", "ROC curve", "AUC", "class imbalance", "multiclass",
            "binary classification", "threshold", "probability calibration",
        ],
    },
]


@dataclass
class ScoredSection:
    notebook_id: str
    cell_index: int
    cell: Cell
    score: float


@dataclass
class SelectedMaterial:
    theme_name: str
    theme_description: str
    selected_notebooks: list[str]
    sections: list[ScoredSection]
    notebook_contents: dict  # notebook_id -> NotebookContent


class SentenceTransformerSelector:
    """Uses sentence-transformers for high-quality semantic embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"  Loading sentence-transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def score_notebooks(self, notebooks: list[NotebookContent], theme: dict) -> list[tuple[NotebookContent, float]]:
        theme_text = theme["description"] + " " + " ".join(theme["seed_keywords"])
        theme_emb = self._embed([theme_text])[0]
        nb_texts = [nb.title + ". " + nb.markdown_text for nb in notebooks]
        nb_embs = self._embed(nb_texts)
        scored = [(nb, self._cosine_sim(theme_emb, emb)) for nb, emb in zip(notebooks, nb_embs)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def score_sections(self, cells: list[Cell], theme: dict) -> list[float]:
        theme_text = theme["description"] + " " + " ".join(theme["seed_keywords"])
        theme_emb = self._embed([theme_text])[0]
        texts = [c.source for c in cells]
        embs = self._embed(texts)
        return [self._cosine_sim(theme_emb, emb) for emb in embs]


def run_selection(
    notebooks: list[NotebookContent],
    theme_index: Optional[int] = None,
    top_n_notebooks: int = 3,
    section_threshold: float = 0.05,
) -> SelectedMaterial:
    """Main entry point for the selection stage."""
    selector = SentenceTransformerSelector()

    # Select theme
    if theme_index is not None and 0 <= theme_index < len(CANDIDATE_THEMES):
        theme = CANDIDATE_THEMES[theme_index]
        print(f"\n  Using specified theme: '{theme['name']}'")
    else:
        print("\n  Available themes:")
        for i, t in enumerate(CANDIDATE_THEMES):
            print(f"    [{i}] {t['name']}")
            print(f"        {t['description'][:100]}...")
        while True:
            try:
                choice = input(f"\n  Select a theme [0-{len(CANDIDATE_THEMES) - 1}]: ").strip()
                choice_idx = int(choice)
                if 0 <= choice_idx < len(CANDIDATE_THEMES):
                    break
                print(f"  Please enter a number between 0 and {len(CANDIDATE_THEMES) - 1}.")
            except (ValueError, EOFError):
                print(f"  Please enter a valid number.")
        theme = CANDIDATE_THEMES[choice_idx]
        print(f"\n  Selected theme: '{theme['name']}'")

    # Score and select notebooks
    scored_nbs = selector.score_notebooks(notebooks, theme)
    selected_nbs = scored_nbs[:top_n_notebooks]

    print(f"\n  Selected notebooks:")
    for nb, score in selected_nbs:
        print(f"    {nb.notebook_id} (score: {score:.4f}) - '{nb.title}'")

    # Score sections within selected notebooks
    all_sections = []
    for nb, _ in selected_nbs:
        filterable_cells = [(i, c) for i, c in enumerate(nb.cells) if len(c.source.strip()) > 30]
        if not filterable_cells:
            continue
        indices, cells_list = zip(*filterable_cells)
        scores = selector.score_sections(list(cells_list), theme)

        for idx, cell, score in zip(indices, cells_list, scores):
            if score >= section_threshold:
                all_sections.append(ScoredSection(
                    notebook_id=nb.notebook_id,
                    cell_index=idx,
                    cell=cell,
                    score=score,
                ))

    # Sort by notebook order for coherence
    all_sections.sort(key=lambda s: (s.notebook_id, s.cell_index))

    nb_map = {nb.notebook_id: nb for nb, _ in selected_nbs}

    print(f"\n  Selected {len(all_sections)} sections (threshold={section_threshold})")

    return SelectedMaterial(
        theme_name=theme["name"],
        theme_description=theme["description"],
        selected_notebooks=[nb.notebook_id for nb, _ in selected_nbs],
        sections=all_sections,
        notebook_contents=nb_map,
    )
