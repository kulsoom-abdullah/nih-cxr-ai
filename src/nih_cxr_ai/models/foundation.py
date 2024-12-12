# src/nih_cxr_ai/models/foundation.py
"""Foundation model for chest X-ray classification.

This module implements the Google foundation model architecture for analyzing chest 
X-rays using modern deep learning techniques adapted from computer vision models.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

from ..utils.transforms import get_train_transforms, get_val_transforms
from .base import ChestXrayBaseModel


class CXRFoundationModel(ChestXrayBaseModel):
    """Model using CXR Foundation embeddings"""

    def __init__(
        self,
        num_classes: int = 14,
        learning_rate: float = 1e-4,
        embedding_dim: int = 768 * 32,  # From documentation
    ):
        super().__init__(num_classes=num_classes, learning_rate=learning_rate)

        # Load foundation model for embeddings
        self.foundation_model = AutoModel.from_pretrained(
            "google/cxr-foundation",
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )

        # Freeze foundation model weights
        for param in self.foundation_model.parameters():
            param.requires_grad = False

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # Get embeddings from foundation model
        with torch.no_grad():
            embeddings = self.foundation_model(x).last_hidden_state

        # Flatten embeddings
        embeddings = embeddings.view(embeddings.size(0), -1)

        # Pass through classifier
        return self.classifier(embeddings)
