"""Our top-1 (torchmetrics, as used in train.py) must equal sklearn's
accuracy on the same synthetic batches. NO hand-rolled metrics anywhere."""

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torchmetrics.classification import MulticlassAccuracy


def test_top1_matches_sklearn_random_preds():
    rs = np.random.RandomState(0)
    for trial in range(5):
        num_classes = int(rs.choice([10, 100]))
        n = int(rs.randint(50, 2000))
        logits = torch.from_numpy(rs.randn(n, num_classes)).float()
        labels = torch.from_numpy(rs.randint(0, num_classes, size=n))
        metric = MulticlassAccuracy(num_classes=num_classes, average="micro")
        # feed in uneven chunks, as the eval loop does
        for chunk_logits, chunk_labels in zip(logits.split(128), labels.split(128)):
            metric.update(chunk_logits, chunk_labels)
        ours = metric.compute().item()
        ref = accuracy_score(labels.numpy(), logits.argmax(dim=1).numpy())
        assert abs(ours - ref) < 1e-6, f"trial {trial}: {ours} != sklearn {ref}"


def test_top1_on_perfect_and_worst_preds():
    num_classes = 10
    labels = torch.arange(num_classes).repeat(7)
    perfect = torch.nn.functional.one_hot(labels, num_classes).float()
    metric = MulticlassAccuracy(num_classes=num_classes, average="micro")
    metric.update(perfect, labels)
    assert metric.compute().item() == 1.0
    metric.reset()
    metric.update(perfect.roll(1, dims=1), labels)  # every prediction off by one
    assert metric.compute().item() == 0.0
