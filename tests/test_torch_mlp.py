import numpy as np
import torch

from src.modeling.classifiers import EmbeddingClassifier


def test_custom_mlp_init_and_forward():
    # Dummy dimensions
    input_dim = 8
    output_dim = 2

    # Create small dummy datasets
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((10, input_dim)).astype(np.float32)
    y_train = rng.integers(0, output_dim, size=10).astype(np.int64)
    X_val = rng.standard_normal((4, input_dim)).astype(np.float32)
    y_val = rng.integers(0, output_dim, size=4).astype(np.int64)

    # Instantiate classifier and build the torch model without training
    clf = EmbeddingClassifier(random_state=0)
    model = clf._train_torch_mlp(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        input_dim=input_dim,
        output_dim=output_dim,
        epochs=0,  # skip training for a fast init/forward sanity check
        batch_size=4,
        lr=1e-3,
    )

    # Forward pass on dummy input
    model.eval()
    device = next(model.parameters()).device
    dummy = torch.zeros((5, input_dim), dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(dummy).detach().cpu()

    assert isinstance(out, torch.Tensor)
    assert out.shape == (5, output_dim)
