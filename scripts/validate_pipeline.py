#!/usr/bin/env python3
"""
Small-Scale Validation of Data Loading and Model Training
验证整个流程在小规模数据上是否正常工作
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 70)
print("Small-Scale Validation: Data Loading and Model Training")
print("=" * 70)

# ============================================================================
# STEP 1: Verify Data Loading
# ============================================================================
print("\n[STEP 1] Verifying Data Loading")
print("-" * 70)

from data.custom_dataset import CustomLungDataset, merge_datasets
from configs import DATASET_PATHS
from torchvision import transforms

# Basic transform to convert PIL Image to Tensor
_to_tensor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

try:
    # Load datasets
    print("Loading Dataset 1 (IQ-OTHNCCD)...")
    ds1 = CustomLungDataset(DATASET_PATHS['dataset1'], 'dataset1', transform=_to_tensor)
    print(f"  OK: {len(ds1)} samples")
    print(f"  Distribution: {ds1.get_class_distribution()}")

    print("\nLoading Dataset 2 (LungColon)...")
    ds2 = CustomLungDataset(DATASET_PATHS['dataset2'], 'dataset2', transform=_to_tensor)
    print(f"  OK: {len(ds2)} samples")
    print(f"  Distribution: {ds2.get_class_distribution()}")

    # Dataset 3 is optional (may not be downloaded yet)
    ds3 = None
    try:
        print("\nLoading Dataset 3 (Lung4Types)...")
        ds3 = CustomLungDataset(DATASET_PATHS['dataset3'], 'dataset3', transform=_to_tensor)
        print(f"  OK: {len(ds3)} samples")
        print(f"  Distribution: {ds3.get_class_distribution()}")
    except Exception as e:
        print(f"  SKIP: Dataset 3 not available ({e})")

    # Get a sample
    print("\nTesting sample retrieval...")
    img, label = ds1[0]
    print(f"  Sample image shape: {img.shape}, label: {label}")
    print("\n[OK] Step 1 PASSED: Data loading works correctly")

except Exception as e:
    print(f"\n[FAILED] Step 1 FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 2: Verify Model Creation
# ============================================================================
print("\n" + "=" * 70)
print("[STEP 2] Verifying Model Creation")
print("-" * 70)

try:
    from model import HybridModel
    import torch

    print("Creating HybridModel (small config: model_dim=128, nhead=4, num_layers=2)...")
    model = HybridModel(num_classes=3, model_dim=128, nhead=4, num_layers=2)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  OK: Model created with {total_params:,} parameters")

    # Test forward pass
    print("\nTesting forward pass with dummy input...")
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"  OK: Forward pass output shape: {output.shape}")
    print(f"  OK: Output logits: {output[0].tolist()}")

    print("\n[OK] Step 2 PASSED: Model creation and forward pass work correctly")

except Exception as e:
    print(f"\n[FAILED] Step 2 FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 3: Verify Training Step (small subset)
# ============================================================================
print("\n" + "=" * 70)
print("[STEP 3] Verifying Training Step")
print("-" * 70)

try:
    from training.trainer import Trainer, TrainingConfig
    from torch.utils.data import DataLoader, Subset

    # Create tiny subset for validation
    print("Creating tiny subset (32 samples) for smoke test...")
    tiny_indices = range(min(32, len(ds1)))
    tiny_dataset = Subset(ds1, tiny_indices)
    train_loader = DataLoader(tiny_dataset, batch_size=8, shuffle=True)
    print(f"  OK: Created DataLoader with {len(train_loader)} batches")

    # Create model and trainer
    print("\nCreating model and trainer...")
    model = HybridModel(num_classes=3, model_dim=128, nhead=4, num_layers=2)
    config = TrainingConfig(
        num_epochs=2,
        batch_size=8,
        use_amp=torch.cuda.is_available(),  # Enable AMP when GPU available
        output_dir=str(PROJECT_ROOT / 'outputs' / 'test_validation'),
        learning_rate=1e-4,
        transformer_lr=5e-4,
    )
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=train_loader,
    )
    print(f"  OK: Trainer created on device: {trainer.device}")

    # Run 1 epoch as smoke test
    print("\nRunning 1 training epoch (smoke test)...")
    train_loss, train_acc = trainer.train_epoch()
    print(f"  Train loss: {train_loss:.4f}")
    print(f"  Train acc: {train_acc:.2f}%")

    print("\n[OK] Step 3 PASSED: Training pipeline works!")

except Exception as e:
    print(f"\n[FAILED] Step 3 FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)
ds3_info = f"{len(ds3)} samples, dist={ds3.get_class_distribution()}" if ds3 else "Not available"
print(f"""
Dataset Loading:
  - Dataset 1 (IQ-OTHNCCD): {len(ds1)} samples, dist={ds1.get_class_distribution()}
  - Dataset 2 (LungColon):  {len(ds2)} samples, dist={ds2.get_class_distribution()}
  - Dataset 3 (Lung4Types): {ds3_info}

Model:
  - HybridModel created successfully
  - Forward pass works correctly

Training:
  - 1 epoch completed: loss={train_loss:.4f}, acc={train_acc:.2f}%

ALL CHECKS PASSED - Pipeline is ready for full experiments!
""")
