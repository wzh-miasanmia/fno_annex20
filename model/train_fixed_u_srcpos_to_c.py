"""Compatibility entry point for the paper's plain FNO and shared utilities.

The canonical implementation is the frozen strict-ablation trainer used by the
paper's unified and resolution-transfer campaigns. Keeping the historical import
name here lets the baseline and GINO modules reuse that same implementation.
For the paper's mask/X/Y input, pass ``--no_velocity_input`` when training.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.train_fixed_u_srcpos_to_c_strict_ablation import (  # noqa: E402,F401
    CasesFNOFieldDataset,
    FNO2D,
    Normalizer,
    SpectralConv2d,
    TargetTransform,
    build_source_mask_gaussian,
    build_source_mask_rect,
    build_source_mask_soft_rect_sigmoid,
    df_to_grid,
    evaluate_metrics,
    gradient_matching_loss,
    is_power_of_two,
    laplacian_smoothness_loss,
    load_case_csv,
    main,
    make_normalizers,
    safe_float_tag,
    sample_cases_uniform_source,
    save_checkpoint,
    save_history_csv,
    save_loss_curve,
    set_seed,
    split_indices,
    split_indices_from_csv,
    train,
)


if __name__ == "__main__":
    main()
