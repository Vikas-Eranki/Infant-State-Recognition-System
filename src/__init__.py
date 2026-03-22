# src/__init__.py
# Public API for the Infant State Recognition System src package.
# Import from here for clean cross-notebook usage:
#   from src import train_svm, evaluate_model, generate_logmel_spectrogram

from src.utils import (
    load_dataset_metadata,
    get_class_weights,
    save_results,
    plot_class_distribution,
    set_random_seed,
    create_label_encoder,
    check_audio_file,
)

from src.features import (
    load_and_preprocess,
    generate_logmel_spectrogram,
    build_dataset,
)

from src.model_ml import (
    train_svm,
    evaluate_model,
    plot_confusion_matrix,
    get_failure_cases,
)

__all__ = [
    # utils
    "load_dataset_metadata",
    "get_class_weights",
    "save_results",
    "plot_class_distribution",
    "set_random_seed",
    "create_label_encoder",
    "check_audio_file",
    # features
    "load_and_preprocess",
    "generate_logmel_spectrogram",
    "build_dataset",
    # model_ml
    "train_svm",
    "evaluate_model",
    "plot_confusion_matrix",
    "get_failure_cases",
]