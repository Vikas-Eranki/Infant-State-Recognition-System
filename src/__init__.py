# src/__init__.py
# Public API for the Infant State Recognition System src package.
# Import from here for clean cross-notebook usage:
#   from src import train_svm, evaluate_model, generate_logmel_spectrogram
#   from src import build_dscnn, MacroF1Score, generate_gradcam
#
# Phase 3 note:
#   MacroF1Score must be importable by name so that
#       tf.keras.models.load_model(path, custom_objects={'MacroF1Score': MacroF1Score})
#   can deserialise the Phase 2 checkpoint (best_dscnn.keras).

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

from src.model_dl import (
    MacroF1Score,
    build_dscnn,
    build_standard_cnn,
    build_dscnn_no_dropout,
    compile_model,
    default_callbacks,
    evaluate_model as evaluate_dscnn,
    get_last_conv_layer,
    generate_gradcam,
    plot_confusion_matrix as plot_confusion_matrix_dl,
    plot_training_curves,
    convert_to_tflite,
)

from src.model_hybrid import (
    extract_cnn_features,
    extract_cnn_probabilities,
    extract_svm_scores,
    build_hybrid_features,
    train_meta_classifier,
    train_mlp_meta,
    evaluate_hybrid,
    validate_svm_ovr_shape,
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
    # model_ml (Phase 1 — SVM)
    "train_svm",
    "evaluate_model",
    "plot_confusion_matrix",
    "get_failure_cases",
    # model_dl (Phase 2 — DS-CNN)
    "MacroF1Score",
    "build_dscnn",
    "build_standard_cnn",
    "build_dscnn_no_dropout",
    "compile_model",
    "default_callbacks",
    "evaluate_dscnn",
    "get_last_conv_layer",
    "generate_gradcam",
    "plot_confusion_matrix_dl",
    "plot_training_curves",
    "convert_to_tflite",
    # model_hybrid (Phase 3)
    "extract_cnn_features",
    "extract_cnn_probabilities",
    "extract_svm_scores",
    "build_hybrid_features",
    "train_meta_classifier",
    "train_mlp_meta",
    "evaluate_hybrid",
    "validate_svm_ovr_shape",
]
