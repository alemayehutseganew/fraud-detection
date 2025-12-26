# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- Fix: apply compatibility patch when loading scikit-learn models that may be missing attributes (e.g., set `LogisticRegression.multi_class = 'ovr'` when absent). This prevents AttributeError during prediction when models are pickled with a newer scikit-learn version. Added test: `test_logistic_regression_missing_multi_class` in `tests/test_api.py`.

### Notes
- Recommended: pin scikit-learn version used for training and serving, or embed sklearn version metadata with saved models to detect mismatches early.
