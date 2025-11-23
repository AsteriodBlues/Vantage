"""
Model Interpretation and Validation Pipeline

This script performs comprehensive model interpretation including:
- Feature importance analysis
- SHAP values and visualizations
- Partial dependence plots
- Cross-validation
- Error analysis
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.feature_importance import analyze_feature_importance
from src.shap_analysis import analyze_shap_values
from src.partial_dependence import analyze_partial_dependence
from src.cross_validation import analyze_cross_validation
from src.error_analysis import analyze_errors


def load_data():
    """Load prepared train/val/test splits."""
    print("Loading data splits...")

    data_dir = Path('data/processed')

    # Load the split files
    train_df = pd.read_csv(data_dir / 'train.csv')
    val_df = pd.read_csv(data_dir / 'val.csv')
    test_df = pd.read_csv(data_dir / 'test.csv')

    # Separate features and target
    # Try different possible target column names
    possible_targets = ['position', 'Position', 'finish_position']
    target_col = None
    for col in possible_targets:
        if col in train_df.columns:
            target_col = col
            break

    if target_col is None:
        raise ValueError(f"Could not find target column. Available: {train_df.columns.tolist()}")

    feature_cols = [col for col in train_df.columns if col != target_col]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # Keep only numeric columns
    numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    X_train = X_train[numeric_cols]
    X_val = X_val[numeric_cols]
    X_test = X_test[numeric_cols]

    print(f"Using {len(numeric_cols)} numeric features")

    print(f"Train: {len(X_train)} samples")
    print(f"Val: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    print(f"Features: {len(X_train.columns)}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_best_model():
    """Load the best performing model from previous analysis."""
    print("\nLoading best model...")

    # Try to load ensemble models (best from previous work)
    ensemble_path = 'results/models/ensemble_models.pkl'
    advanced_path = 'results/models/advanced_models.pkl'

    if Path(ensemble_path).exists():
        with open(ensemble_path, 'rb') as f:
            models_dict = pickle.load(f)

        # Prefer voting_weighted as it performed best
        if 'voting_weighted' in models_dict:
            model = models_dict['voting_weighted']
            print(f"Loaded voting_weighted ensemble from: {ensemble_path}")
            return model, 'voting', 'voting_weighted'
        elif 'voting_equal' in models_dict:
            model = models_dict['voting_equal']
            print(f"Loaded voting_equal ensemble from: {ensemble_path}")
            return model, 'voting', 'voting_equal'
        else:
            # Use first available model
            model_name = list(models_dict.keys())[0]
            model = models_dict[model_name]
            print(f"Loaded {model_name} from: {ensemble_path}")
            model_type = 'stacking' if 'stacking' in model_name else 'voting'
            return model, model_type, model_name

    elif Path(advanced_path).exists():
        with open(advanced_path, 'rb') as f:
            models_dict = pickle.load(f)

        # Prefer tuned models
        for name in ['catboost_tuned', 'xgboost_tuned', 'lightgbm_tuned']:
            if name in models_dict:
                model = models_dict[name]
                print(f"Loaded {name} from: {advanced_path}")
                model_type = name.split('_')[0]
                return model, model_type, name

        # Use first available
        model_name = list(models_dict.keys())[0]
        model = models_dict[model_name]
        print(f"Loaded {model_name} from: {advanced_path}")
        model_type = model_name.split('_')[0]
        return model, model_type, model_name

    raise FileNotFoundError("No trained models found. Please run training first.")


def main():
    """Run complete model interpretation pipeline."""
    print("="*80)
    print("MODEL INTERPRETATION AND VALIDATION PIPELINE")
    print("="*80)

    # Create output directory
    output_dir = Path('results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # Load best model
    model, model_type, model_name = load_best_model()
    print(f"Model type: {model_type}")

    # Get predictions for validation set
    print("\nGenerating validation predictions...")
    y_val_pred = model.predict(X_val)
    val_mae = np.mean(np.abs(y_val - y_val_pred))
    val_rmse = np.sqrt(np.mean((y_val - y_val_pred) ** 2))
    val_r2 = 1 - (np.sum((y_val - y_val_pred) ** 2) / np.sum((y_val - y_val.mean()) ** 2))

    print(f"Validation Performance:")
    print(f"  MAE: {val_mae:.3f}")
    print(f"  RMSE: {val_rmse:.3f}")
    print(f"  R²: {val_r2:.3f}")

    # Results storage
    results = {
        'model_name': model_name,
        'model_type': model_type,
        'val_mae': val_mae,
        'val_rmse': val_rmse,
        'val_r2': val_r2
    }

    # =========================================================================
    # 1. Feature Importance Analysis
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 1: FEATURE IMPORTANCE ANALYSIS")
    print("="*80)

    try:
        importance_results = analyze_feature_importance(
            model,
            X_val,
            y_val,
            model_type=model_type,
            output_dir=output_dir
        )
        results['importance'] = importance_results

        # Save importance to CSV
        builtin_importance = importance_results['builtin_importance']
        builtin_importance.to_csv(output_dir / 'feature_importance.csv', index=False)
        print(f"\nSaved feature importance to {output_dir / 'feature_importance.csv'}")

    except Exception as e:
        print(f"Error in feature importance analysis: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # 2. SHAP Analysis
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: SHAP VALUE ANALYSIS")
    print("="*80)

    try:
        # Determine appropriate SHAP analysis approach
        if model_type in ['xgboost', 'lightgbm', 'catboost']:
            shap_model_type = model_type
        elif model_type == 'voting':
            # Use first estimator for SHAP
            print("Using first estimator from voting ensemble for SHAP analysis")
            shap_model = model.estimators_[0]
            shap_model_type = 'xgboost'
        elif model_type == 'stacking':
            shap_model = model
            shap_model_type = 'stacking'
        else:
            shap_model = model
            shap_model_type = 'xgboost'

        if model_type in ['voting']:
            model_for_shap = shap_model
        else:
            model_for_shap = model

        shap_results = analyze_shap_values(
            model_for_shap,
            X_val,
            y_val,
            model_type=shap_model_type,
            X_train=X_train,
            output_dir=output_dir,
            sample_size=1000
        )
        results['shap'] = shap_results

    except Exception as e:
        print(f"Error in SHAP analysis: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # 3. Partial Dependence Analysis
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 3: PARTIAL DEPENDENCE ANALYSIS")
    print("="*80)

    try:
        pdp_results = analyze_partial_dependence(
            model,
            X_val,
            importance_results['builtin_importance'],
            output_dir=output_dir,
            top_n_features=6,
            ice_n_samples=100
        )
        results['pdp'] = pdp_results

    except Exception as e:
        print(f"Error in partial dependence analysis: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # 4. Cross-Validation Analysis
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 4: CROSS-VALIDATION ANALYSIS")
    print("="*80)

    try:
        # Check for circuit column
        circuit_col = None
        for col in ['circuit_name', 'circuit', 'track']:
            if col in X_train.columns:
                circuit_col = col
                break

        cv_results = analyze_cross_validation(
            model,
            X_train,
            y_train,
            output_dir=output_dir,
            n_folds=5,
            circuit_column=circuit_col
        )
        results['cv'] = cv_results

        # Save CV results
        cv_summary = pd.DataFrame({
            'metric': ['MAE', 'RMSE', 'R²'],
            'kfold_mean': [
                cv_results['kfold']['stats']['mae_mean'],
                cv_results['kfold']['stats']['rmse_mean'],
                cv_results['kfold']['stats']['r2_mean']
            ],
            'kfold_std': [
                cv_results['kfold']['stats']['mae_std'],
                cv_results['kfold']['stats']['rmse_std'],
                cv_results['kfold']['stats']['r2_std']
            ],
            'timeseries_mean': [
                cv_results['timeseries']['stats']['mae_mean'],
                cv_results['timeseries']['stats']['rmse_mean'],
                cv_results['timeseries']['stats']['r2_mean']
            ],
            'timeseries_std': [
                cv_results['timeseries']['stats']['mae_std'],
                cv_results['timeseries']['stats']['rmse_std'],
                cv_results['timeseries']['stats']['r2_std']
            ]
        })
        cv_summary.to_csv(output_dir / 'cv_summary.csv', index=False)
        print(f"\nSaved CV summary to {output_dir / 'cv_summary.csv'}")

    except Exception as e:
        print(f"Error in cross-validation analysis: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # 5. Error Analysis
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 5: ERROR ANALYSIS")
    print("="*80)

    try:
        categorical_features = []
        for col in ['circuit_name', 'team', 'driver_name']:
            if col in X_val.columns:
                categorical_features.append(col)

        error_results = analyze_errors(
            model,
            X_val,
            y_val,
            output_dir=output_dir,
            categorical_features=categorical_features
        )
        results['errors'] = error_results

        # Save worst predictions
        worst_predictions = error_results['worst_predictions']
        worst_predictions.to_csv(output_dir / 'worst_predictions.csv', index=False)
        print(f"\nSaved worst predictions to {output_dir / 'worst_predictions.csv'}")

    except Exception as e:
        print(f"Error in error analysis: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "="*80)
    print("INTERPRETATION PIPELINE COMPLETE")
    print("="*80)

    print(f"\nModel: {model_name}")
    print(f"Validation MAE: {val_mae:.3f} positions")

    if 'importance' in results:
        print(f"\nTop 5 Important Features:")
        for i, row in results['importance']['builtin_importance'].head(5).iterrows():
            print(f"  {i+1}. {row['feature']}: {row['gain']:.4f}")

    if 'cv' in results:
        print(f"\nCross-Validation:")
        print(f"  K-Fold MAE: {results['cv']['kfold']['stats']['mae_mean']:.3f} ± "
              f"{results['cv']['kfold']['stats']['mae_std']:.3f}")
        print(f"  Time Series MAE: {results['cv']['timeseries']['stats']['mae_mean']:.3f} ± "
              f"{results['cv']['timeseries']['stats']['rmse_std']:.3f}")

    if 'errors' in results:
        print(f"\nError Statistics:")
        print(f"  80% predictions within: {np.percentile(results['errors']['errors']['abs_errors'], 80):.2f} positions")
        print(f"  90% predictions within: {np.percentile(results['errors']['errors']['abs_errors'], 90):.2f} positions")

    print(f"\nAll visualizations saved to: {output_dir}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
