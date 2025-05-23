import pandas as pd
import numpy as np
import xgboost as xgb
import argparse
import logging
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
from scipy import stats

def main():
    parser = argparse.ArgumentParser(description='Train XGBoost models for krill presence and abundance.')
    parser.add_argument('--search', type=bool, default=False, help='Enable hyperparameter search', choices=[True, False])
    args = parser.parse_args()
    krill_model = KrillXGBoost(search=args.search)
    krill_model.load_data()
    krill_model.train_models()

class KrillXGBoost:
    def __init__(self, search):
        self.data = None
        self.presence_model = None
        self.abundance_model = None
        self.search = search
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Ensure output directory exists
        os.makedirs('output', exist_ok=True)

    def load_data(self):
        self.data = pd.read_csv("input/fusedData.csv")
        
        # Remove unnamed column if it exists
        unnamed_cols = [col for col in self.data.columns if 'Unnamed' in col]
        if unnamed_cols:
            self.logger.info(f"Removing unnamed columns: {unnamed_cols}")
            self.data = self.data.drop(columns=unnamed_cols)
        
        self.logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
        self.logger.info(f"Columns: {list(self.data.columns)}")
        
        # Check for missing values and handle them
        if self.data.isnull().sum().sum() > 0:
            self.logger.info("Filling missing values with column means")
            self.data.fillna(self.data.mean(), inplace=True)

    def train_models(self):
        # Separate features and targets
        X = self.data.drop(columns=['KRILL_PRESENCE', 'KRILL_LOG10', 'KRILL_SQRT', 'KRILL_LOGN', 'KRILL_QUAN', 'KRILL_ORIGINAL', 'LONGITUDE', 'LATITUDE'])
        y_presence = self.data['KRILL_PRESENCE']
        y_abundance = self.data['KRILL_ORIGINAL']
        
        # Feature scaling for better model performance
        self.logger.info("Applying feature scaling")
        X = (X - X.mean()) / X.std()
        
        # Split data for presence model (using all data)
        X_train_presence, X_test_presence, y_presence_train, y_presence_test = train_test_split(
            X, y_presence, test_size=0.2, random_state=42
        )
        
        # Filter data for abundance model (only where krill is present)
        presence_mask = self.data['KRILL_PRESENCE'] == 1
        X_abundance = X[presence_mask]
        y_abundance = y_abundance[presence_mask]
        
        # Focus on abundance values within -2 to +2.5 range
        self.logger.info("Filtering abundance data to focus on values within -2 to +2.5 range")
        
        # Calculate the distribution of abundance values
        total_samples = len(y_abundance)
        in_range_mask = (y_abundance >= np.percentile(y_abundance,20)) & (y_abundance <= np.percentile(y_abundance,80))
        in_range_count = in_range_mask.sum()
        out_range_count = total_samples - in_range_count
        
        self.logger.info(f"Total abundance samples: {total_samples}")
        self.logger.info(f"Samples within range [{np.percentile(y_abundance,20):.3f}, {np.percentile(y_abundance,80):.3f}]: {in_range_count} ({in_range_count/total_samples*100:.2f}%)")
        self.logger.info(f"Samples outside range: {out_range_count} ({out_range_count/total_samples*100:.2f}%)")
        
        # Original range for reference
        self.logger.info(f"Original abundance range: [{y_abundance.min():.3f}, {y_abundance.max():.3f}]")
        
        # Filter features and target to only include values within the specified range
        X_abundance_filtered = X_abundance[in_range_mask]
        y_abundance_filtered = y_abundance[in_range_mask]
        
        self.logger.info(f"Filtered abundance range: [{y_abundance_filtered.min():.3f}, {y_abundance_filtered.max():.3f}]")
        self.logger.info(f"Filtered abundance dataset shape: {X_abundance_filtered.shape}")
        
        # Split data for abundance model using the filtered data
        X_train_abundance, X_test_abundance, y_abundance_train, y_abundance_test = train_test_split(
            X_abundance_filtered, y_abundance_filtered, test_size=0.2, random_state=42
        )
        
        self.logger.info(f"Presence training data shape: {X_train_presence.shape}, Test data shape: {X_test_presence.shape}")
        self.logger.info(f"Abundance training data shape: {X_train_abundance.shape}, Test data shape: {X_test_abundance.shape}")
        
        # Train presence model (classification)
        self.train_presence_model(X_train_presence, X_test_presence, y_presence_train, y_presence_test)
        
        # Train abundance model (regression)
        self.train_abundance_model(X_train_abundance, X_test_abundance, y_abundance_train, y_abundance_test)
        
        # Make predictions on test set
        self.evaluate_models(X_test_presence, y_presence_test, X_test_abundance, y_abundance_test)
    

    def train_presence_model(self, X_train, X_test, y_train, y_test):
        self.logger.info("Training presence classification model...")
        
        # Create DMatrix for XGBoost
        dtrain_presence = xgb.DMatrix(X_train, label=y_train)
        dtest_presence = xgb.DMatrix(X_test, label=y_test)
        
        # Set parameters for classification
        params_classification = {
            'objective': 'binary:logistic',
            'learning_rate': 0.1,
            'max_depth': 5,
            'alpha': 10,
            'eval_metric': 'auc'
        }
       
        
        # Train model with early stopping
        self.presence_model = xgb.train(
            params=params_classification, 
            dtrain=dtrain_presence,
            num_boost_round=100,
            evals=[(dtest_presence, 'eval')],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        # Evaluate model
        preds = self.presence_model.predict(dtest_presence)
        preds_binary = (preds > 0.5).astype(int)
        accuracy = accuracy_score(y_test, preds_binary)
        auc = roc_auc_score(y_test, preds)
        
        self.logger.info(f"Presence Model Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
        
        # Save model
        os.makedirs("output/models", exist_ok=True)
        self.presence_model.save_model("output/models/presence_model.json")
        self.logger.info("Presence model saved to output/models/presence_model.json")

    def train_abundance_model(self, X_train, X_test, y_train, y_test):
        self.logger.info("Training abundance regression model...")
        self.logger.info(f"Using {len(y_train)} samples for training abundance model (krill present only)")
        
        # Create DMatrix for XGBoost
        dtrain_abundance = xgb.DMatrix(X_train, label=y_train)
        dtest_abundance = xgb.DMatrix(X_test, label=y_test)
        
        # Set parameters for regression
        params_regression = {
            'objective': 'reg:squarederror',
            'booster': 'gbtree',
            'gamma': 0.5,
            'eta': 0.3,
            'learning_rate': 0.1,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'max_depth': 2,
            'alpha': 10,
            'eval_metric': 'rmse'
        }
        if self.search:
            reg = xgb.XGBRegressor(nthread=-1)
            # run randomized search
            params = {
                'n_estimators':[50, 100, 200, 500],
                'min_child_weight':[4,5], 
                'gamma':[i/10.0 for i in range(3,6)],  
                'subsample':[i/10.0 for i in range(6,11)],
                'colsample_bytree':[i/10.0 for i in range(6,11)], 
                'max_depth': [2,3,4,6,7],
                'objective': ['reg:squarederror'],
                'booster': ['gbtree'],
                'eval_metric': ['rmse'],
                'eta': [i/10.0 for i in range(3,6)],
            }
            n_iter_search = 20
            random_search = RandomizedSearchCV(reg, param_distributions=params,
                                               n_iter=n_iter_search, cv=5, scoring='neg_mean_squared_error')
            random_search.fit(X_train, y_train)
            best_regressor = random_search.best_estimator_
            params_regression = best_regressor.get_params()
        
        # Train model with early stopping
        self.abundance_model = xgb.train(
            params=params_regression, 
            dtrain=dtrain_abundance,
            num_boost_round=100,
            evals=[(dtest_abundance, 'eval')],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        # Evaluate model
        preds = self.abundance_model.predict(dtest_abundance)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        
        self.logger.info(f"Abundance Model MSE: {mse:.4f}, RMSE: {rmse:.4f}")
        
        # Save model
        os.makedirs("output/models", exist_ok=True)
        self.abundance_model.save_model("output/models/abundance_model.json")
        self.logger.info("Abundance model saved to output/models/abundance_model.json")

    def evaluate_models(self, X_test_presence, y_presence_test, X_test_abundance, y_abundance_test):
        self.logger.info("Evaluating combined model performance...")
        
        # Create DMatrix for prediction
        dtest_presence = xgb.DMatrix(X_test_presence)
        dtest_abundance = xgb.DMatrix(X_test_abundance)
        
        # Make predictions
        presence_preds = self.presence_model.predict(dtest_presence)
        abundance_preds = self.abundance_model.predict(dtest_abundance)
        
        # Note: We can't directly calculate conditional abundance on the test set
        # because the test sets for presence and abundance are different
        # This is just for demonstration of the concept
        self.logger.info("Note: For a new prediction, we would first predict presence, then")
        self.logger.info("      only predict abundance if presence is predicted as 1")
        
        # Save feature importance
        presence_importance = self.presence_model.get_score(importance_type='gain')
        abundance_importance = self.abundance_model.get_score(importance_type='gain')
        
        # Log top features
        self.logger.info("Top 5 features for presence model:")
        for feature, importance in sorted(presence_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
            self.logger.info(f"  {feature}: {importance:.4f}")
            
        self.logger.info("Top 5 features for abundance model:")
        for feature, importance in sorted(abundance_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
            self.logger.info(f"  {feature}: {importance:.4f}")
        
        self.logger.info("Model evaluation complete")

if __name__ == '__main__':
    main()