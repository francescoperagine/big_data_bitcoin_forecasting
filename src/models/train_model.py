import sys
sys.dont_write_bytecode = True
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.decomposition import PCA
from sklearn.experimental import enable_halving_search_cv
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, TimeSeriesSplit, LearningCurveDisplay, ValidationCurveDisplay, HalvingGridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.utils.constants import *
from src.visualization.visualize import *

class BTCForecasting:
    """
    A class to train and evaluate a model for Bitcoin price forecasting.
    """

    def __init__(
            self,
            data,
            ground_truth,
            test_size=0.2,
            random_state=42,
            n_splits: int = 5,
            pca_variance_threshold: float = 0.95,
            smoteenn: bool = False,
            feature_selection: str =  None,
            n_jobs: int = -1

        ) -> None:
        self.test_size = test_size
        self.random_state = random_state
        self.ground_truth = ground_truth
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.X_test_scaled = None
        self.X_test_pca = None
        self.n_splits = n_splits

        # Split the data for cross-validation respecting the temporal order
        self.tscv = TimeSeriesSplit(n_splits=self.n_splits)
        self.model = None
        self.search = None
        self.results = None
        self.pipeline = None
        self.classifier = None
        self.scaler = StandardScaler()
        self.pca = PCA(pca_variance_threshold)

        # SMOTEENN resampling
        self.smoteenn = smoteenn

        self.feature_importances = None

        self.feature_selection = feature_selection
        # Recursive Feature Elimination (RFE) feature selection
        self.rfe = True if feature_selection == 'rfe' else False

        # SelectFromModel feature selection
        self.select_from_model = True if feature_selection == 'sfm' else False

        self.n_jobs = n_jobs

        # Label encoder
        self.le = LabelEncoder()

        self._set_data(data)
        
    def _set_data(self, data) -> None:
        """
        Merges the data with the ground truth labels and splits the data into training and testing sets
        """
        merged_data =  pd.merge(data, self.ground_truth, on='origin_time', how='inner')
        self.X = merged_data.drop(columns=['origin_time', 'label'])
        self.y = self.le.fit_transform(merged_data['label'])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state, shuffle=False)

    def _set_pipeline(self, classifier) -> ImbPipeline:
        """
        Sets the pipeline for the model, including the SMOTEENN resampling, feature selection, scaling, PCA and classifier steps
        """
        steps = []
        if self.smoteenn:
            steps.append(('smoteenn', SMOTEENN()))

        if self.rfe:
            rfe_step = RFE(estimator=classifier, n_features_to_select=None, step=3)
            steps.append(('rfe', rfe_step))
        elif self.select_from_model:
            sfm_step = SelectFromModel(estimator=classifier, threshold='1.25*mean', prefit=False, importance_getter = 'auto')
            steps.append(('sfm', sfm_step))

        steps.extend([
            ('scaler', self.scaler),
            ('pca', self.pca),
            ('classifier', classifier)
        ])
        return ImbPipeline(steps=steps)
    
    def train(self, classifier, params, factor=3, aggressive_elimination = False, verbose=3) -> dict:
        """
        Trains the model using the HalvingGridSearchCV algorithm and stores the best model in the model attribute        
        """
        
        self.pipeline = self._set_pipeline(classifier)

        self.search = HalvingGridSearchCV(
            estimator=self.pipeline,
            param_grid=params,
            aggressive_elimination=aggressive_elimination,
            factor=factor,
            cv=self.tscv,
            scoring="f1_weighted",
            n_jobs=self.n_jobs,
            verbose=verbose,
            return_train_score=True,
            refit=True,
            error_score='raise'
        )

        self.search.fit(self.X_train, self.y_train)
        self.model = self.search.best_estimator_

        self.results = {
            'best_params': self.search.best_params_,
            'best_score': self.search.best_score_,
            'cv_results': self.search.cv_results_
        }
       
    def evaluate(self):
            self._evaluate_feature_importances()    
            """
            Evaluates the model on the test set and stores the results in model.results
            """

            self.y_pred = self.model.predict(self.X_test)

            accuracy = balanced_accuracy_score(self.y_test, self.y_pred)
            precision_macro, recall_macro, fscore_macro, _ = precision_recall_fscore_support(self.y_test, self.y_pred, average='macro')
            precision_weighted, recall_weighted, fscore_weighted, _ = precision_recall_fscore_support(self.y_test, self.y_pred, average='weighted')

            report = classification_report(self.y_test, self.y_pred, target_names=self.le.classes_)
            conf_matrix = confusion_matrix(self.y_test, self.y_pred)
            
            self.results.update({
                'accuracy_balanced': accuracy,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'fscore_macro': fscore_macro,
                'precision_weighted': precision_weighted,
                'recall_weighted': recall_weighted,
                'fscore_weighted': fscore_weighted,
                'classification_report': report,
                'conf_matrix': conf_matrix,
            })

    def _evaluate_feature_importances(self):
        """
        Extracts the feature importances from the classifier step, directly considers the PCA components as features,
        stores them in model.results, and returns them as a DataFrame.
        """
        classifier_step = self.model.named_steps['classifier']
        
        # Check if the classifier has the feature_importances_ attribute
        if not hasattr(classifier_step, 'feature_importances_'):
            print("Classifier does not have feature_importances_ attribute. Skipping feature importance evaluation.")
            return None
        
        feature_importances = classifier_step.feature_importances_
        
        # If PCA is used, the features are the PCA components
        if 'pca' in self.model.named_steps:
            pca_step = self.model.named_steps['pca']
            n_components = pca_step.n_components_
            component_names = [f'PC{i+1}' for i in range(n_components)]
            feature_names = component_names
        else:
            feature_names = self.X.columns
        
        # Store and sort the feature importances
        self.results.update({
            'feature_importance': {
                'Feature': feature_names,
                'Importance': feature_importances,
            }
        })
        feature_importance_df = pd.DataFrame(self.results['feature_importance'])
        self.feature_importances = feature_importance_df.sort_values(by='Importance', ascending=False)
        
        return self.feature_importances




    def plot_learn_cm_feat(self,filename):
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
        

        # Extract train and test scores from HalvingGridSearchCV results
        train_scores_mean = np.array(self.results['cv_results']['mean_train_score'])
        train_scores_std = np.array(self.results['cv_results']['std_train_score'])
        test_scores_mean = np.array(self.results['cv_results']['mean_test_score'])
        test_scores_std = np.array(self.results['cv_results']['std_test_score'])

        # Reduce the number of points for plotting
        num_points = 30
        indices = np.linspace(0, len(train_scores_mean) - 1, num_points).astype(int)
        
        train_scores_mean = train_scores_mean[indices]
        train_scores_std = train_scores_std[indices]
        test_scores_mean = test_scores_mean[indices]
        test_scores_std = test_scores_std[indices]
        train_sizes = np.linspace(0.1, 1.0, num_points)

        axes[0].plot(train_sizes, train_scores_mean, color='orange', marker='o', markersize=1, label='Train')
        axes[0].fill_between(train_sizes, train_scores_mean + train_scores_std, train_scores_mean - train_scores_std, alpha=0.05, color='orange')
        axes[0].plot(train_sizes, test_scores_mean, color='blue', marker='+', markersize=1, label='Test')
        axes[0].fill_between(train_sizes, test_scores_mean + test_scores_std, test_scores_mean - test_scores_std, alpha=0.05, color='blue')

        axes[0].set_title('Learning Curves')
        axes[0].set_xlabel('Training examples')
        axes[0].set_ylabel('F1-score (weighted)')
        axes[0].legend()
        axes[0].grid()

        disp = ConfusionMatrixDisplay(confusion_matrix=self.results['conf_matrix'], display_labels=list(self.le.classes_))
        disp.plot(ax=axes[1])
        axes[1].set_title('Confusion Matrix')

        # Plot Feature Importance
        if self.feature_selection is not None:

            axes[2].barh(self.feature_importances['Feature'], self.feature_importances['Importance'], color='skyblue')
            axes[2].set_xlabel('Feature Importance')
            axes[2].set_ylabel('Features')
            axes[2].set_title('Feature Importance')
            axes[2].invert_yaxis()
            axes[2].grid()

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_PATH, filename))
        plt.show()

    def plot_bias_variance_tradeoff(self, filename='bias_variance_tradeoff.png'):
        # Convert cv_results to a DataFrame
        results_df = pd.DataFrame(self.results['cv_results'])

        # Extract the parameters used in cross-validation
        params = [key for key in results_df.columns if key.startswith('param_')]
        
        # Determine the number of rows needed for the grid
        n_params = len(params)
        n_cols = 2
        n_rows = (n_params + n_cols - 1) // n_cols  # Calculate number of rows needed
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = axes.flatten()  # Flatten the axes array for easy iteration

        for i, param in enumerate(params):
            # Group by the specific parameter and compute mean and std for train and test scores
            grouped_results = results_df.groupby(param).agg(
                mean_train_score=('mean_train_score', 'mean'),
                std_train_score=('std_train_score', 'mean'),
                mean_test_score=('mean_test_score', 'mean'),
                std_test_score=('std_test_score', 'mean')
            ).reset_index()

            param_values = grouped_results[param]
            mean_train_scores = grouped_results['mean_train_score']
            mean_test_scores = grouped_results['mean_test_score']
            std_train_scores = grouped_results['std_train_score']
            std_test_scores = grouped_results['std_test_score']

            # Plotting the train and validation scores with error bars
            ax = axes[i]
            ax.errorbar(param_values, mean_train_scores, yerr=std_train_scores, label='Mean Train Score', marker='o', capsize=5)
            ax.errorbar(param_values, mean_test_scores, yerr=std_test_scores, label='Mean Test Score', marker='o', capsize=5)
            ax.set_xlabel(param)
            ax.set_ylabel('F1 Score (Weighted)')
            ax.set_title(f'Bias-Variance Tradeoff: {param}')
            ax.legend()
            ax.grid(True)

        # Hide any remaining empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_PATH, filename))
        plt.show()

