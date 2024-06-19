from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.decomposition import PCA
from sklearn.experimental import enable_halving_search_cv
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, TimeSeriesSplit, LearningCurveDisplay, ValidationCurveDisplay, HalvingGridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.utils.constants import *
from src.visualization.visualize import *

class BTCForecasting:

    def __init__(
            self,
            data,
            ground_truth,
            test_size=0.2,
            random_state=42,
            n_splits: int = 5,
            pca_variance_threshold: float = 0.95,
            smoteenn: bool = False,
            rfe: bool = False
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
        self.X_test_scaled = None
        self.X_test_pca = None
        self.n_splits = n_splits
        self.tscv = None
        self.model = None
        self.search = None
        self.results = None
        self.pipeline = None
        self.classifier = None
        self.scaler = StandardScaler()
        self.pca = PCA(pca_variance_threshold)

        # SMOTEENN resampling
        self.smoteenn = smoteenn

        # Recursive Feature Elimination (RFE) feature selection
        self.rfe = rfe
        self.n_features_to_select = None

        # Label encoder
        self.le = LabelEncoder()

        self._set_data(data)
        
    def _set_data(self, data) -> None:
        merged_data =  pd.merge(data, self.ground_truth, on='origin_time', how='inner')
        self.X = merged_data.drop(columns=['origin_time', 'label'])
        self.y = self.le.fit_transform(merged_data['label'])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state, shuffle=False)
    
    def train_rfc_grid(self, classifier, params, features_to_select = None, factor=3, aggressive_elimination = False, verbose = 1) -> dict:
        

        self.classifier = classifier

        # Split the data for cross-validation respecting the temporal order
        self.tscv = TimeSeriesSplit(n_splits=self.n_splits)

        steps = []
        if self.smoteenn:
            steps.append(('smoteenn', SMOTEENN()))
        if self.rfe:
            self.n_features_to_select = features_to_select if features_to_select is not None else len(self.X.columns) // 2
            rfe_step = RFE(estimator=self.classifier, n_features_to_select=self.n_features_to_select, step=1)
            steps.append(('rfe', rfe_step))
        steps.extend([('scaler', self.scaler), ('pca', self.pca), ('classifier', self.classifier)])

        self.pipeline = ImbPipeline(steps=steps)

        self.search = HalvingGridSearchCV(
            estimator=self.pipeline,
            param_grid=params,
            aggressive_elimination=aggressive_elimination,
            factor=factor,
            cv=self.tscv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=verbose,
            return_train_score=True,
            refit=True
        )

        self.search.fit(self.X_train, self.y_train)
        self.model = self.search.best_estimator_

        self.results = {
            'best_params': self.search.best_params_,
            'best_score': self.search.best_score_,
            'cv_results': self.search.cv_results_
        }

        if self.rfe:
            rfe_step = self.model.named_steps['rfe']
            selected_features_mask = rfe_step.support_
            feature_ranking = rfe_step.ranking_
            selected_feature_names = self.X.columns[selected_features_mask]

            feature_importances = self.model.named_steps['classifier'].feature_importances_
            feature_importance_df = pd.DataFrame({'Feature': selected_feature_names, 'Importance': feature_importances})
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

            self.results.update({
                'rfe': {
                    'selected_features': selected_feature_names,
                    'feature_ranking': feature_ranking,
                    'feature_importance': feature_importance_df,
                    'selected_features_mask': selected_features_mask
                }
            })

            print("Selected features mask:", selected_features_mask)
            print("Selected features:", selected_feature_names)

            # Refit the model on the selected features
            # X_train_selected = self.X_train.iloc[:, selected_features_mask]
            # self.model.named_steps['classifier'].fit(X_train_selected, self.y_train)


    def train_model(self, classifier, classifier_params=None, features_to_select=None, verbose=1) -> dict:
        self.classifier = classifier

        self.tscv = TimeSeriesSplit(n_splits=self.n_splits)

        steps = []
        if self.smoteenn:
            steps.append(('smoteenn', SMOTEENN()))
        if self.rfe:
            self.n_features_to_select = features_to_select if features_to_select is not None else len(self.X.columns)
            rfe_step = RFE(estimator=self.classifier, n_features_to_select=self.n_features_to_select, step=1)
            steps.append(('rfe', rfe_step))
        steps.extend([('scaler', self.scaler), ('pca', self.pca), ('classifier', self.classifier)])

        self.pipeline = ImbPipeline(steps=steps)

        # Set classifier parameters if provided
        if classifier_params:
            self.pipeline.set_params(**classifier_params)

        # Use cross_val_score to perform cross-validation
        scores = cross_val_score(self.pipeline, self.X_train, self.y_train, cv=self.tscv, scoring='accuracy', n_jobs=-1)
        
        self.pipeline.fit(self.X_train, self.y_train)
        self.model = self.pipeline

        self.results = {
            'cv_scores': scores,
            'mean_cv_score': scores.mean(),
            'std_cv_score': scores.std()
        }

        if self.rfe:
            rfe_step = self.model.named_steps['rfe']
            selected_features_mask = rfe_step.support_
            feature_ranking = rfe_step.ranking_
            selected_feature_names = self.X.columns[selected_features_mask]

            feature_importances = self.model.named_steps['classifier'].feature_importances_
            feature_importance_df = pd.DataFrame({'Feature': selected_feature_names, 'Importance': feature_importances})
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

            self.results.update({
                'rfe': {
                    'selected_features': selected_feature_names,
                    'feature_ranking': feature_ranking,
                    'feature_importance': feature_importance_df,
                    'selected_features_mask': selected_features_mask
                }
            })

            print("Selected features mask:", selected_features_mask)
            print("Selected features:", selected_feature_names)

    def evaluate_model(self):
            # self.X_test_scaled = self.scaler.transform(self.X_test) 
            # self.X_test_pca = self.pca.transform(self.X_test_scaled)

            # if self.rfe:
            #     selected_features_mask = self.results['rfe']['selected_features_mask']
            #     X_test_selected = self.X_test.iloc[:, selected_features_mask]
            #     y_pred = self.model.named_steps['classifier'].predict(X_test_selected)
            # else:
            y_pred = self.model.predict(self.X_test)

            accuracy = accuracy_score(self.y_test, y_pred)
            precision, recall, fscore, _ = precision_recall_fscore_support(btcf.y_test, y_pred, average='macro')
            report = classification_report(self.y_test, y_pred, target_names=self.le.classes_)
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            
            self.results.update({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'fscore': fscore,
                'report': report,
                'conf_matrix': conf_matrix,
            })

    def plot_learning_curves(self, filename='learning_curve.png'):
        plt.clf()
        X_train_selected = self.X_train.iloc[:, self.results['rfe']['selected_features_mask']] if self.rfe else self.X_train
        LearningCurveDisplay.from_estimator(
            self.model.named_steps['classifier'],
            X_train_selected,
            self.y_train,
            cv=self.tscv,
            scoring='accuracy',
            n_jobs=-1
        )
        plt.title('Learning Curves')
        plt.xlabel('Training examples')
        plt.ylabel('Score')
        plt.legend(loc="best")
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_PATH, filename))
        plt.show()

    def plot_validation_curves(self, param_name, param_range, filename='validation_curve.png'):
        plt.clf()
        X_train_selected = self.X_train.iloc[:, self.results['rfe']['selected_features_mask']] if self.rfe else self.X_train
        ValidationCurveDisplay.from_estimator(
            self.model.named_steps['classifier'],
            X_train_selected,
            self.y_train,
            param_name=param_name,
            param_range=param_range,
            cv=self.tscv,
            scoring='accuracy',
            n_jobs=-1
        )
        plt.title(f'Validation Curve with {param_name}')
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.legend(loc="best")
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_PATH, filename))
        plt.show()

    def plot_feature_importance(self, filename):
        feature_importance_df = self.results.get('rfe', {}).get('feature_importance')
        if feature_importance_df is not None:
            plt.clf()
            plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
            plt.xlabel('Feature Importance')
            plt.ylabel('Features')
            plt.title('Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURE_PATH, filename))
            plt.show()
        else:
            print("Feature importance data is not available.")