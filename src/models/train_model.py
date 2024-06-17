from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.feature_selection import RFE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.utils.constants import *
from src.visualization.visualize import *
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split, TimeSeriesSplit, LearningCurveDisplay, ValidationCurveDisplay
from sklearn.feature_selection import RFE, SelectFromModel
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import  HalvingGridSearchCV, HalvingRandomSearchCV

class BTCForecasting:

    def __init__(self, n_splits: int = 5, smoteenn: bool = False, rfe: bool = False) -> None:
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.n_splits = n_splits
        self.tscv = None
        self.model = None
        self.search = None
        self.results = None
        self.pipeline = None
        self.classifier = None

        # SMOTEENN resampling, RFE feature selection and Random Forest Classifier pipeline
        self.smoteenn = smoteenn

        # Recursive Feature Elimination (RFE) feature selection
        self.rfe = rfe
        self.n_features_to_select = None

        # Label encoder
        self.le = LabelEncoder()
        
    def set_data(self, df) -> None:
        self.X = df.drop(columns=['origin_time', 'label'])
        self.y = self.le.fit_transform(df['label'])

        # Needed by the initial train-test split ensure a holdout set for the final evaluation

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    def train_model(self, classifier, params, features_to_select = None, factor=3, aggressive_elimination = False, verbose = 3) -> dict:

        self.classifier = classifier

        # Split the data for cross-validation respecting the temporal order
        self.tscv = TimeSeriesSplit(n_splits=self.n_splits)

        steps = []
        if self.smoteenn:
            steps.append(('smoteenn', SMOTEENN()))
        if self.rfe:
            self.n_features_to_select = features_to_select if features_to_select is not None else len(self.X.columns)
            rfe_step = RFE(estimator=self.classifier, n_features_to_select=self.n_features_to_select, step=1)
            steps.append(('rfe', rfe_step))
        steps.append(('classifier', self.classifier))

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
            return_train_score=True
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
            X_train_selected = self.X_train.iloc[:, selected_features_mask]
            self.model.named_steps['classifier'].fit(X_train_selected, self.y_train)

    def evaluate_model(self):
            if self.rfe:
                selected_features_mask = self.results['rfe']['selected_features_mask']
                X_test_selected = self.X_test.iloc[:, selected_features_mask]
                y_pred = self.model.named_steps['classifier'].predict(X_test_selected)
            else:
                y_pred = self.model.predict(self.X_test)

            accuracy = accuracy_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred, target_names=self.le.classes_)
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            
            self.results.update({
                'accuracy': accuracy,
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