import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import lightgbm as lgb
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

class TourismDataProcessor:
    def __init__(self):
        self.df = None
        self.rating_model = None
        self.visit_mode_model = None
        self.label_encoder = None
        self.recommendation_system = None
        self.X_reg = None
        self.y_reg = None
        self.X_clf = None
        self.y_clf = None
        # New attributes for recommendation system
        self.cosine_sim = None
        self.attraction_indices = None
        self.attraction_features = None

    def load(self):
        transaction = pd.read_excel('Transaction.xlsx')
        user = pd.read_excel('User.xlsx')
        city = pd.read_excel('City.xlsx')
        attraction_type = pd.read_excel('Type.xlsx')
        visit_mode = pd.read_excel('Mode.xlsx')
        continent = pd.read_excel('Continent.xlsx')
        country = pd.read_excel('Country.xlsx')
        region = pd.read_excel('Region.xlsx')
        attraction = pd.read_excel('Item.xlsx')
        return transaction, user, city, attraction_type, visit_mode, continent, country, region, attraction
    
    def clean(self, df):
        cols_to_keep = [
            'UserId', 'VisitYear', 'VisitMonth', 'VisitMode', 'AttractionId', 'Rating',
            'Continent', 'Region', 'Country', 'CityName', 'AttractionType', 
            'Attraction', 'AttractionAddress'
        ]
        df = df.loc[:, ~df.columns.duplicated()]
        df = df[[col for col in cols_to_keep if col in df.columns]]
        df = df.dropna(subset=['Rating'])
        for col in ['VisitMode', 'Continent', 'Region', 'Country', 'CityName', 'AttractionType']:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        categorical_cols = ['VisitMode', 'Continent', 'Region', 'Country', 'CityName', 'AttractionType']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        return df

    def merge(self, transaction, user, city, attraction_type, visit_mode, 
              continent, country, region, attraction):
        print("\nInitial DataFrame Columns:")
        print("Transaction:", transaction.columns.tolist())
        print("User:", user.columns.tolist())
        print("City:", city.columns.tolist())
        print("Country:", country.columns.tolist())
        print("Region:", region.columns.tolist())
        print("Continent:", continent.columns.tolist())
        print("Attraction:", attraction.columns.tolist())
        print("Visit Mode:", visit_mode.columns.tolist())
        print("Attraction Type:", attraction_type.columns.tolist())

        visit_mode['VisitMode'] = visit_mode['VisitMode'].astype(str)
        transaction['VisitMode'] = transaction['VisitMode'].astype(str)

        user = user.merge(city, on='CityId', how='left')
        if 'CountryId_y' in user.columns:
            user = user.rename(columns={'CountryId_y': 'CountryId'})
            if 'CountryId_x' in user.columns:
                user = user.drop(columns=['CountryId_x'])
        user = user.merge(country, on='CountryId', how='left')
        if 'RegionId_y' in user.columns:
            user = user.rename(columns={'RegionId_y': 'RegionId'})
            if 'RegionId_x' in user.columns:
                user = user.drop(columns=['RegionId_x'])
        user = user.merge(region, on='RegionId', how='left')
        if 'ContinentId_y' in user.columns:
            user = user.rename(columns={'ContinentId_y': 'ContinentId'})
            if 'ContinentId_x' in user.columns:
                user = user.drop(columns=['ContinentId_x'])
        user = user.merge(continent, on='ContinentId', how='left')

        attraction = attraction.merge(city, left_on='AttractionCityId', right_on='CityId', how='left')
        for col in ['CountryId', 'RegionId', 'ContinentId']:
            if f'{col}_y' in attraction.columns:
                attraction = attraction.rename(columns={f'{col}_y': col})
                if f'{col}_x' in attraction.columns:
                    attraction = attraction.drop(columns=[f'{col}_x'])
        attraction = attraction.merge(country, on='CountryId', how='left')
        attraction = attraction.merge(region, on='RegionId', how='left')
        attraction = attraction.merge(continent, on='ContinentId', how='left')
        attraction = attraction.merge(attraction_type, on='AttractionTypeId', how='left')

        df = transaction.merge(user, on='UserId', how='left')
        df = df.merge(attraction, on='AttractionId', how='left')
    
        if 'VisitMode' in df.columns and 'VisitModeId' in visit_mode.columns:
            df = df.merge(visit_mode, on='VisitMode', how='left')
        elif 'VisitModeId' in df.columns:
            df = df.merge(visit_mode, on='VisitModeId', how='left')
        else:
            raise ValueError("Cannot merge with visit_mode - missing both VisitMode and VisitModeId")

        df.columns = [col.replace('_x', '').replace('_y', '') for col in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]  # Additional deduplication
        
        cols_to_keep = [
            'UserId', 'VisitYear', 'VisitMonth', 'VisitMode', 'AttractionId', 'Rating',
            'Continent', 'Region', 'Country', 'CityName', 'AttractionType', 
            'Attraction', 'AttractionAddress'
        ]
        df = df[[col for col in cols_to_keep if col in df.columns]]
    
        print("\nFinal Merged DataFrame Columns:", df.columns.tolist())
        return df

    def feature_engineering(self, df):
        user_stats = df.groupby('UserId').agg({
            'Rating': 'mean',
            'AttractionId': 'count',
            'VisitMode': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown',
            'Continent': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
        }).rename(columns={
            'Rating': 'UserAvgRating',
            'AttractionId': 'UserVisitCount',
            'VisitMode': 'UserCommonVisitMode',
            'Continent': 'UserContinent'
        })
        
        attraction_stats = df.groupby('AttractionId').agg({
            'Rating': 'mean',
            'UserId': 'count',
            'VisitMode': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
        }).rename(columns={
            'Rating': 'AttractionAvgRating',
            'UserId': 'AttractionVisitCount',
            'VisitMode': 'AttractionCommonVisitMode'
        })
        
        df = df.merge(user_stats, on='UserId', how='left')
        df = df.merge(attraction_stats, on='AttractionId', how='left')
        
        current_year = 2025
        df['YearsSinceFirstVisit'] = current_year - df['VisitYear']
        
        return df

    def prepare_regression(self, df):
        features = [
            'UserAvgRating', 'UserVisitCount', 'UserCommonVisitMode', 'UserContinent',
            'AttractionAvgRating', 'AttractionVisitCount', 'AttractionCommonVisitMode',
            'AttractionType', 'VisitMode', 'Continent', 'YearsSinceFirstVisit'
        ]
        features = [f for f in features if f in df.columns]
        X = df[features]
        y = df['Rating']
        
        categorical_cols = X.select_dtypes(include=['category', 'object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        return X, y

    def train_regression_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        lgbm = lgb.LGBMRegressor(random_state=42)
        lgbm.fit(X_train, y_train)
        xgb_model = xgb.XGBRegressor(random_state=42)
        xgb_model.fit(X_train, y_train)
        
        models = {'Random Forest': rf, 'LightGBM': lgbm, 'XGBoost': xgb_model}
        results = {}
        for name, model in models.items():
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            results[name] = {'RMSE': rmse, 'R2': r2}
        
        results_df = pd.DataFrame(results).T
        best_model_name = results_df['R2'].idxmax()
        best_model = models[best_model_name]
        print(f"Best regression model: {best_model_name} with R2: {results_df.loc[best_model_name, 'R2']:.3f}")
        return best_model, results_df

    def prepare_classification(self, df):
        features = [
            'UserAvgRating', 'UserVisitCount', 'UserCommonVisitMode', 'UserContinent', 'UserCountry',
            'AttractionAvgRating', 'AttractionVisitCount', 'AttractionCommonVisitMode',
            'AttractionType', 'Continent', 'YearsSinceFirstVisit', 'VisitMonth'
        ]
        features = [f for f in features if f in df.columns]
        X = df[features]
        y = df['VisitMode']
        
        categorical_cols = X.select_dtypes(include=['category', 'object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        le = LabelEncoder()
        y = le.fit_transform(y)
        return X, y, le

    def train_classification_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        lgbm = lgb.LGBMClassifier(random_state=42)
        lgbm.fit(X_train, y_train)
        xgb_model = xgb.XGBClassifier(random_state=42)
        xgb_model.fit(X_train, y_train)
        
        models = {'Random Forest': rf, 'LightGBM': lgbm, 'XGBoost': xgb_model}
        results = {}
        for name, model in models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            results[name] = {
                'Accuracy': accuracy,
                'Precision (weighted)': report['weighted avg']['precision'],
                'Recall (weighted)': report['weighted avg']['recall'],
                'F1 (weighted)': report['weighted avg']['f1-score']
            }
        
        results_df = pd.DataFrame(results).T
        best_model_name = results_df['Accuracy'].idxmax()
        best_model = models[best_model_name]
        print(f"Best classification model: {best_model_name} with Accuracy: {results_df.loc[best_model_name, 'Accuracy']:.3f}")
        return best_model, results_df

    def build_recommendation_system(self, df):
        # Use the full df to ensure all AttractionIds are included
        self.attraction_features = df[['AttractionId', 'AttractionType', 'Continent', 'Country']].drop_duplicates()
        self.attraction_features['combined_features'] = self.attraction_features.apply(
            lambda x: f"{x['AttractionType']} {x['Continent']} {x['Country']}", axis=1)
    
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(self.attraction_features['combined_features'])
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        self.attraction_indices = pd.Series(self.attraction_features.index, index=self.attraction_features['AttractionId']).drop_duplicates()
    
        self.recommendation_system = self.hybrid_recommendations

    def hybrid_recommendations(self, user_id, top_n=5):
        user_ratings = self.df[self.df['UserId'] == user_id]
        if user_ratings.empty:
            return self.df.groupby('Attraction')['Rating'].mean().sort_values(ascending=False).head(top_n)
        
        liked_attractions = user_ratings[user_ratings['Rating'] >= 4]['AttractionId'].unique()
        if len(liked_attractions) == 0:
            liked_attractions = user_ratings['AttractionId'].unique()
        
        sim_scores = []
        for att_id in liked_attractions:
            if att_id in self.attraction_indices:
                idx = self.attraction_indices[att_id]
                sim_scores.append(self.cosine_sim[idx])
        
        if not sim_scores:
            return pd.Series([])
        
        avg_sim_scores = np.mean(sim_scores, axis=0)
        content_recs = list(zip(self.attraction_features['AttractionId'], avg_sim_scores))
        content_recs = sorted(content_recs, key=lambda x: x[1], reverse=True)
        content_recs = [rec for rec in content_recs if rec[0] not in user_ratings['AttractionId'].values]
        content_recs = content_recs[:top_n]
        
        recs_with_names = []
        for att_id, score in content_recs:
            att = self.df[self.df['AttractionId'] == att_id].iloc[0]
            recs_with_names.append((att['Attraction'], att['AttractionType'], 
                                   att['Country'], score))
        
        return recs_with_names

    def process(self):
        transaction, user, city, attraction_type, visit_mode, continent, country, region, attraction = self.load()
        df = self.merge(transaction, user, city, attraction_type, visit_mode, continent, country, region, attraction)
        df = self.clean(df)
        df = self.feature_engineering(df)
        self.df = df
        
        self.X_reg, self.y_reg = self.prepare_regression(df)
        self.rating_model, reg_results = self.train_regression_model(self.X_reg, self.y_reg)
        
        self.X_clf, self.y_clf, self.label_encoder = self.prepare_classification(df)
        self.visit_mode_model, clf_results = self.train_classification_model(self.X_clf, self.y_clf)
        
        self.build_recommendation_system(df)
        
        print("\nModel Training Complete!")
        print("Regression Results:")
        print(reg_results)
        print("\nClassification Results:")
        print(clf_results)

    def save_models(self):
        with open('tourism_models.pkl', 'wb') as f:
            pickle.dump({
                'df': self.df,
                'rating_model': self.rating_model,
                'visit_mode_model': self.visit_mode_model,
                'label_encoder': self.label_encoder,
                'cosine_sim': self.cosine_sim,
                'attraction_indices': self.attraction_indices,
                'attraction_features': self.attraction_features,
                'X_reg': self.X_reg,
                'X_clf': self.X_clf
            }, f)

if __name__ == '__main__':
    processor = TourismDataProcessor()
    processor.process()
    processor.save_models()
    print("Data processing complete. Models saved to 'tourism_models.pkl'")