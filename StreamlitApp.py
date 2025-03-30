import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Tourism Recommendation", layout="wide")
# Load the processed data and models
@st.cache_data
def load_models():
    with open('tourism_models.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

data = load_models()
df = data['df']
rating_model = data['rating_model']
visit_mode_model = data['visit_mode_model']
label_encoder = data['label_encoder']
cosine_sim = data['cosine_sim']
attraction_indices = data['attraction_indices']
attraction_features = data['attraction_features']
X_reg = data['X_reg']
X_clf = data['X_clf']

# Define the recommendation function with the fix
def hybrid_recommendations(user_id, top_n=5):
    user_ratings = df[df['UserId'] == user_id]
    if user_ratings.empty:
        return df.groupby('Attraction')['Rating'].mean().sort_values(ascending=False).head(top_n)
    
    liked_attractions = user_ratings[user_ratings['Rating'] >= 4]['AttractionId'].unique()
    if len(liked_attractions) == 0:
        liked_attractions = user_ratings['AttractionId'].unique()
    
    sim_scores = []
    for att_id in liked_attractions:
        if att_id in attraction_indices:
            # Ensure idx is a scalar integer
            idx = attraction_indices[att_id]
            if isinstance(idx, pd.Series):
                idx = idx.iloc[0]  # Take the first index if multiple exist
            if idx < cosine_sim.shape[0]:  # Safe comparison with scalar idx
                sim_scores.append(cosine_sim[idx])
            else:
                print(f"Warning: Index {idx} for AttractionId {att_id} is out of bounds for cosine_sim with size {cosine_sim.shape[0]}")
    
    if not sim_scores:
        return df.groupby('Attraction')['Rating'].mean().sort_values(ascending=False).head(top_n)  # Fallback to popular attractions
    
    avg_sim_scores = np.mean(sim_scores, axis=0)
    content_recs = list(zip(attraction_features['AttractionId'], avg_sim_scores))
    content_recs = sorted(content_recs, key=lambda x: x[1], reverse=True)
    content_recs = [rec for rec in content_recs if rec[0] not in user_ratings['AttractionId'].values]
    content_recs = content_recs[:top_n]
    
    recs_with_names = []
    for att_id, score in content_recs:
        att = df[df['AttractionId'] == att_id].iloc[0]
        recs_with_names.append((att['Attraction'], att['AttractionType'], 
                               att['Country'], score))
    
    return recs_with_names

# Define the input preparation function
def prepare_input_for_prediction(user_data, attraction_data, visit_mode_name):
    # Prepare input for rating prediction
    input_data = {
        'UserAvgRating': user_data['UserAvgRating'],
        'UserVisitCount': user_data['UserVisitCount'],
        'UserCommonVisitMode': user_data['UserCommonVisitMode'],
        'UserContinent': user_data['UserContinent'],
        'AttractionAvgRating': attraction_data['AttractionAvgRating'],
        'AttractionVisitCount': attraction_data['AttractionVisitCount'],
        'AttractionCommonVisitMode': attraction_data['AttractionCommonVisitMode'],
        'AttractionType': attraction_data['AttractionType'],
        'VisitMode': visit_mode_name,
        'Continent': attraction_data['Continent'],
        'YearsSinceFirstVisit': user_data['YearsSinceFirstVisit']
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical variables using the same encoders as training
    categorical_cols = ['UserCommonVisitMode', 'UserContinent', 'AttractionCommonVisitMode', 
                       'AttractionType', 'VisitMode', 'Continent']
    
    for col in categorical_cols:
        if col in X_reg.columns:
            le = LabelEncoder()
            # Fit with all possible values from training data to avoid unseen labels
            le.fit(X_reg[col].astype(str))
            
            # Transform input, handling unseen labels
            input_col = input_df[col].astype(str).values
            unseen = ~np.isin(input_col, le.classes_)
            if any(unseen):
                # Replace unseen labels with a default value
                default_label = le.classes_[0]
                input_col[unseen] = default_label
            input_df[col] = le.transform(input_col)
    
    return input_df

# Define the EDA function
def perform_eda(df):
    st.title("Tourism Data Explorer")
    
    # Rating distribution
    st.subheader("Rating Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Rating'], bins=5, kde=True, ax=ax)
    st.pyplot(fig)
    
    # Visit mode distribution
    st.subheader("Visit Mode Distribution")
    fig, ax = plt.subplots()
    df['VisitMode'].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)
    
    # Top attractions by average rating
    st.subheader("Top Attractions by Average Rating")
    top_attractions = df.groupby('Attraction')['Rating'].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots()
    top_attractions.plot(kind='barh', ax=ax)
    st.pyplot(fig)
    
    # Rating by continent
    st.subheader("Average Rating by Continent")
    rating_by_continent = df.groupby('Continent')['Rating'].mean().sort_values()
    fig, ax = plt.subplots()
    rating_by_continent.plot(kind='barh', ax=ax)
    st.pyplot(fig)
    
    # Visit mode by continent
    st.subheader("Visit Mode Distribution by Continent")
    fig, ax = plt.subplots(figsize=(10, 6))
    pd.crosstab(df['Continent'], df['VisitMode']).plot(kind='bar', stacked=True, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Main app function
def run_app():
    st.title("Tourism Recommendation and Prediction System")
    
    # Sidebar for user input
    st.sidebar.header("User Input")
    user_id = st.sidebar.selectbox("User ID", df['UserId'].unique())
    
    # Get user data
    user_data = df[df['UserId'] == user_id].iloc[0] if user_id in df['UserId'].values else None
    
    if user_data is not None:
        # Display user info
        st.subheader("User Profile")
        col1, col2, col3 = st.columns(3)
        col1.metric("Continent", user_data['Continent'])
        col2.metric("Country", user_data['Country'])
        col3.metric("City", user_data['CityName'])
        
        # Predict visit mode
        st.subheader("Predicted Visit Mode")
        X_user = X_clf[df['UserId'] == user_id].iloc[0:1]
        visit_mode_pred = visit_mode_model.predict(X_user)
        visit_mode_name = label_encoder.inverse_transform(visit_mode_pred)[0]
        st.write(f"The predicted visit mode for this user is: **{visit_mode_name}**")
        
        # Rating prediction section
        st.subheader("Rating Predictor")
        attraction_name = st.selectbox("Select Attraction", df['Attraction'].unique())
        attraction_data = df[df['Attraction'] == attraction_name].iloc[0]
        
        # Prepare input and predict
        input_df = prepare_input_for_prediction(user_data, attraction_data, visit_mode_name)
        predicted_rating = rating_model.predict(input_df)[0]
        
        st.write(f"Predicted rating for {attraction_name}: **{predicted_rating:.1f}/5**")
        
        # Recommendations section
        st.subheader("Personalized Recommendations")
        recs = hybrid_recommendations(user_id)
        
        if recs:
            st.write("Top attractions you might like:")
            for att_name, att_type, country, score in recs:
                st.write(f"- **{att_name}** ({att_type}, {country}) - Similarity: {score:.2f}")
        else:
            st.write("Popular attractions:")
            popular = df.groupby('Attraction')['Rating'].mean().sort_values(ascending=False).head(5)
            for att_name, rating in popular.items():
                st.write(f"- **{att_name}** (Avg Rating: {rating:.1f})")
    
    # EDA Section
    st.header("Data Exploration")
    perform_eda(df)

if __name__ == '__main__':
    run_app()
