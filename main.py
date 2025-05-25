import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the wine quality dataset


@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(url, sep=';')
    return data

# Train the model


@st.cache_resource
def train_model(data):
    X = data.drop('quality', axis=1)
    y = data['quality']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("SSSSSSSSSS")
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)

    return model, scaler, mse

# Main function


def main():
    st.set_page_config(
        page_title="Прогноз качества вина",  # Заголовок страницы
        page_icon="🍷",          # Иконка (опционально)
    )
    st.title("🍷 Прогноз качества вина")
    st.write("""
    Это приложение прогнозирует качество вина на основе его химических характеристик.
Отрегулируйте ползунки, чтобы ввести параметры вина и увидеть прогнозируемое качество.
    """)

    # Load data
    data = load_data()

    # Train model
    model, scaler, mse = train_model(data)
    st.sidebar.title("🍷 Прогноз качества вина")
    # Sidebar with user input
    st.sidebar.header('Входные параметры вина')

    def user_input_features():
        fixed_acidity = st.sidebar.slider('Fixed acidity', float(data['fixed acidity'].min(
        )), float(data['fixed acidity'].max()), float(data['fixed acidity'].mean()))
        volatile_acidity = st.sidebar.slider('Volatile acidity', float(data['volatile acidity'].min(
        )), float(data['volatile acidity'].max()), float(data['volatile acidity'].mean()))
        citric_acid = st.sidebar.slider('Citric acid', float(data['citric acid'].min(
        )), float(data['citric acid'].max()), float(data['citric acid'].mean()))
        residual_sugar = st.sidebar.slider('Residual sugar', float(data['residual sugar'].min(
        )), float(data['residual sugar'].max()), float(data['residual sugar'].mean()))
        chlorides = st.sidebar.slider('Chlorides', float(data['chlorides'].min()), float(
            data['chlorides'].max()), float(data['chlorides'].mean()))
        free_sulfur_dioxide = st.sidebar.slider('Free sulfur dioxide', float(data['free sulfur dioxide'].min(
        )), float(data['free sulfur dioxide'].max()), float(data['free sulfur dioxide'].mean()))
        total_sulfur_dioxide = st.sidebar.slider('Total sulfur dioxide', float(data['total sulfur dioxide'].min(
        )), float(data['total sulfur dioxide'].max()), float(data['total sulfur dioxide'].mean()))
        density = st.sidebar.slider('Density', float(data['density'].min()), float(
            data['density'].max()), float(data['density'].mean()))
        pH = st.sidebar.slider('pH', float(data['pH'].min()), float(
            data['pH'].max()), float(data['pH'].mean()))
        sulphates = st.sidebar.slider('Sulphates', float(data['sulphates'].min()), float(
            data['sulphates'].max()), float(data['sulphates'].mean()))
        alcohol = st.sidebar.slider('Alcohol', float(data['alcohol'].min()), float(
            data['alcohol'].max()), float(data['alcohol'].mean()))

        features = pd.DataFrame({
            'fixed acidity': [fixed_acidity],
            'volatile acidity': [volatile_acidity],
            'citric acid': [citric_acid],
            'residual sugar': [residual_sugar],
            'chlorides': [chlorides],
            'free sulfur dioxide': [free_sulfur_dioxide],
            'total sulfur dioxide': [total_sulfur_dioxide],
            'density': [density],
            'pH': [pH],
            'sulphates': [sulphates],
            'alcohol': [alcohol]
        })
        return features

    input_df = user_input_features()

    # Display user input
    st.subheader('Параметры, введенные пользователем')
    st.write(input_df)

    # Make prediction
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    st.subheader('Прогнозируемое качество вина')
    st.write(f"**{prediction[0]:.1f}** из 10")

    # Show quality scale interpretation
    st.write("""
    **Интепретация результатов:**
    - прогноз <= 5 : Низкое качество 
    - 5 < прогноз < 7: Среднее качество
    - 7 <= прогноз: Bысокое качество
    """)

    # Feature importance
    st.subheader('Bажности признаков для данного прогноза')

    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = input_df.columns

    # Create a DataFrame for feature importance
    feature_importance = pd.DataFrame({
        'Признаки': features,
        'Значение': importances
    }).sort_values(by='Значение', ascending=False)

    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Значение', y='Признаки', hue='Признаки',
                data=feature_importance, palette='viridis', legend=False)
    plt.title('Bажность признаков')
    plt.xlabel('Относительные значения')
    st.pyplot(fig)

    # Model performance info
    st.sidebar.subheader('Model Information')
    st.sidebar.write(f"Model: Random Forest Regressor")
    st.sidebar.write(f"Mean Squared Error: {mse:.2f}")
    st.sidebar.write(f"Dataset size: {len(data)} samples")

    # Data exploration
    if st.checkbox('Посмотреть датасет'):
        st.subheader('Wine Quality Dataset')
        url = "https://www.kaggle.com/datasets/rajyellow46/wine-quality"
        st.write("Данные для обучения модели взяты из [kaggle](%s)" % url)
        st.write(data)

    if st.checkbox('Посмотреть распределения признаков'):
        st.subheader('Распределение признака')
        selected_feature = st.selectbox(
            'Bыберите признак для визуализации распределения', data.columns[:-1])
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data[selected_feature], kde=True, bins=30)
        plt.axvline(x=input_df[selected_feature].values[0],
                    color='r', linestyle='--', label='Bаше введенное значение')
        plt.title(f'Распределение признака {selected_feature}')
        plt.legend()
        st.pyplot(fig)


if __name__ == '__main__':
    main()
