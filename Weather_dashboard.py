import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import streamlit as st
import joblib

st.set_page_config(layout="wide")

st.title("Rain Prediction Model 🌧️")

df = pd.read_csv("weatherAUS.csv")

disp_chk = st.sidebar.checkbox("Display Original Data")
if disp_chk:
    st.write(df)

st.sidebar.title("Lets Go !! 🏃‍➡️")

sel_task_radio = st.sidebar.radio("Select Task",["EDA Report","Train Model","Predict Data"])

if sel_task_radio == "EDA Report":

    sel_type = st.sidebar.selectbox("Select Type: ",["Descriptive Statistics","Univariate Analysis","Bivariate Analysis","Multivariate Analysis"])

    if sel_type == "Descriptive Statistics":
        st.subheader("Descriptive Analysis !")
        st.write(df.describe())

    elif sel_type == "Univariate Analysis":
        sel_col = st.selectbox("Select Column",df.columns)
        col1, col2 = st.columns([1,1])
        if df[sel_col].dtype == "object":
            counts = df[sel_col].value_counts().reset_index()

            fig1 = px.histogram(df, x=sel_col, title=f"Count of {sel_col}")
            col1.plotly_chart(fig1)

            fig2 = px.pie(counts, values='count',names=sel_col)
            col2.plotly_chart(fig2)

        else:
            fig1 = px.histogram(df, x=sel_col, nbins=50, title=f"Distribution of {sel_col}")
            col1.plotly_chart(fig1)

            fig2 = px.box(df, y=sel_col, title=f"Boxplot of {sel_col}")
            col2.plotly_chart(fig2)

    elif sel_type == "Bivariate Analysis":
        x_col = st.selectbox("Select X column",df.columns)
        y_col = st.selectbox("Select Y column",df.columns)

        st.subheader(f"Bivariate Analysis: {x_col} vs {y_col}")

        if df[x_col].dtype != "object" and df[y_col].dtype != "object":
            fig = px.scatter(df, x=x_col, y=y_col,title=f"f{x_col} vs {y_col}")

        else:
            fig = px.box(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")

        st.plotly_chart(fig, use_container_width=True)

    elif sel_type == "Multivariate Analysis":

        x_col = st.selectbox("Select X column",df.columns)
        y_col = st.selectbox("Select Y column",df.columns)
        color_col = st.selectbox("Select color column",df.columns)

        st.subheader("Multivariate Analysis")

        fig = px.scatter(df,x=x_col,y=y_col, color=color_col, title=f"{x_col} vs {y_col} colored by {color_col}")

        st.plotly_chart(fig, use_container_width=True)

elif sel_task_radio == "Train Model":

    df = df.drop(columns=['Date','Evaporation','Sunshine','Cloud9am','Cloud3pm'])
    df1 = df.dropna().reset_index()

    sel_encoder = st.sidebar.selectbox("Select Encoder",["LabelEncoder","OneHotEncoder"])

    if sel_encoder == "LabelEncoder":

        encoder_location = LabelEncoder()
        encoder_WindGustDir = LabelEncoder()
        encoder_WindDir9am = LabelEncoder()
        encoder_WindDir3pm = LabelEncoder()
        encoder_RainToday = LabelEncoder()
        encoder_RainTomorrow = LabelEncoder()

        df1['Location'] = encoder_location.fit_transform(df1['Location'])
        df1['WindGustDir'] = encoder_WindGustDir.fit_transform(df1['WindGustDir'])
        df1['WindDir9am'] = encoder_WindDir9am.fit_transform(df1['WindDir9am'])
        df1['WindDir3pm'] = encoder_WindDir3pm.fit_transform(df1['WindDir3pm'])
        df1['RainToday'] = encoder_RainToday.fit_transform(df1['RainToday'])
        df1['RainTomorrow'] = encoder_RainTomorrow.fit_transform(df1['RainTomorrow'])

        X = df1.drop('RainTomorrow',axis=1)
        Y = df1['RainTomorrow']

    sel_scaling = st.sidebar.selectbox("Select Scaler",["StandardScaler","MinMaxScaler"])

    if sel_scaling == "StandardScaler":

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    testsize = st.sidebar.slider("Select Test Size",0.1,0.95,0.2,0.05)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y, test_size=testsize)

    s1 = st.selectbox("Select Model: ",["RandomForest","XGBoosting","DecisionTree"])

    model_rf = RandomForestClassifier()
    model_xg = XGBClassifier()
    model_dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
    if s1 == "RandomForest":
        model = model_rf
        model.fit(Xtrain,Ytrain)

    elif s1 == "XGBoosting":
        model = model_xg
        model.fit(Xtrain,Ytrain)

    elif s1 == "DecisionTree":
        model = model_dt
        model.fit(Xtrain,Ytrain)

    save_btn = st.sidebar.button("💾 Save Model")

    if save_btn:
        model_dict = {}

        model_dict["RandomForest"] = model_rf.fit(Xtrain, Ytrain)
        model_dict["XGBoosting"] = model_xg.fit(Xtrain, Ytrain)
        model_dict["DecisionTree"] = model_dt.fit(Xtrain,Ytrain)

        defaults = df1.drop("RainTomorrow",axis=1).mean().to_dict()

        pipeline = {
            "models": model_dict,
            "scaler": scaler,
            "encoders": {
                "Location": encoder_location,
                "WindGustDir": encoder_WindGustDir,
                "WindDir9am": encoder_WindDir9am,
                "WindDir3pm": encoder_WindDir3pm,
                "RainToday": encoder_RainToday
            },
            "columns": df1.drop('RainTomorrow', axis=1).columns.tolist(),
            "defaults": defaults
        }

        joblib.dump(pipeline, "C:/Users/aadia/Downloads/data_models_cls.pkl")

    b1 = st.button("Predict")

    if b1:

        st.subheader("Performance Of The Model 🚀")

        Ypred = model.predict(Xtest)
        acc = accuracy_score(Ytest,Ypred)
        pre = precision_score(Ytest,Ypred, average= None)
        rec = recall_score(Ytest,Ypred, average= None)
        f1 = f1_score(Ytest,Ypred, average= None)

        st.metric("Accuracy Score",f"{acc:.2f}")

        c1,c2 = st.columns(2)

        with c1:
            st.markdown("NO RAIN")
            st.metric("Precision", f"{pre[0]:.2f}")
            st.metric("Recall: ",f"{rec[0]:.2f}")
            st.metric("F1 Score: ",f"{f1[0]:.2f}")

        with c2:
            st.markdown("RAIN")
            st.metric("Precision", f"{pre[1]:.2f}")
            st.metric("Recall: ", f"{rec[1]:.2f}")
            st.metric("F1 Score: ", f"{f1[1]:.2f}")


elif sel_task_radio == "Predict Data":
        pipeline = joblib.load("C:/Users/aadia/Downloads/data_models_cls.pkl")
        models = pipeline["models"]
        scaler = pipeline["scaler"]
        encoders = pipeline["encoders"]
        columns = pipeline["columns"]
        defaults = pipeline["defaults"]

        sel_model = st.selectbox("Select Model",models.keys())

        model = models[sel_model]

        important_features = [
            'Humidity3pm',
            'Pressure3pm',
            'Temp3pm',
            'Rainfall',
            'WindSpeed3pm',
            'RainToday',
        ]

        user_input = {}

        for col in important_features:

            if col in encoders:
                val = st.selectbox(col,encoders[col].classes_)
                user_input[col] = encoders[col].transform([val])[0]

            else:
                user_input[col] = st.number_input(col, value=0.0)

        full_input = defaults.copy()
        full_input.update(user_input)

        for col in user_input:
            full_input[col] = user_input[col]

        input_df = pd.DataFrame([full_input])

        input_scaled = scaler.transform(input_df)

        if st.button("Predict"):

            pred = model.predict(input_scaled)
            prob = model.predict_proba(input_scaled)

            prob = prob[0][1]  # rain probability

            if pred[0] == 1:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg,#4facfe,#00f2fe);
                            padding:20px;
                            border-radius:15px;
                            text-align:center;
                            color:white;
                            font-size:22px;
                            font-weight:bold;">
                    🌧️ Rain Expected <br>
                    <span style="font-size:28px;">{prob * 100:.2f}%</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg,#43e97b,#38f9d7);
                            padding:20px;
                            border-radius:15px;
                            text-align:center;
                            color:white;
                            font-size:22px;
                            font-weight:bold;">
                    ☀️ No Rain <br>
                    <span style="font-size:28px;">{(1 - prob) * 100:.2f}%</span>
                </div>
                """, unsafe_allow_html=True)







