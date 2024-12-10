import pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def cal_SOi(so2):
    si=0
    if (so2<=40):
        si= so2*(50/40)
    elif (so2>40 and so2<=80):
        si= 50+(so2-40)*(50/40)
    elif (so2>80 and so2<=380):
        si= 100+(so2-80)*(100/300)
    elif (so2>380 and so2<=800):
        si= 200+(so2-380)*(100/420)
    elif (so2>800 and so2<=1600):
        si= 300+(so2-800)*(100/800)
    elif (so2>1600):
        si= 400+(so2-1600)*(100/800)
    return si
def cal_Noi(no2):
    ni=0
    if(no2<=40):
        ni= no2*50/40
    elif(no2>40 and no2<=80):
        ni= 50+(no2-40)*(50/40)
    elif(no2>80 and no2<=180):
        ni= 100+(no2-80)*(100/100)
    elif(no2>180 and no2<=280):
        ni= 200+(no2-180)*(100/100)
    elif(no2>280 and no2<=400):
        ni= 300+(no2-280)*(100/120)
    else:
        ni= 400+(no2-400)*(100/120)
    return ni

def cal_RSPMi(rspm):
    rpi=0
    if(rspm<=100):
        rpi = rspm
    elif(rspm>=101 and rspm<=150):
        rpi= 101+(rspm-101)*((200-101)/(150-101))
    elif(rspm>=151 and rspm<=350):
        ni= 201+(rspm-151)*((300-201)/(350-151))
    elif(rspm>=351 and rspm<=420):
        ni= 301+(rspm-351)*((400-301)/(420-351))
    elif(rspm>420):
        ni= 401+(rspm-420)*((500-401)/(420-351))
    return rpi

def cal_SPMi(spm):
    spi=0
    if(spm<=50):
        spi=spm*50/50
    elif(spm>50 and spm<=100):
        spi=50+(spm-50)*(50/50)
    elif(spm>100 and spm<=250):
        spi= 100+(spm-100)*(100/150)
    elif(spm>250 and spm<=350):
        spi=200+(spm-250)*(100/100)
    elif(spm>350 and spm<=430):
        spi=300+(spm-350)*(100/80)
    else:
        spi=400+(spm-430)*(100/430)
    return spi

def cal_pmi(pm2_5):
    pmi=0
    if(pm2_5<=50):
        pmi=pm2_5*(50/50)
    elif(pm2_5>50 and pm2_5<=100):
        pmi=50+(pm2_5-50)*(50/50)
    elif(pm2_5>100 and pm2_5<=250):
        pmi= 100+(pm2_5-100)*(100/150)
    elif(pm2_5>250 and pm2_5<=350):
        pmi=200+(pm2_5-250)*(100/100)
    elif(pm2_5>350 and pm2_5<=450):
        pmi=300+(pm2_5-350)*(100/100)
    else:
        pmi=400+(pm2_5-430)*(100/80)
    return pmi

st.set_page_config(layout="wide")

st.title('AQI Predictions')
KNN = pickle.load(open('KNNReg.pkl', 'rb'))
Dtree = pickle.load(open('DecisionTreeReg.pkl', 'rb'))
LR = pickle.load(open('LinearReg.pkl','rb'))
Ridge = pickle.load(open('RidgeReg.pkl', 'rb'))

col1, col2 = st.columns(2, vertical_alignment="top", gap="large")


with col1:
    c1,c2 = st.columns(2,gap="large")
    with c1:
        so2 = st.number_input( "Insert SO2", value=0)
        so2 = cal_SOi(so2)
        no2 = st.number_input("Insert NO2", value=0)
        no2 = cal_Noi(no2)
        rspm = st.number_input("Insert RSPM", value=0)
        rspm = cal_RSPMi(rspm)
    with c2:
        spm = st.number_input("Insert SPM", value=0)
        spm = cal_SPMi(spm)
        pm2_5 = st.number_input("Insert PM 2.5", value=0)
        pm2_5 = cal_pmi(pm2_5)



with col2:
    aqi_knn = KNN.predict([[so2, no2, rspm, spm, pm2_5]])
    aqi_dtree = Dtree.predict([[so2, no2, rspm, spm, pm2_5]])
    aqi_lr = LR.predict([[so2, no2, rspm, spm, pm2_5]])
    aqi_ridge = Ridge.predict([[so2, no2, rspm, spm, pm2_5]])

    st.header("Output")

    st.write(
    pd.DataFrame(
        {
            "Model": ["KNN", "Decision Tree", "Linear Regression", "Ridge Regression"],
            "AQI": [np.round(aqi_knn),np.round(aqi_dtree),np.round(aqi_lr),np.round(aqi_ridge)],
            "Accuracy": ["99.95%","99.91%","97.90%","97.90%"],
            "RMSE": ["1.90","2.59","12.63","12.63"]
        }
    )
)

levels = ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"]
ranges = [50, 100, 150, 200, 300, 500]
colors = ["green", "yellow", "orange", "red", "purple", "maroon"]
sectors = [
        {"range": [0, ranges[0]], "color": colors[0]},
        {"range": [ranges[0], ranges[1]], "color": colors[1]},
        {"range": [ranges[1], ranges[2]], "color": colors[2]},
        {"range": [ranges[2], ranges[3]], "color": colors[3]},
        {"range": [ranges[3], ranges[4]], "color": colors[4]},
        {"range": [ranges[4], ranges[5]], "color": colors[5]},
]

col1_1,col2_1,col3_1,col4_1 = st.columns(4, gap="medium")

with col1_1:
    aqi = np.round(aqi_knn)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi[0],
        title={'text': "KNN: Air Quality Index (AQI)"},
        gauge={
            'axis': {'range': [0, 500], 'visible': False},  # Hide the axis line
            'steps': [{"range": sector["range"], "color": sector["color"]} for sector in sectors],
            'bar': {'color': "white"}  # Hide the center bar
        }
    ))

    fig.update_layout(
        width=300,
        height=200,
        margin=dict(t=50, b=50, l=50, r=50),
        paper_bgcolor="black"  # Set background to white
    )

    st.plotly_chart(fig, theme="streamlit")

with col2_1:
    aqi = np.round(aqi_dtree)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi[0],
        title={'text': "Decision Tree: Air Quality Index (AQI)"},
        gauge={
            'axis': {'range': [0, 500], 'visible': False},  # Hide the axis line
            'steps': [{"range": sector["range"], "color": sector["color"]} for sector in sectors],
            'bar': {'color': "white"}  # Hide the center bar
        }
    ))

    fig.update_layout(
        width=300,
        height=200,
        margin=dict(t=50, b=50, l=50, r=50),
        paper_bgcolor="black"  # Set background to white
    )

    st.plotly_chart(fig, theme="streamlit")

with col3_1:
    aqi = np.round(aqi_lr)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi[0],
        title={'text': "Linear: Air Quality Index (AQI)"},
        gauge={
            'axis': {'range': [0, 500], 'visible': False},  # Hide the axis line
            'steps': [{"range": sector["range"], "color": sector["color"]} for sector in sectors],
            'bar': {'color': "white"}  # Hide the center bar
        }
    ))

    fig.update_layout(
        width=300,
        height=200,
        margin=dict(t=50, b=50, l=50, r=50),
        paper_bgcolor="black"  # Set background to white
    )

    st.plotly_chart(fig, theme="streamlit")

with col4_1:
    aqi = np.round(aqi_ridge)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi[0],
        title={'text': "Ridge: Air Quality Index (AQI)"},
        gauge={
            'axis': {'range': [0, 500], 'visible': False},  # Hide the axis line
            'steps': [{"range": sector["range"], "color": sector["color"]} for sector in sectors],
            'bar': {'color': "white"}  # Hide the center bar
        }
    ))

    fig.update_layout(
        width=300,
        height=200,
        margin=dict(t=50, b=50, l=50, r=50),
        paper_bgcolor="black"  # Set background to white
    )

    st.plotly_chart(fig, theme="streamlit")

