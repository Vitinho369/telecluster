import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt


@st.cache_resource
def load_model():
    return joblib.load("telco_kmeans_pipeline.pkl")

pipe = load_model()

st.title("TeleCluster - Agrupamento Inteligente de Perfis de Clientes")

st.divider()

st.subheader("Preencha as informações do cliente")

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gênero", ["Masculino", "Feminino"], index=None)
        SeniorCitizen = st.selectbox("Idoso?", ["Sim", "Não"], index=None)
        Partner = st.selectbox("Tem Parceiro?", ["Sim", "Não"], index=None)
        Dependents = st.selectbox("Dependentes", ["Sim", "Não"], index=None)
        tenure = st.number_input("Meses no plano", min_value=0, max_value=100, value=0)
        PhoneService = st.selectbox("Telefone", ["Sim", "Não"], index=None)
        MultipleLines = st.selectbox("Multiplas Linhas", ["Sim", "Não", "Sem serviço de telefone"], index=None)
        InternetService = st.selectbox("Internet", ["DSL", "Fibra óptica", "Nenhum"], index=None)
        OnlineSecurity = st.selectbox("Segurança Online", ["Sim", "Não", "Sem serviço de internet"], index=None)
        OnlineBackup = st.selectbox("Backup Online", ["Sim", "Não", "Sem serviço de internet"], index=None)
        DeviceProtection = st.selectbox("Proteção de Dispositivo", ["Sim", "Não", "Sem serviço de internet"], index=None)

    with col2:
        TechSupport = st.selectbox("Suporte Técnico", ["Sim", "Não", "Sem serviço de internet"], index=None)
        StreamingTV = st.selectbox("Streaming TV", ["Sim", "Não", "Sem serviço de internet"], index=None)
        StreamingMovies = st.selectbox("Filmes", ["Sim", "Não", "Sem serviço de internet"], index=None)
        Contract = st.selectbox("Contrato", ["Mensal","Um ano","Dois anos"], index=None)
        PaperlessBilling = st.selectbox("Fatura Digital", ["Sim", "Não"], index=None)
        PaymentMethod = st.selectbox("Pagamento", [
            "Cheque Eletrônico",
            "Cheque enviado pelo correio",
            "Transferência bancária (automática)",
            "Cartão de crédito (automático)"
        ], index=None)
        MonthlyCharges = st.number_input("Custo Mensal", min_value=0.0, max_value=500.0, value=0.0)
        TotalCharges = st.number_input("Total Pago", min_value=0.0, max_value=8000.0, value=0.0)

    submit = st.form_submit_button("Identificar Cluster")

if submit:

    if MultipleLines == "Sim":
        MultipleLines = "Yes"
    elif MultipleLines == "Não":
        MultipleLines = "No"
    else:
        MultipleLines = "No phone service"

    if InternetService == "Fibra óptica":
        InternetService = "Fiber optic"
    elif InternetService == "Nenhum":
        InternetService = "No"

    if OnlineSecurity == "Sim":
        OnlineSecurity = "Yes"
    elif OnlineSecurity == "Não":
        OnlineSecurity = "No"
    else:
        OnlineSecurity = "No internet service"

    if OnlineBackup == "Sim":
        OnlineBackup = "Yes"
    elif OnlineBackup == "Não":
        OnlineBackup = "No"
    else:
        OnlineBackup = "No internet service"

    if DeviceProtection == "Sim":
        DeviceProtection = "Yes"
    elif DeviceProtection == "Não":
        DeviceProtection = "No"
    else:
        DeviceProtection = "No internet service"

    if TechSupport == "Sim":
        TechSupport = "Yes"
    elif TechSupport == "Não":
        TechSupport = "No"
    else:
        TechSupport = "No internet service"
        
    if StreamingTV == "Sim":
        StreamingTV = "Yes"
    elif StreamingTV == "Não":
        StreamingTV = "No"
    else:
        StreamingTV = "No internet service"


    if StreamingMovies == "Sim":
        StreamingMovies = "Yes"
    elif StreamingMovies == "Não":
        StreamingMovies = "No"
    else:
        StreamingMovies = "No internet service"

    if Contract == "Mensal":
        Contract = "Month-to-month" 
    elif Contract == "Um ano":
        Contract = "One year"
    else:
        Contract = "Two year"

    if PaymentMethod == "Cheque Eletrônico":
        PaymentMethod = "Electronic check"
    elif PaymentMethod == "Cheque enviado pelo correio":
        PaymentMethod = "Mailed check"
    elif PaymentMethod == "Transferência bancária (automática)":
        PaymentMethod = "Bank transfer (automatic)"

    df_user = pd.DataFrame([{
        "gender": "Male" if gender == "Masculino" else "Female",
        "SeniorCitizen": 1 if SeniorCitizen == "Sim" else 0,
        "Partner": "Yes" if Partner == "Sim" else "No",
        "Dependents": "Yes" if Dependents == "Sim" else "No",
        "tenure": tenure,
        "PhoneService": "Yes" if PhoneService == "Sim" else "No",
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": "Yes" if PaperlessBilling == "Sim" else "No",
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }])

    cluster = pipe.predict(df_user)[0]

    st.success(f"Este cliente pertence ao grupo **{cluster}**")

    st.divider()
    st.subheader("Interpretação do Segmento")

    if cluster == 0:
        st.write("""
        **Cluster 0**
        Cliente possui risco moderado de se desvincular do serviço cotratado (Risco médio).
        """)
    elif cluster == 1:
        st.write("""
        **Cluster 1**
        Cliente possui maior risco de se desvincular do serviço cotratado (Risco alto).
        """)
    elif cluster == 2:
        st.write("""
        **Cluster 2**
        Cliente possui menor risco de se desvincular do serviço cotratado (Risco baixo).    
        """)

    st.subheader("Grupos de clientes existente")
    
    df_raw = pd.read_csv("archive/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df_raw["TotalCharges"] = pd.to_numeric(df_raw["TotalCharges"], errors="coerce")
    df_raw.dropna(subset=["TotalCharges"], inplace=True)
    df_features = df_raw.drop(columns=["customerID", "Churn"])
    
    preprocess = pipe.named_steps["preprocess"]
    pca = pipe.named_steps["pca"]
    kmeans = pipe.named_steps["kmeans"]

    X = preprocess.transform(df_features)
    X_pca = pca.transform(X)
    labels = kmeans.predict(X_pca)

    plt.figure(figsize=(7,5))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, alpha=0.4)
    
    user_pca = pca.transform(preprocess.transform(df_user))
    plt.scatter(user_pca[:,0], user_pca[:,1], c="red", s=200, marker="x")
    plt.title("Clusters com PCA (cliente marcado)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    st.pyplot(plt)
    st.write("O ponto vermelho indica a qual grupo o cliente pertence.")