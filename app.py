import streamlit as st # Isso deve ser a PRIMEIRA linha agora
import os
import subprocess
import sys
import tempfile
from datetime import date

# --- BLOCO DE SEGURANÇA: FORÇA A INSTALAÇÃO SE O SERVIDOR FALHAR ---
try:
    from langchain.chains import create_retrieval_chain
except ImportError:
    # Se der erro, ele instala e o st.rerun() agora vai funcionar porque o 'st' já existe
    subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain", "langchain-openai", "langchain-community", "faiss-cpu", "pypdf", "fpdf", "tiktoken"])
    st.rerun()

# Importações que dependem da instalação acima
from fpdf import FPDF
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Vacci-Assist | Hub de Imunização",
    page_icon="💉",
    layout="wide"
)

# Estilos CSS para deixar visualmente profissional
st.markdown("""
<style>
    .stButton>button {border-radius: 20px; font-weight: bold; background-color: #0e4da4; color: white;}
    h1 {color: #0e4da4;}
    .stSidebar {background-color: #f0f2f6;}
</style>
""", unsafe_allow_html=True)

# --- FUNÇÕES DE APOIO ---

def processar_pdf(uploaded_file, api_key):
    if not uploaded_file: return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings(api_key=api_key)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        return vectorstore
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

def get_response_chain(vectorstore, api_key):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1, api_key=api_key)
    if vectorstore:
        retriever = vectorstore.as_retriever()
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Você é o Vacci-Assist. Use APENAS o contexto abaixo:\n\n{context}"),
            ("human", "{input}")
        ])
        chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
        return chain, True
    else:
        prompt = ChatPromptTemplate.from_messages([("system", "Você é um assistente de vacinas."), ("human", "{input}")])
        return prompt | llm, False

def mostrar_calculadora():
    st.header("📅 Calculadora de Vacinação (PNI)")
    data_nasc = st.date_input("Data de Nascimento do Paciente", min_value=date(1920, 1, 1))
    if data_nasc:
        hoje = date.today()
        idade_meses = (hoje.year - data_nasc.year) * 12 + (hoje.month - data_nasc.month)
        st.metric("Idade em Meses", f"{idade_meses} meses")
        vacinas = ["BCG", "Hepatite B"] if idade_meses < 2 else ["Consultar Calendário Completo"]
        for v in vacinas: st.success(f"💉 {v}")

# --- BARRA LATERAL E NAVEGAÇÃO ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966334.png", width=80)
    st.title("Vacci-Assist")
    menu = st.radio("Navegação", ["🏠 Início", "🤖 Assistente IA", "📅 Calculadora"])
    st.divider()
    api_key = st.text_input("Sua OpenAI API Key", type="password")
    uploaded_file = st.file_uploader("📂 Base de Conhecimento (PDF)", type="pdf")

# --- LÓGICA DAS ABAS ---
if menu == "🏠 Início":
    st.title("Domine a Vacinação na Prática")
    st.write("Bem-vindo ao Vacci-Assist. Use o menu ao lado para começar.")
elif menu == "🤖 Assistente IA":
    st.subheader("Consultor Virtual")
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    user_input = st.chat_input("Pergunte algo...")
    if user_input:
        if not api_key: st.error("Falta a API Key!")
        else:
            with st.spinner("Pensando..."):
                vs = processar_pdf(uploaded_file, api_key) if uploaded_file else None
                chain, is_rag = get_response_chain(vs, api_key)
                res = chain.invoke({"input": user_input})
                ans = res["answer"] if is_rag else res.content
                st.write(ans)
elif menu == "📅 Calculadora":
    mostrar_calculadora()
