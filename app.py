import streamlit as st
import os
import tempfile
from datetime import date

# Importações da IA (Agora que o servidor já as possui)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Vacci-Assist | A Jornada", page_icon="💉", layout="wide")

# CSS para tornar o layout elegante e responsivo
st.markdown("""
<style>
    .main { background-color: #f4f7f9; }
    .stButton>button {
        width: 100%; border-radius: 12px; height: 3em;
        background-color: #0e4da4; color: white; font-weight: bold;
    }
    .card {
        background: white; padding: 20px; border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05); margin-bottom: 20px;
        border-left: 5px solid #0e4da4;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNÇÃO DE PROCESSAMENTO ---
def processar_pdf(uploaded_file, api_key):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name
    try:
        loader = PyPDFLoader(path)
        splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(loader.load())
        vs = FAISS.from_documents(splits, OpenAIEmbeddings(api_key=api_key))
        return vs
    finally:
        if os.path.exists(path): os.remove(path)

# --- BARRA LATERAL ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>💉 Vacci-Assist</h2>", unsafe_allow_html=True)
    st.divider()
    menu = st.radio("NAVEGAÇÃO", ["🏠 Início", "🤖 Consultoria IA", "📅 Calculadora PNI", "📖 Comportamento Intencional"])
    st.divider()
    api_key = st.text_input("OpenAI API Key", type="password")
    pdf_doc = st.file_uploader("📂 Protocolo em PDF", type="pdf")
    st.sidebar.link_button("🆘 Suporte Mentor", "https://wa.me/seu_numero_aqui")

# --- LÓGICA DAS PÁGINAS ---

if menu == "🏠 Início":
    st.title("🚀 Bem-vindo à sua Jornada de Elite")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class="card">
            <h3>Domine a Vacinação com Intencionalidade</h3>
            <p>Este hub foi criado para que você, mentorado, tenha segurança técnica e 
            estratégica em cada atendimento.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.info("💡 **Dica do Dia:** Networking é sobre servir antes de pedir.")

elif menu == "🤖 Consultoria IA":
    st.header("🤖 Consultor Técnico")
    if "messages" not in st.session_state: st.session_state.messages = []
    
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if prompt := st.chat_input("Perqunte sobre doses, intervalos ou abordagens..."):
        if not api_key: st.error("Insira a API Key na lateral!"); st.stop()
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            llm = ChatOpenAI(model="gpt-4o", temperature=0.3, api_key=api_key)
            system_prompt = "Você é o mentor Vacci-Assist. Use um tom encorajador e técnico."
            
            if pdf_doc:
                vs = processar_pdf(pdf_doc, api_key)
                chain = create_retrieval_chain(vs.as_retriever(), create_stuff_documents_chain(llm, ChatPromptTemplate.from_template(system_prompt + "\n\n{context}\n\n{input}")))
                res = chain.invoke({"input": prompt})["answer"]
            else:
                res = llm.invoke([("system", system_prompt), ("human", prompt)]).content
            
            st.markdown(res)
            st.session_state.messages.append({"role": "assistant", "content": res})

elif menu == "📅 Calculadora PNI":
    st.header("📅 Calculadora Rápida")
    nasc = st.date_input("Data de Nascimento")
    if nasc:
        hoje = date.today()
        meses = (hoje.year - nasc.year) * 12 + (hoje.month - nasc.month)
        st.metric("Idade do Paciente", f"{meses} Meses")

elif menu == "📖 Comportamento Intencional":
    st.header("📖 A Filosofia do Sucesso")
    st.markdown("""
    <div class="card">
        <h3>Princípio da Intencionalidade</h3>
        <p>Não é sobre a agulha, é sobre a conexão. Use cada atendimento para construir sua autoridade.</p>
    </div>
    """, unsafe_allow_html=True)
