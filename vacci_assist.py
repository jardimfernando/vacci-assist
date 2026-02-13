import streamlit as st
import tempfile
import os
from datetime import date
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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS Personalizados
st.markdown("""
<style>
    .stButton>button {border-radius: 20px; font-weight: bold;}
    .reportview-container {background: #fdfdfd;}
    h1 {color: #0e4da4;}
    .stSidebar {background-color: #f0f2f6;}
</style>
""", unsafe_allow_html=True)

# --- FUNÇÕES DE BACKEND (Lógica do Sistema) ---

def processar_pdf(uploaded_file, api_key):
    """Lê o PDF e cria o banco de dados da IA."""
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
    except Exception as e:
        st.error(f"Erro ao ler PDF: {e}")
        return None
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

def get_response_chain(vectorstore, api_key):
    """Cria a inteligência da resposta (RAG ou Geral)."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1, api_key=api_key)
    
    if vectorstore:
        retriever = vectorstore.as_retriever()
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Você é o Vacci-Assist. Responda APENAS com base no contexto abaixo:\n\n{context}"),
            ("human", "{input}")
        ])
        chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
        return chain, True
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Você é um assistente de vacinação. Responda com base em conhecimentos gerais de saúde."),
            ("human", "{input}")
        ])
        chain = prompt | llm
        return chain, False

def gerar_pdf_conversa(historico):
    """Gera um arquivo PDF com o resumo do chat."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Resumo Vacci-Assist", ln=True, align='C')
    pdf.ln(10)
    
    for msg in historico:
        role = "Paciente/Usuário" if isinstance(msg, HumanMessage) else "Vacci-Assist"
        texto = f"{role}: {msg.content}"
        texto = texto.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, txt=texto)
        pdf.ln(2)
        
    return pdf.output(dest='S').encode('latin-1')

def mostrar_calculadora():
    """Exibe a calculadora de vacinas."""
    st.header("📅 Calculadora de Vacinas (PNI)")
    st.info("Informe a data de nascimento para ver as vacinas indicadas.")
    
    data_nasc = st.date_input("Data de Nascimento", min_value=date(1920, 1, 1))
    
    if data_nasc:
        hoje = date.today()
        meses = (hoje.year - data_nasc.year) * 12 + (hoje.month - data_nasc.month)
        
        col1, col2 = st.columns(2)
        col1.metric("Idade Atual", f"{meses} meses")
        
        st.subheader(f"Indicado para {meses} meses:")
        
        # Lógica simplificada (pode ser expandida)
        vacinas = []
        if months := meses:
            if months < 2: vacinas = ["BCG", "Hepatite B"]
            elif 2 <= months < 4: vacinas = ["Penta (1ª)", "VIP (1ª)", "Rotavírus (1ª)", "Pneumo-10 (1ª)"]
            elif 4 <= months < 6: vacinas = ["Penta (2ª)", "VIP (2ª)", "Rotavírus (2ª)", "Pneumo-10 (2ª)"]
            elif 6 <= months < 9: vacinas = ["Penta (3ª)", "VIP (3ª)", "Influenza"]
            elif months == 9: vacinas = ["Febre Amarela"]
            elif 12 <= months < 15: vacinas = ["Tríplice Viral", "Meningo C", "Pneumo-10 (Reforço)"]
            elif 15 <= months < 48: vacinas = ["DTP", "VOP", "Tetra Viral", "Hepatite A"]
            else: vacinas = ["Verificar Reforços e Campanhas Anuais"]
        
        for v in vacinas:
            st.success(f"💉 {v}")

# --- MENU LATERAL E NAVEGAÇÃO ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966334.png", width=80)
    st.title("Vacci-Assist")
    st.caption("Tecnologia & Mentoria")
    
    menu = st.radio(
        "Navegue por aqui:", 
        ["🏠 Início & Mentoria", "🤖 Assistente IA", "📅 Calculadora Vacinal", "👤 Sobre o Autor"]
    )
    
    st.divider()
    
    # Configurações globais (aparecem em todas as telas)
    if menu == "🤖 Assistente IA":
        st.subheader("⚙️ Configuração IA")
        # Tenta pegar a chave dos Segredos do Streamlit, senão pede na tela
        if "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
            st.success("Chave de API detectada!")
        else:
            api_key = st.text_input("Sua OpenAI API Key", type="password")
            
        uploaded_file = st.file_uploader("Carregar Protocolo (PDF)", type="pdf")
        
        if st.button("Limpar Conversa"):
            st.session_state.chat_history = []
            st.rerun()

# --- LÓGICA DAS PÁGINAS ---

if menu == "🏠 Início & Mentoria":
    st.title("Domine a Vacinação na Prática")
    st.markdown("""
    ### Bem-vindo ao hub oficial Vacci-Assist.
    
    Aqui unimos tecnologia de ponta com o método **Comportamento Intencional** para transformar sua farmácia.
    
    #### 🚀 Nossas Soluções:
    1.  **Assistente IA:** Tire dúvidas técnicas instantâneas baseadas em protocolos oficiais.
    2.  **Calculadora PNI:** Agilidade no balcão da farmácia.
    3.  **Educação Continuada:** Mentoria e materiais para vacinadores.
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("📘 **E-book: Comportamento Intencional**\n\nDescubra como sua postura muda seus resultados e fideliza pacientes.")
    with col2:
        st.success("🎓 **Mentoria Vacinação 360**\n\nTorne-se referência em serviços de imunização na sua cidade.")
        st.link_button("Falar com Mentor no WhatsApp", "https://wa.me/5500000000000") 
        # Substitua o número acima pelo seu WhatsApp real

elif menu == "🤖 Assistente IA":
    st.subheader("Consultor Virtual de Imunização")
    
    # Inicializa histórico
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Olá! Sou o Vacci-Assist. Posso ajudar com dúvidas técnicas ou analisar bulas.")]
    
    # Processa PDF
    if 'uploaded_file' in locals() and uploaded_file and api_key:
        if "vectorstore" not in st.session_state: st.session_state.vectorstore = None
        if "last_file" not in st.session_state or st.session_state.last_file != uploaded_file.name:
            with st.spinner("Lendo documento oficial..."):
                st.session_state.vectorstore = processar_pdf(uploaded_file, api_key)
                st.session_state.last_file = uploaded_file.name
            st.toast("Documento assimilado!", icon="✅")

    # Exibe Chat
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        avatar = "👤" if role == "user" else "💉"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg.content)
            
    # Input do usuário
    user_input = st.chat_input("Digite sua dúvida sobre vacinas...")
    if user_input:
        if not api_key:
            st.warning("⚠️ Insira a API Key na barra lateral.")
            st.stop()
            
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
            
        with st.chat_message("assistant", avatar="💉"):
            with st.spinner("Consultando base de conhecimento..."):
                vs = st.session_state.vectorstore if "vectorstore" in st.session_state else None
                chain, is_rag = get_response_chain(vs, api_key)
                
                res = chain.invoke({"input": user_input})
                ans = res["answer"] if is_rag else res.content
                fonte = "\n\n*Fonte: Documento Oficial Carregado*" if is_rag else "\n\n*Fonte: IA (Base Geral)*"
                
                st.markdown(ans + fonte)
                st.session_state.chat_history.append(AIMessage(content=ans + fonte))
    
    # Botão de Download do PDF da conversa
    if len(st.session_state.chat_history) > 1:
        pdf_bytes = gerar_pdf_conversa(st.session_state.chat_history)
        st.download_button("📥 Baixar Resumo da Consulta", pdf_bytes, "resumo_vacinas.pdf", "application/pdf")

elif menu == "📅 Calculadora Vacinal":
    mostrar_calculadora()

elif menu == "👤 Sobre o Autor":
    st.header("Sobre o Criador")
    st.write("""
    **[Seu Nome Aqui]** é Farmacêutico especialista em Vacinação e Empreendedorismo.
    
    Criador do método **A Jornada** e autor do livro sobre Comportamento Intencional,
    tem como missão capacitar profissionais de saúde para oferecerem serviços de excelência.
    """)
    st.info("Entre em contato: contato@vacci-assist.com.br")