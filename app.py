import streamlit as st
import os
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import requests

# Verificar si la clave de API de OpenAI está configurada en Streamlit Secrets
if "OPENAI_API_KEY" not in st.secrets:
    st.error("La clave de API de OpenAI no está configurada. Por favor, configura la clave de API de OpenAI en Streamlit Secrets.")
    st.stop()

openai_api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key # Para que Langchain la detecte

# =============================================================================
# 2. Definir las herramientas (tools)
# =============================================================================
@tool
def get_weather(city: str) -> str:
    """
    Obtiene información del clima para una ciudad usando OpenWeather.
    """
    try:
        WEATHER_API_KEY = "c40fdb036b5f15b452329740da94389d"  # <-- Sustituye con tu API Key real
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric&lang=es"
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200:
            temp = data['main']['temp']
            description = data['weather'][0]['description']
            return f"El clima en {city}: {description}, temperatura: {temp}°C"
        else:
            return f"Error al obtener el clima: {data.get('message', 'Ciudad no encontrada')}"
    except Exception as e:
        return f"Error al procesar la solicitud: {str(e)}"

@tool
def multiply_numbers(input: str) -> str:
    """
    Multiplica dos números dados en una cadena separados por coma.
    Ejemplo: "2, 3" devolverá "6".
    """
    try:
        partes = input.split(',')
        if len(partes) != 2:
            return "Error: Se deben proporcionar exactamente dos números separados por una coma."
        a = int(partes[0].strip())
        b = int(partes[1].strip())
        return str(a * b)
    except Exception as e:
        return f"Error procesando la entrada: {str(e)}"

@tool
def add_numbers(input: str) -> str:
    """
    Suma dos números dados en una cadena separados por coma.
    Ejemplo: "2, 3" devolverá "5".
    """
    try:
        partes = input.split(',')
        if len(partes) != 2:
            return "Error: Se deben proporcionar exactamente dos números separados por una coma."
        a = int(partes[0].strip())
        b = int(partes[1].strip())
        return str(a + b)
    except Exception as e:
        return f"Error procesando la entrada: {str(e)}"

import json

@tool("search_product", return_direct=True)
def search_product(query: str) -> dict:
    """
    Consume el endpoint de la API para buscar productos con la palabra clave proporcionada.
    """
    url = f"https://www.premiumbaby.com.ar/api/productos?busqueda={query}"
    headers = {"NC-TOKEN": "3f2b7e91-8a3e-4f7c-902f-28d94c5f6c88"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "No se pudo obtener la información del producto."}

# =============================================================================
# 3. Configurar la base de datos vectorial a partir de un PDF (FUNCIONES ADAPTADAS PARA STREAMLIT)
# =============================================================================

@st.cache_resource
def load_data(pdf_path): # Changed to accept pdf_path directly
    """Carga datos desde un documento PDF."""
    try:
        loader = PyPDFLoader(pdf_path) # Pass the file path directly
        documentos = loader.load()
        return documentos
    except Exception as e:
        st.error(f"Error al cargar el PDF: {e}")
        return None

@st.cache_resource
def crear_vector_store(_documentos):
    """Crea un vector store FAISS a partir de documentos."""
    if _documentos is None:
        return None
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0) # Using CharacterTextSplitter as in the example
    split_docs = text_splitter.split_documents(_documentos) # Use _documentos here
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_documents(split_docs, embeddings)
    return vector_store

# =============================================================================
# 4. Definir la herramienta para consultar la base de datos vectorial (ADAPTADA PARA USAR VECTOR DB CREADA)
# =============================================================================
db = None # Placeholder for vector database, will be initialized later

@tool
def query_vector_db(query: str) -> str:
    """
    Consulta la base de datos vectorial con una consulta de texto.
    Devuelve los fragmentos relevantes del documento.
    """
    global db # Access the global db variable
    if db is None:
        return "Error: La base de datos vectorial no está inicializada."
    try:
        results = db.similarity_search(query, k=3)
        if results:
            # Focus on returning a concise summary from the top result
            response = results[0].page_content if results else "No se encontraron resultados relevantes."
            return response
        else:
            return "No se encontraron resultados en la base de datos vectorial."
    except Exception as e:
        return f"Error consultando la base de datos vectorial: {str(e)}"

# =============================================================================
# 5. Lista de herramientas disponibles para el agente
# =============================================================================
tools = [get_weather, add_numbers, multiply_numbers, query_vector_db, search_product]

# =============================================================================
# 6. Definir el prompt del agente (modificado) -  REUTILIZANDO EL PROMPT DEL EJEMPLO - **PROMPT REFINADO PARA CONCISENESS**
# =============================================================================
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un asistente especializado. Tu función principal es responder preguntas concisas y directas relacionadas con el documento PDF que se ha cargado.  Para responder preguntas sobre el contenido del PDF, debes usar la herramienta 'query_vector_db' y proporcionar respuestas breves y enfocadas.  Además, puedes usar las siguientes herramientas para tareas específicas: 'get_weather' para clima, 'add_numbers' y 'multiply_numbers' para cálculos, y 'search_product' para buscar productos. Si la pregunta no es sobre el PDF o estas herramientas, responde: 'Lo siento, solo preguntas sobre el PDF o herramientas.'"
        ),
        ("placeholder", "{chat_history}"),  # Historial de conversación
        ("human", "{input}"),          # Entrada del usuario
        ("placeholder", "{agent_scratchpad}"),  # Acciones y observaciones del agente
    ]
)


# =============================================================================
# 7. Configurar el modelo de lenguaje (LLM) y crear el agente - REUTILIZANDO DEL EJEMPLO
# =============================================================================
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False) # Verbose False for cleaner output in Streamlit


# =============================================================================
# 8.  INTERFAZ DE USUARIO DE STREAMLIT (ADAPTADA PARA AGENTE - CARGA DE PDF DESDE CODIGO)
# =============================================================================
def main():
    global db # To modify the global db variable

    st.title("Chatbot Agente RAG con PDF y Herramientas")

    # --- CARGA DE PDF DESDE CODIGO ---
    pdf_path = "flashia.pdf"  # **RUTA HARDCODEADA AQUÍ**
    documentos = load_data(pdf_path)
    if documentos:
        db = crear_vector_store(documentos) # Initialize the global db here
    else:
        st.error("Error al cargar el PDF desde la ruta especificada en el código.")
        return # Exit main function if PDF loading fails

    if db: # Proceed only if vector database is created successfully

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Escribe tu pregunta:"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                try:
                    with st.spinner("Procesando..."):
                        response_agent = agent_executor.invoke({"input": prompt, "chat_history": st.session_state.messages}) # Use agent_executor
                        full_response = response_agent['output'] # Extract the output from agent response
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    st.error(f"Error: {e}")
                    message_placeholder.markdown("Lo siento, hubo un error al procesar tu pregunta.")
    else:
        st.error("Error al crear el vector store.")


if __name__ == "__main__":
    main()
