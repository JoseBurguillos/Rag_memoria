from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma #Base de datos vectorial
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document
from typing import TypedDict,Literal, Annotated
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

modelo=ChatOllama(
    model="qwen2.5:7b",
    temperature=0
)

embeddings = OllamaEmbeddings( #convierte el texto en vectores
    model="embeddinggemma"
)

base_vectorial=Chroma(
    collection_name="documentos",
    embedding_function=embeddings,
    persist_directory="./base_chroma"    
)

class EstadoRAG(TypedDict): #Clase para todo el rag, TypedDict caja con etiquetas
    pregunta:str
    pregunta_original:str
    documentos:list[Document]
    respuesta:str
    mensajes:Annotated[list, add_messages]  # Historial: add_messages AÑADE los mensajes en vez de sustituirlos

class EvaluacionDocumento(BaseModel): #Para poder validar el documento, BaseModel es como si fuera un formulario con un revisor
    valido:Literal["si","no"]=Field(
        description="Escribe 'si' si el documento contiene la respuesta a la pregunta o una parte clara de ella. Si la pregunta es general sobre un tema, escribe 'si' cuando el documento dé información útil sobre ese tema. Escribe 'no' si el documento no responde a lo que se pregunta, aunque trate de un tema parecido."
    )

prompt_rewrite="""Eres un asistente que reformula las preguntas de los clientes de VelozNet, una empresa de internet por fibra, para buscarlas mejor en los documentos de atención al cliente.

                Reglas:
                    - Reescribe la pregunta para que sea clara y esté bien formulada.
                    - No añadas temas, conceptos ni detalles que no aparezcan en la pregunta ni en la conversación anterior.                    
                    - Si la pregunta es vaga o general, mantenla general: solo hazla más clara, no elijas un aspecto concreto.
                    - No cambies el significado de la pregunta.
                    - Responde ÚNICAMENTE con la pregunta reescrita, sin explicaciones, sin comillas y sin frases como "Aquí tienes".
                    - Si la nueva pregunta hace referencia a algo de la conversación anterior (por ejemplo "la más barata", "eso", "¿y tiene permanencia?"), complétala con la información de la conversación para que se entienda sola.
                    - Si la nueva pregunta no tiene relación con la conversación anterior, reescríbela sin tenerla en cuenta.

            Ejemplos:
                - "eso de la factura" → ¿Qué información hay sobre las facturas?
                - "cuando viene el tecnico" → ¿Cuándo viene el técnico?
                - "el wifi va fatal" → ¿Qué puedo hacer si el wifi funciona mal?
                - "me quiero ir de veloznet" → ¿Cómo puedo darme de baja de VelozNet?"""


def rewrite_question(state:EstadoRAG): # Reescribe la pregunta del cliente usando la conversación anterior, para que se entienda sola
    pregunta_original=state["pregunta_original"]
    mensaje=state["mensajes"]
    
    historial=mensaje[:-1] #Tenemos todo el historial menos la ultima pregunta que se acaba de añadir y vamos a responder
    
    texto_historial="" #Lo metemos todo en un texto para pasarselo a ollama tambien el contexto de la conversacion
    
    for m in historial:
        if isinstance(m,HumanMessage):
            texto_historial += "Cliente: " + m.content + "\n"
        else:
            texto_historial += "Asistente: " + m.content + "\n"
    
    

    mensajes=[
        SystemMessage(
            prompt_rewrite
        ),
        HumanMessage(
            f"Conversación anterior:\n{texto_historial}\nNueva pregunta: {pregunta_original}"
        )
    ]

    pregunta_nueva=modelo.invoke(
        mensajes
    )
    
    return{
        "pregunta":pregunta_nueva.content
    }


def retrieve(state:EstadoRAG): # Busca en Chroma los 3 fragmentos más parecidos a la pregunta reescrita
    pregunta=state["pregunta"]
    
    posibles_respuestas=base_vectorial.similarity_search(
        pregunta,
        k=3
    )
    
    return{
        "documentos":posibles_respuestas
    }
    
prompt_juez="""
Eres un evaluador estricto del servicio de atención al cliente de VelozNet, una empresa de internet por fibra. Tu única tarea es decidir si un documento sirve para responder a la pregunta de un cliente.

Reglas:
- Responde 'si' si el documento contiene la respuesta a la pregunta o una parte clara de ella.
- Si la pregunta es general sobre un tema, responde 'si' cuando el documento dé información útil sobre ese tema.
- Si el documento dice claramente que VelozNet no ofrece o no permite lo que pregunta el cliente, responde 'si': eso también es una respuesta.
- Responde 'no' si el documento solo trata de un tema parecido pero no responde a lo que se pregunta.
- No respondas a la pregunta del cliente. Solo evalúa el documento.

Ejemplos:
- Pregunta: "¿Cuánto tarda la instalación?" / Documento: "El plazo habitual de instalación es de 3 a 7 días laborables desde la confirmación del pedido." → si
- Pregunta: "¿Cuánto tarda la instalación?" / Documento: "Para contratar VelozNet, el cliente necesitará su DNI y una cuenta bancaria." → no (es del proceso de alta, pero no dice el plazo)
- Pregunta: "¿Dónde descargo mis facturas?" / Documento: "El servicio técnico está disponible las 24 horas del día." → no (otro tema)
- Pregunta: "lo del cambio de titular" / Documento: "Para cambiar el titular del contrato, ambas personas deben firmar una solicitud." → si (pregunta general sobre ese tema)
"""
juez=modelo.with_structured_output(EvaluacionDocumento)

def grade_document(state:EstadoRAG): #Un LLM juez se queda solo con los fragmentos que sirven para responder a la pregunta
    pregunta=state["pregunta"]
    pregunta_original=state["pregunta_original"]
    documentos=state["documentos"]
    
    relevantes=[]
    
    for documento in documentos:
        evaluacion=juez.invoke([
            SystemMessage(prompt_juez),
            HumanMessage(
                f"Documento:{documento.page_content}\nPregunta_reescrita:{pregunta}\nPregunta:{pregunta_original}"
            )
        ])
        
        if evaluacion.valido=="si":
            relevantes.append(documento)
            
    return{
        "documentos":relevantes
    }
    
    
def no_answer(state:EstadoRAG): # Si no queda ningún fragmento relevante, responde con una frase fija (sin LLM, así no inventa nada)
    frase="Lo siento, no tengo información sobre eso en este momento. Puedes contactar con nuestro equipo de atención al cliente por teléfono o desde el chat de tu área de cliente, y te ayudaremos."
    return{
        "respuesta":frase,
        "mensajes":[AIMessage(frase)]
    }
    
    
def decidir(state:EstadoRAG): #Esto depende de los documentos que tengamos ejecuta un metodo u otro
    documentos=state["documentos"]
    
    if len(documentos)==0:
        return "no_answer"
    else:
        return "generate"
    
prompt_generate="""Eres el asistente de atención al cliente de VelozNet, una empresa de internet por fibra. Respondes a los clientes de forma amable, clara y breve.

Reglas:
- Responde únicamente con la información incluida en el contexto.
- No inventes precios, plazos, condiciones ni servicios que no aparezcan en el contexto.
- Si la pregunta es general sobre un tema, resume la información del contexto sobre ese tema.
- Si el contexto dice que VelozNet no ofrece o no permite lo que pregunta el cliente, díselo claramente.
- Si el contexto no contiene la respuesta, responde exactamente: "Lo siento, no tengo información sobre eso en este momento. Puedes contactar con nuestro equipo de atención al cliente por teléfono o desde el chat de tu área de cliente, y te ayudaremos."
- Habla de tú al cliente y responde siempre en español."""

def generate(state:EstadoRAG): #Generamos la respuesta finalmente
    documentos=state["documentos"]
    pregunta_original=state["pregunta_original"]
    pregunta=state["pregunta"]
    
    contexto=""
    for documento in documentos:
        contexto += documento.page_content + "\n\n"
    
    resultado=modelo.invoke([
        SystemMessage(prompt_generate),
        HumanMessage(f"Contexto:\n{contexto}\nPregunta del cliente: {pregunta_original}\nPregunta completa: {pregunta}"
        )
    ])
    
    return{
        "respuesta":resultado.content,
        "mensajes":[AIMessage(resultado.content)]
    }


grafo=StateGraph(EstadoRAG) #Creamos el Grapho con los parametros de abajo

grafo.add_node("rewrite_question", rewrite_question)
grafo.add_node("retrieve",retrieve)
grafo.add_node("grade_document",grade_document)
grafo.add_node("no_answer",no_answer)
grafo.add_node("generate", generate)

grafo.add_edge(START,"rewrite_question")
grafo.add_edge("rewrite_question","retrieve")
grafo.add_edge("retrieve","grade_document")
grafo.add_conditional_edges(
    "grade_document",
    decidir
)
grafo.add_edge("no_answer",END)
grafo.add_edge("generate",END)

memoria=MemorySaver() # Guarda el State entre preguntas (en RAM: se pierde al cerrar el programa)

app=grafo.compile(checkpointer=memoria) #Añadimos el checkpointer en el grafo

config={"configurable":{
    "thread_id":"cliente_1"
}}

while True:
    
    pregunta=input("Asistente de VelozNet. Escribe ‘salir’ para terminar ")
    
    if pregunta.lower()=="salir":
        break
    
    resultado=app.invoke(
        {
            "pregunta_original":pregunta,
            "mensajes":[HumanMessage(pregunta)]
        },
        config # El thread_id identifica la conversación: así el checkpointer sabe qué historial cargar
    )
    
    print(resultado["respuesta"])
    