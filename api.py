from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from Chatbot_memoria import app as chatbot

app = FastAPI() #Creamos la API

@app.get("/") #Ruta para ver el estado de la API
def inicio():
    return {"mensaje": "API de VelozNet funcionando"}

class PeticionChat(BaseModel): #Definimos el formato de los datos que entran 
    cliente:str
    pregunta:str

class RespuestaChat(BaseModel):#Definimos el formato de los datos que salen
    respuesta:str    

@app.post("/chat") #Recibe la pregunta de un cliente y devuelve la respuesta del chatbot
def chat(peticion:PeticionChat)->RespuestaChat:
    config={ 
        "configurable":{"thread_id":peticion.cliente}
        #Cada cliente tiene su propia conversación
    }
    
    resultado=chatbot.invoke( #llamamos al chatbot y le mandamos los datos que pide
        {
            "pregunta_original":peticion.pregunta,
            "mensajes":[HumanMessage(peticion.pregunta)]
        },config
    )
    
    return RespuestaChat(respuesta=resultado["respuesta"]) #Y delvolvemos la respuesta 




