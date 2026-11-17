from pathlib import Path
import hashlib

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


rutas=Path("documentos").glob("*.txt") #rutas = Path("documentos").glob("*.txt")  # Busca todos los archivos .txt de la carpeta documentos

documentos=[] #para guardar los documentos creados

for ruta in rutas:
    contenido=ruta.read_text(encoding="utf-8") #leemos el contenido de los archivos

    documento=Document( #Creamos un objeto document
        page_content=contenido,
        metadata={
            "source":str(ruta),
            "nombre":ruta.name
        }
    )

    documentos.append(documento) #lo agregamos a la lista de docuemntos

divisor=RecursiveCharacterTextSplitter( #la manera en la que vamos a dividir los documentos
    chunk_size=500,
    chunk_overlap=100
)

fragmentos=divisor.split_documents(documentos) #fragmentamos los documentos

#print("Documentos:", len(documentos))
#print("Fragmentos:", len(fragmentos))

#print(fragmentos[0].metadata)
#print(fragmentos[0].page_content)

identificadores=[]

for fragmento in fragmentos:
    texto=f"{fragmento.metadata["source"]}{fragmento.page_content}" #unimos la ruta y el contenid de cada 1, asi tneemos un id distinto cada vez
    textobyte=texto.encode("utf-8") #Pasamos el texto a bytes
    generador_hash=hashlib.sha256(textobyte)#Lo hasheamos para luego pasarlo a id
    identificador=generador_hash.hexdigest()#porpiedad para sacar el id del hash

    identificadores.append(identificador) #lo guardamos en la lista

print("IDs:", len(identificadores))
print("IDs únicos:", len(set(identificadores)))

embeddings = OllamaEmbeddings( #convierte el texto en vectores
    model="embeddinggemma"
)

base_vectorial=Chroma( #Para guardar lo fragmentos con sus IDS
    collection_name="documentos",
    embedding_function=embeddings,
    persist_directory="./base_chroma"    
)

base_vectorial.add_documents( #añadimos cada fragmento y su id
    documents=fragmentos,
    ids=identificadores
)

datos= base_vectorial.get()#Los datos amacenados ahí
print("Guardados en Chroma:", 
      len(datos["ids"]))














