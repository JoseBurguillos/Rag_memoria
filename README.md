# Chatbot de atención al cliente con memoria (RAG + LangGraph)

Asistente de atención al cliente para **VelozNet**, una empresa ficticia de internet por fibra. Responde **solo** con la información de los documentos internos de la empresa (tarifas, instalación, averías, facturación y bajas), dice claramente cuándo no tiene la respuesta en lugar de inventarla y **recuerda la conversación**, así que entiende preguntas encadenadas como "¿Y tiene permanencia?" o "¿Y si está en rojo?".

Se puede usar de tres formas: desde la terminal, como **API REST** (FastAPI) o desde una **interfaz web de chat** (Streamlit). Todo funciona en local con Ollama, sin APIs de pago.

<!-- Para añadir una captura: guarda la imagen en la carpeta del proyecto (por ejemplo, captura.png) y quita estas marcas de comentario de la línea de abajo -->
<!-- ![Interfaz del chatbot](captura.png) -->

## Arquitectura

```
Interfaz web (Streamlit)  ──POST /chat──►  API (FastAPI)  ──invoke──►  Graph de LangGraph
     localhost:8501                        127.0.0.1:8000              (Chatbot_memoria.py)
                                                                              │
                                                              Chroma + qwen2.5:7b + memoria
```

Cada pieza tiene una sola responsabilidad: la interfaz solo muestra el chat y habla con la API; la API recibe las peticiones, las valida y llama al graph; el graph es el que razona y responde. Si se cambia la interfaz (por ejemplo, por una web en React), la API y el graph no cambian.

## Ejemplo

```
Cliente: ¿Cuánto cuesta la Fibra 600?
Bot:     La Fibra 600 cuesta 32 euros al mes.
Cliente: ¿Y tiene permanencia?
Bot:     Sí, la Fibra 600 tiene una permanencia de 12 meses.
Cliente: ¿Y si me doy de baja antes?
Bot:     Si te das de baja antes de los 12 meses, deberás pagar una penalización
         de 10 euros por cada mes que falte para cumplir la permanencia.
Cliente: ¿Tenéis televisión?
Bot:     Lo siento, no ofrecemos televisión. Nuestro servicio se centra
         exclusivamente en internet por fibra y telefonía fija.
```

## Cómo funciona

```
START → rewrite_question → retrieve → grade_document ──decidir──┬─→ generate  → END
                                                                └─→ no_answer → END
```

| Nodo | Qué hace |
|---|---|
| `rewrite_question` | Reescribe la pregunta del cliente usando la conversación anterior, para que se entienda sola y se pueda buscar mejor |
| `retrieve` | Busca en Chroma los 3 fragmentos más parecidos a la pregunta reescrita |
| `grade_document` | Un LLM juez, con salida estructurada (`si` / `no`), se queda solo con los fragmentos que sirven para responder |
| `decidir` | Si queda algún fragmento relevante, va a `generate`; si no, a `no_answer` |
| `generate` | Redacta la respuesta como agente de atención al cliente, usando solo los fragmentos aprobados |
| `no_answer` | Devuelve una frase fija sin llamar al LLM, así es imposible que invente nada |

## Memoria

La memoria se consigue con tres piezas:

- **Checkpointer (`MemorySaver`)**: guarda el State del graph al terminar cada pregunta y lo recupera al empezar la siguiente. Sin él, cada pregunta empezaría de cero.
- **`thread_id`**: identifica la conversación. Cada cliente tendría el suyo, para que sus historiales no se mezclen.
- **Historial `mensajes` con `add_messages`**: por defecto, LangGraph sustituye el valor de cada campo. `add_messages` es un *reducer* que, en lugar de sustituir, **añade** los mensajes nuevos al final. Así el historial va creciendo: la pregunta del cliente entra como `HumanMessage` y la respuesta del bot como `AIMessage`.

`rewrite_question` lee ese historial (sin el último mensaje, que es la pregunta actual), lo convierte en texto ("Cliente: ... / Asistente: ...") y se lo pasa al modelo junto con la pregunta nueva. Así, después de listar las tarifas, "¿Y la más barata tiene permanencia?" se convierte en **"¿La Fibra 300 tiene permanencia?"**, que ya se puede buscar en los documentos.

## Tecnologías

- **LangGraph**: flujo con nodos, arista condicional y memoria (checkpointer)
- **LangChain**: mensajes, documentos y salida estructurada
- **ChromaDB**: base de datos vectorial
- **Ollama**: modelos en local (`qwen2.5:7b` como LLM y `embeddinggemma` para los embeddings)
- **Pydantic**: salida estructurada del juez (`Literal["si", "no"]`) y validación de los datos de la API
- **FastAPI + Uvicorn**: API REST para usar el chatbot desde otros programas
- **Streamlit**: interfaz web de chat
- **Requests**: peticiones HTTP de la interfaz a la API

## Estructura

```
├── documentos/           # Documentos internos de VelozNet (empresa ficticia)
├── indexar.py            # Trocea los documentos y los guarda en Chroma
├── Chatbot_memoria.py    # El graph (el chatbot) + chat por terminal
├── api.py                # API REST con FastAPI (POST /chat)
├── interfaz.py           # Interfaz web de chat con Streamlit
└── requirements.txt
```

## Instalación y uso

1. Instala [Ollama](https://ollama.com) y descarga los modelos:
   ```bash
   ollama pull qwen2.5:7b
   ollama pull embeddinggemma
   ```
2. Crea un entorno virtual con Python 3.13 e instala las dependencias:
   ```bash
   py -3.13 -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Indexa los documentos (solo la primera vez, o si cambian):
   ```bash
   python indexar.py
   ```
4. Usa el chatbot de una de estas tres formas.

### Opción A: en la terminal

```bash
python Chatbot_memoria.py
```

### Opción B: como API

```bash
python -m uvicorn api:app --reload
```

La API queda en `http://127.0.0.1:8000`. La documentación interactiva está en `http://127.0.0.1:8000/docs`.

| Método | Ruta | Qué hace |
|---|---|---|
| `GET` | `/` | Comprueba que la API está funcionando |
| `POST` | `/chat` | Envía una pregunta y devuelve la respuesta |

Ejemplo de petición a `POST /chat`:

```json
{
  "cliente": "ana",
  "pregunta": "¿Cuánto cuesta la Fibra 600?"
}
```

Respuesta:

```json
{
  "respuesta": "La Fibra 600 cuesta 32 euros al mes."
}
```

El campo `cliente` se usa como `thread_id`: cada cliente tiene su propia conversación y su propia memoria.

### Opción C: interfaz web

Necesita la API en marcha, así que hacen falta **dos terminales**:

```bash
# Terminal 1: la API
python -m uvicorn api:app --reload
```

```bash
# Terminal 2: la interfaz
streamlit run interfaz.py
```

La interfaz se abre en `http://localhost:8501`.

## Decisiones de diseño

- **IDs con hash (SHA-256) al indexar:** cada fragmento tiene un ID calculado a partir de su archivo de origen y su contenido. Si se ejecuta `indexar.py` varias veces, Chroma reconoce los IDs que ya existen y no duplica nada (siempre quedan 15 fragmentos). El ID incluye el contenido porque, si solo usara el nombre del archivo, todos los fragmentos de un mismo archivo tendrían el mismo ID y se sobrescribirían.
- **Un juez después de la búsqueda:** Chroma busca por *parecido*, no por *relevancia*, y siempre devuelve 3 fragmentos aunque no tengan nada que ver con la pregunta. El juez lee cada fragmento y descarta los que no responden. Usa `Literal["si", "no"]` para que el modelo no pueda contestar con texto libre.
- **`no_answer` sin LLM:** si no queda ningún fragmento relevante, el chatbot responde con una frase fija. Al no llamar al modelo, es imposible que invente un precio o una condición, que en atención al cliente sería lo más grave.
- **Un "no" claro también es una respuesta:** si un documento dice que VelozNet no ofrece algo (por ejemplo, televisión, o el pago con tarjeta), el juez lo aprueba y el chatbot lo dice claramente. Para un cliente, "no ofrecemos televisión" es mucho más útil que "no tengo información".
- **`generate` recibe la pregunta original y la reescrita:** al principio solo recibía la original. En la conversación "Fibra 600 → ¿Y si me doy de baja antes?", no sabía de qué tarifa se hablaba y mezclaba información de la Fibra 300 ("tu baja nunca tendrá penalización"). La reescrita ya resume el contexto de la conversación en una sola frase, así que se la paso a `generate` en lugar del historial completo, que crecería con cada pregunta y haría el chatbot más lento.

## Pruebas

| Tipo | Conversación | Resultado |
|---|---|---|
| Encadenada | ¿Cuánto cuesta la Fibra 600? → ¿Y tiene permanencia? → ¿Y si me doy de baja antes? | ✅ |
| Encadenada | No me va internet → Ya lo he reiniciado y sigue igual → ¿Y si está en rojo? | ✅ |
| Encadenada | ¿Cuándo me cobráis? → ¿Y si me devuelven el recibo? → ¿Y si sigo sin pagar? | ✅ |
| Memoria | ¿Qué tarifas tenéis? → ¿Y la más barata tiene permanencia? | ✅ Se reescribe como "¿La Fibra 300 tiene permanencia?" (sin memoria respondía Fibra 600, falso) |
| Trampa | ¿Tenéis televisión? | ✅ "No ofrecemos televisión" |
| Trampa | ¿Puedo pagar con tarjeta? | ✅ Solo domiciliación bancaria |
| Trampa | ¿Cuánto cuesta la de 10 Gb? | ✅ No inventa precio (`no_answer`) |
| Cambio de tema | Varios temas en la misma conversación | ✅ No mezcla temas |
| API: memoria por cliente | **ana**: ¿Cuánto cuesta la Fibra 600? → **luis**: ¿Y tiene permanencia? → **ana**: ¿Y tiene permanencia? | ✅ Luis recibe "no tengo información" (no ha hablado de ninguna tarifa); Ana recibe "Sí, la Fibra 600 tiene una permanencia de 12 meses" |
| Interfaz | QUE TARIFA ES LA MAS BARATA → y tiene permanencia? | ✅ Entiende la pregunta en mayúsculas y sin tildes, y recuerda que se habla de la Fibra 300 |

## Limitaciones y próximos pasos

- `MemorySaver` guarda en RAM: la conversación se pierde al cerrar el programa. Siguiente paso: `SqliteSaver` o `PostgresSaver`, para que sobreviva a reinicios.
- A veces el modelo añade información que no está en los documentos (por ejemplo, "puedes comprobarlo en tu área de cliente"). Siguiente paso: un nodo que compruebe que la respuesta está respaldada por los fragmentos.
- **Latencia:** cada respuesta tarda entre 7 y 20 segundos, porque cada pregunta hace 5 llamadas a un LLM de 7B que se ejecuta en local (1 de reescritura, 3 del juez y 1 de respuesta). Se podría reducir evaluando los 3 fragmentos en una sola llamada, o usando un servidor con GPU.
- La interfaz usa siempre el mismo cliente (`"user"`), así que todas las personas que la abran comparten memoria. Siguiente paso: generar un identificador único por sesión.
- Cuando una pregunta depende de un contexto que no existe (por ejemplo, "¿Y tiene permanencia?" sin haber hablado de ninguna tarifa), responde "no tengo información". Sería más útil preguntar al cliente a qué tarifa se refiere.
- Siguiente paso: empaquetarlo con Docker.

## Lo que aprendí

- **Construir un RAG completo desde cero:** indexar documentos (trocear, IDs con hash, embeddings) y consultarlos.
- **Similar no es lo mismo que relevante:** por qué hace falta un juez después de la búsqueda vectorial.
- **Salida estructurada con Pydantic** para que un LLM responda en un formato fiable.
- **Memoria en LangGraph:** checkpointer, `thread_id` y reducers (`add_messages`), y por qué el checkpointer solo no basta.
- **Las decisiones de diseño dependen del sistema:** pasar a `generate` solo la pregunta original era correcto sin memoria, pero dejó de serlo al añadirla.
- **Exponer un modelo como servicio:** una API con FastAPI que valida los datos con Pydantic, y una interfaz con Streamlit que la consume. Separar la lógica (el graph) de la forma de acceder a ella (terminal, API o web) permite cambiar una sin tocar las otras.
