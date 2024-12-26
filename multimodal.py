# MultiModal Fusion

import os
import pyaudio
import wave
from unstructured.partition.pdf import partition_pdf
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage
from google.cloud import speech_v1p1beta1 as speech
from google.cloud.speech_v1p1beta1 import RecognitionConfig, RecognitionAudio
from dotenv import load_dotenv
import chainlit as cl
import uuid
import base64
from typing import List, Dict, Any

load_dotenv()

class MultimodalRAG:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.vectorstore = Chroma(
            collection_name="multi_modal_rag",
            embedding_function=OpenAIEmbeddings()
        )
        self.store = InMemoryStore()
        self.id_key = "doc_id"
        self.retriever = self._setup_retriever()
        self.chunks = self._process_pdf()
        
    # MultiVectorRetriever
    def _setup_retriever(self) -> MultiVectorRetriever:
        
        return MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            id_key=self.id_key,
        )

    # Partitioning the Pdf
    def _process_pdf(self) -> List:
        
        return partition_pdf(
            filename=self.pdf_path,
            infer_table_structure=True,
            strategy="hi_res",
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
            chunking_strategy="by_title",
            max_characters=10000,
            combine_text_under_n_chars=2000,
            new_after_n_chars=6000
        )

    # Separating Tables, Texts and Images
    def _separate_elements(self) -> tuple:
        
        tables = []
        texts = []
        images = []

        for chunk in self.chunks:
            if "Table" in str(type(chunk)):
                tables.append(chunk)
            elif "CompositeElement" in str(type(chunk)):
                texts.append(chunk)
                # Extract images from composite elements
                if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
                    for el in chunk.metadata.orig_elements:
                        if "Image" in str(type(el)):
                            images.append(el.metadata.image_base64)

        return tables, texts, images

    # Generating Summaries for all the elements (text, tables and images)
    async def generate_summaries(self) -> tuple:
        
        groq_model = ChatGroq(temperature=0.5, model="llama-3.1-8b-instant")
        summary_prompt = ChatPromptTemplate.from_template(
            "Give a concise summary of the following content: {element}"
        )
        summarize_chain = {"element": lambda x: x} | summary_prompt | groq_model | StrOutputParser()

        tables, texts, images = self._separate_elements()
        
        text_summaries = await summarize_chain.abatch(texts)
        table_summaries = await summarize_chain.abatch([t.metadata.text_as_html for t in tables])
        
        image_prompt = ChatPromptTemplate.from_messages([
            (
                "user",
                [
                    {"type": "text", "text": "Describe the image in detail, focusing on its content and relationship to transformers architecture."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{image}"},
                    },
                ],
            )
        ])
        
        image_chain = image_prompt | ChatOpenAI(model="gpt-4-vision-preview") | StrOutputParser()
        image_summaries = await image_chain.abatch(images)

        return text_summaries, table_summaries, image_summaries, tables, texts, images

    # Load summaries and original content to vectorstore
    async def load_to_vectorstore(self, summaries: tuple):
        text_summaries, table_summaries, image_summaries, tables, texts, images = summaries
        
        doc_ids = [str(uuid.uuid4()) for _ in texts]
        summary_texts = [
            Document(page_content=summary, metadata={self.id_key: doc_ids[i]})
            for i, summary in enumerate(text_summaries)
        ]
        self.retriever.vectorstore.add_documents(summary_texts)
        self.retriever.docstore.mset(list(zip(doc_ids, texts)))

        table_ids = [str(uuid.uuid4()) for _ in tables]
        summary_tables = [
            Document(page_content=summary, metadata={self.id_key: table_ids[i]})
            for i, summary in enumerate(table_summaries)
        ]
        self.retriever.vectorstore.add_documents(summary_tables)
        self.retriever.docstore.mset(list(zip(table_ids, tables)))

        img_ids = [str(uuid.uuid4()) for _ in images]
        summary_img = [
            Document(page_content=summary, metadata={self.id_key: img_ids[i]})
            for i, summary in enumerate(image_summaries)
        ]
        self.retriever.vectorstore.add_documents(summary_img)
        self.retriever.docstore.mset(list(zip(img_ids, images)))

    # Parse retrieved documents into images and texts
    def _parse_retrieved_docs(self, docs: List[Document]) -> Dict[str, List]:
        images = []
        texts = []
        for doc in docs:
            try:
                base64.b64decode(str(doc))
                images.append(doc)
            except:
                texts.append(doc)
        return {"images": images, "texts": texts}

    def _build_prompt(self, context: Dict[str, List], question: str) -> ChatPromptTemplate:
        context_text = "\n".join([str(text) for text in context["texts"]])
        
        prompt_content = [{
            "type": "text",
            "text": f"Answer based on this context:\n{context_text}\nQuestion: {question}"
        }]
        
        for image in context["images"]:
            prompt_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"}
            })
            
        return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])

    # Chain
    def build_chain(self):
        return (
            {
                "context": self.retriever | RunnableLambda(self._parse_retrieved_docs),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(lambda x: self._build_prompt(x["context"], x["question"]))
            | ChatOpenAI(model="gpt-4-vision-preview")
            | StrOutputParser()
        )

 # Voice Assistant
 
async def record_and_transcribe(duration: int = 5) -> str:
    
    filename = 'user_audio.wav'
    
    # Record audio
    chunk = 1024
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=chunk)
    
    frames = []
    for _ in range(0, int(44100 / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Save audio file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(frames))
    
    # Transcribe audio
    client = speech.SpeechClient()
    with open(filename, 'rb') as audio_file:
        content = audio_file.read()
    
    audio = RecognitionAudio(content=content)
    config = RecognitionConfig(
        encoding=RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code='en-US'
    )
    
    response = client.recognize(config=config, audio=audio)
    return next((result.alternatives[0].transcript for result in response.results), "")

@cl.on_chat_start
async def start():
    rag = MultimodalRAG("C://Users//USER//OneDrive//Desktop//Testing//Specialization-Project//attention.pdf")
    summaries = await rag.generate_summaries()
    await rag.load_to_vectorstore(summaries)
    chain = rag.build_chain()
    
    cl.user_session.set("chain", chain)
    
    msg = cl.Message(content="Starting the Multimodal RAG bot...")
    await msg.send()
    msg.content = "Hi! I can help you with questions about the document, including both text and images. You can also record voice questions by typing 'record voice'. How can I assist you?"
    await msg.update()

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    if not chain:
        return
    
    try:
        content = message.content
        if content.lower() == 'record voice':
            content = await record_and_transcribe()
            
        response = await chain.ainvoke(content)
        await cl.Message(content=response).send()
    except Exception as e:
        await cl.Message(content=f"An error occurred: {e}").send()

