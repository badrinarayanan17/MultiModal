# Hackathon
# Loading the Libraries

import os
import pyaudio
import wave
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.embeddings import OllamaEmbeddings
import chainlit as cl
from google.cloud import speech_v1p1beta1 as speech
from google.cloud.speech_v1p1beta1 import RecognitionConfig, RecognitionAudio

# Detecting environment variables
load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\USER\\Downloads\\testing-433908-bcfcc7a638e4.json"

# Prompt template
prompt_template = """
You are an AI assistant trained to answer questions based on multimodal content including text, tables, and images.

Context: {context}
Question: {question}

Provide a concise and clear answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    return prompt

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm, retriever=db.as_retriever(), chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# Loads the LLM MultiModal from ChatGroq
def load_llm():
    groqllm = ChatGroq(
        model="llama-3.2-90b-vision-preview", temperature=0
    )
    return groqllm

# Vector Embeddings
def qa_bot():
    data = PyPDFLoader('C://Users//USER//OneDrive//Desktop//Hackathon//content//attention.pdf')
    loader = data.load()
    chunk = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=500)
    splitdocs = chunk.split_documents(loader)
    index_name = "multimodal"
    db = PineconeVectorStore.from_documents(splitdocs[:5], OllamaEmbeddings(model="mxbai-embed-large"), index_name=index_name)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

# Voice Assistant Feature
def record_audio(filename, duration=5):
    """Record audio and save it as a WAV file."""
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    rate = 44100

    p = pyaudio.PyAudio()

    stream = p.open(format=format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)

    print("Recording...")

    frames = []
    for _ in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

def transcribe_audio(filename):
    """Transcribe audio using Google Cloud Speech-to-Text."""
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

    for result in response.results:
        print('Transcript: {}'.format(result.alternatives[0].transcript))
        return result.alternatives[0].transcript

@cl.on_chat_start
async def start():
    """Initialize the chatbot when the chat starts."""
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to the Multimodal RAG Bot! You can ask me questions based on text, tables, and images in the document. Additionally, you can record your voice by typing 'record voice' to interact with the bot. How can I assist you today?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    """Handle user messages."""
    chain = cl.user_session.get("chain")
    if chain is None:
        return

    try:
        if message.content.lower() == 'record voice':
            audio_filename = 'user_audio.wav'
            record_audio(audio_filename)
            transcript = transcribe_audio(audio_filename)
            message.content = transcript

        res = await chain.acall({'query': message.content})
        answer = res['result']
        await cl.Message(content=answer).send()
    except Exception as e:
        await cl.Message(content=f"An error occurred: {e}").send()

