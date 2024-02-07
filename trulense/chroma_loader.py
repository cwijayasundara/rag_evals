from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

load_dotenv()

loader = DirectoryLoader('./generator/data', glob="**/*.txt", show_progress=True, loader_cls=TextLoader)

call_transcripts = loader.load()

print(len(call_transcripts))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,
                                               chunk_overlap=0)

texts = text_splitter.split_documents(call_transcripts)

print("there are {} texts".format(len(texts)))

persist_directory = 'vectorstore'

# here we are using OpenAI embeddings but in future we will swap out to local embeddings
embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embedding,
                                 persist_directory=persist_directory)

vectordb.persist()
