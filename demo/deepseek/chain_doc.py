#!pip install pypdf

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("data/WhatisChatGPT.pdf")
pages = loader.load_and_split()

# print(pages[0].page_content)
# overlap 前后重叠长度
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,  # 思考：为什么要做overlap
    length_function=len,
    add_start_index=True,
)

paragraphs = text_splitter.create_documents([pages[0].page_content])
for para in paragraphs[:5]:
    print(para.page_content)
    print('-------')