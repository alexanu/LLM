from datetime import date
from newsapi import NewsApiClient
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import json
import ollama

#Paste your Api key
newsapi = NewsApiClient(api_key='')

def latest_news(data):
    try:
        all_articles = newsapi.get_everything(q=data, language='en', sort_by='publishedAt')
        extracted_data = []
        k=0
        for article in all_articles['articles']:
            if k>8:
                break
            extracted_data.append({
                'description': article.get('description', 'No description available'),
                'url': article.get('url', 'No Url')
                        })
        with open('news.json', 'w') as p:
            json.dump(extracted_data, p)
    except Exception as e:
        print(f"Failed to fetch news articles: {e}")
        return None
    
def create_vector_store(file_path):

    # Load documents from a JSON file
    loader = JSONLoader(file_path=file_path, jq_schema='.[] | { description: .description, url: .url}', text_content=False)
    documents = loader.load()

    # Split documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embedding_model = OllamaEmbeddings(model="llama3")
    vector_store = Chroma.from_documents(documents=documents, embedding=embedding_model)

    return vector_store.as_retriever()


def generate_newsletter(topic):
    latest_news(topic)   
    question = f"""
        # Your Daily Digest: {date.today()}
    
        Welcome to your curated news update, bringing you the latest and most relevant headlines directly to your inbox.
    
        ## Today's Top Story
        ### [Title of the Main News Article](URL_to_article)
        Provide a brief introduction to the top story of the day, emphasizing the main points succinctly.
    
        ---
    
        ## More News
    
        ### [Second News Article Title](URL_to_second_article)
        **Summary**: Offer a concise summary of the second most important news of the day.
    
        ### [Third News Article Title](URL_to_third_article)
        **Summary**: Summarize this article, highlighting key details that inform the reader effectively.
    
        ### [Fourth News Article Title](URL_to_fourth_article)
        **Summary**: Briefly cover the fourth article, focusing on crucial points.
    
        ### [Fifth News Article Title](URL_to_fifth_article)
        **Summary**: Sum up the fifth article, ensuring to pinpoint essential information.
    
        ---
    
        **Instructions**:
        - Write a news summary for the topic: '{topic}'.
        - Ensure the news summaries do not repeat information.
        - Follow the structure provided above as a template for the news summary.
        """


    retriever = create_vector_store('news.json')
    
    formatted_context = "\n\n".join(doc.page_content for doc in retriever.invoke(topic))
    formatted_prompt = f"Question: {question}\n\nContext: {formatted_context}"
    llm_response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formatted_prompt}])
    return llm_response['message']['content']


# newsletter = generate_newsletter('World News')
# display(Markdown(newsletter))
