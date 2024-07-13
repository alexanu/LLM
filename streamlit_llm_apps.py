

# Stocks performance from Yahoo
        import streamlit as st
        from phi.assistant import Assistant
        from phi.llm.openai import OpenAIChat
        from phi.tools.yfinance import YFinanceTools

        # Set up the Streamlit app
        st.title("AI Investment Agent 📈🤖")
        st.caption("This app allows you to compare the performance of two stocks and generate detailed reports.")

        # Get OpenAI API key from user
        openai_api_key = st.text_input("OpenAI API Key", type="password")

        if openai_api_key:
            # Create an instance of the Assistant
            assistant = Assistant(
                llm=OpenAIChat(model="gpt-4o", api_key=openai_api_key),
                tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True)],
                show_tool_calls=True,
            )

            # Input fields for the stocks to compare
            stock1 = st.text_input("Enter the first stock symbol")
            stock2 = st.text_input("Enter the second stock symbol")

            if stock1 and stock2:
                # Get the response from the assistant
                query = f"Compare {stock1} to {stock2}. Use every tool you have."
                response = assistant.run(query, stream=False)
                st.write(response)


# Talk to webpage content (ollama=> llama3, langchain)

        # https://github.com/Shubhamsaboo/awesome-llm-apps/tree/main/llama3_local_rag

        '''
        1) Input a webpage URL
        2) Ask questions about the content of the webpage
        3) Get accurate answers using RAG and the Llama-3 model running locally on your computer

        '''

        import streamlit as st
        import ollama
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import WebBaseLoader
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import OllamaEmbeddings

        st.title("Chat with Webpage 🌐")
        st.caption("This app allows you to chat with a webpage using local llama3 and RAG")

        # Get the webpage URL from the user
        webpage_url = st.text_input("Enter Webpage URL", type="default")

        if webpage_url:
            # 1. Load the data
            loader = WebBaseLoader(webpage_url)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10) # splits into chunks
            splits = text_splitter.split_documents(docs)

            # 2. Create Ollama embeddings and vector store
            embeddings = OllamaEmbeddings(model="llama3")
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

            # 3. Call Ollama Llama3 model
            def ollama_llm(question, context):
                formatted_prompt = f"Question: {question}\n\nContext: {context}"
                response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formatted_prompt}])
                return response['message']['content']

            # 4. RAG Setup
            retriever = vectorstore.as_retriever()

            def combine_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            def rag_chain(question):
                retrieved_docs = retriever.invoke(question)
                formatted_context = combine_docs(retrieved_docs)
                return ollama_llm(question, formatted_context) # Llama-3 model is called

            st.success(f"Loaded {webpage_url} successfully!")

            # Ask a question about the webpage
            prompt = st.text_input("Ask any question about the webpage")

            # Chat with the webpage
            if prompt:
                result = rag_chain(prompt)
                st.write(result)


# Websearch (phi => openai, claude, duckduckgo)

    import streamlit as st
    from phi.assistant import Assistant
    from phi.tools.duckduckgo import DuckDuckGo
    from phi.llm.anthropic import Claude
    from phi.llm.openai import OpenAIChat


    # Set up the Streamlit app
    st.title("AI Web Search Assistant 🤖")
    st.caption("This app allows you to search the web using GPT-4o or Claude Sonnet 3.5")


    # Get API key from user
    api_key = st.text_input("LLM API Key", type="password")
    # Get LLM type from from user
    key_type = st.radio("Select mode:",('Light Mode', 'Dark Mode'))

    if api_key:
        if key_type == 'Claude':
            llm_model = Claude(model="claude-3-5-sonnet-20240620",max_tokens=1024,temperature=0.9,api_key=api_key)
        else:
            llm_model = OpenAIChat(model="gpt-4o",max_tokens=1024,temperature=0.9,api_key=api_key) 
        assistant = Assistant(llm=llm_model, tools=[DuckDuckGo()], show_tool_calls=True)
        # Get the search query from the user
        query= st.text_input("Enter the Search Query", type="default")
        
        if query:
            # Search the web using the AI Assistant
            response = assistant.run(query, stream=False)
            st.write(response)


# Webpage scraper

        import streamlit as st
        from scrapegraphai.graphs import SmartScraperGraph

        # Set up the Streamlit app
        st.title("Web Scrapping AI Agent 🕵️‍♂️")
        st.caption("This app allows you to scrape a website using OpenAI API")

        llm_provider = st.radio(
            "Select the provider of LLM",
            ["OpenAI", "Llama3"],
            index=0,
        )          

        if llm_provider == "OpenAI":
            # Get OpenAI API key from user
            openai_access_token = st.text_input("OpenAI API Key", type="password")

            if openai_access_token:
                model = st.radio(
                    "Select the model",
                    ["gpt-3.5-turbo", "gpt-4"],
                    index=0,
                )    
                graph_config = {
                    "llm": {
                        "api_key": openai_access_token,
                        "model": model,
                    },
                }
        else:
            # Set up the configuration for the SmartScraperGraph
            graph_config = {
                "llm": {
                    "model": "ollama/llama3",
                    "temperature": 0,
                    "format": "json",  # Ollama needs the format to be specified explicitly
                    "base_url": "http://localhost:11434",  # set Ollama URL
                },
                "embeddings": {
                    "model": "ollama/nomic-embed-text",
                    "base_url": "http://localhost:11434",  # set Ollama URL
                },
                "verbose": True,
            }            

        url = st.text_input("Enter the URL of the website you want to scrape")
        user_prompt = st.text_input("What you want the AI agent to scrape from the website?")
                
        # Create a SmartScraperGraph object
        smart_scraper_graph = SmartScraperGraph(
            prompt=user_prompt,
            source=url,
            config=graph_config
        )

        # Scrape the website
        if st.button("Scrape"):
            result = smart_scraper_graph.run()
            st.write(result)


# AI Journalist (phi=>openai, Serp, several agents)

        from textwrap import dedent
        from phi.assistant import Assistant
        from phi.tools.serpapi_tools import SerpApiTools
        from phi.tools.duckduckgo import DuckDuckGo
        from phi.tools.newspaper_toolkit import NewspaperToolkit
        import streamlit as st
        from phi.llm.openai import OpenAIChat

        st.title("AI Journalist Agent 🗞️")
        st.caption("Generate High-quality articles with AI Journalist by researching, wriritng and editing quality articles on autopilot using GPT-4o")


        search_engine = st.radio(
            "Select the prefered search engine",
            ["Google", "DuckDuckGo"],
            index=0,
        )       
        openai_api_key = st.text_input("Enter OpenAI API Key to access GPT-4o", type="password")
        

        if openai_api_key:
            if search_engine == 'Google':
                serp_api_key = st.text_input("Enter Serp API Key for Search functionality", type="password")
                if serp_api_key:
                    used_tools = SerpApiTools(api_key=serp_api_key)
            else:
                used_tools = DuckDuckGo()

            searcher = Assistant(
                name="Searcher",
                role="Searches for top URLs based on a topic",
                llm=OpenAIChat(model="gpt-4o", api_key=openai_api_key),
                description=dedent(
                    """\
                You are a world-class journalist for the New York Times. Given a topic, generate a list of 3 search terms
                for writing an article on that topic. Then search the web for each term, analyse the results
                and return the 10 most relevant URLs.
                """
                ),
                instructions=[
                    "Given a topic, first generate a list of 3 search terms related to that topic.",
                    "For each search term, `search_google` and analyze the results."
                    "From the results of all searcher, return the 10 most relevant URLs to the topic.",
                    "Remember: you are writing for the New York Times, so the quality of the sources is important.",
                ],
                tools=[used_tools],
                show_tool_calls=True, 
                add_datetime_to_instructions=True,
            )
            writer = Assistant(
                name="Writer",
                role="Retrieves text from URLs and writes a high-quality article",
                llm=OpenAIChat(model="gpt-4o", api_key=openai_api_key),
                description=dedent(
                    """\
                You are a senior writer for the New York Times. Given a topic and a list of URLs,
                your goal is to write a high-quality NYT-worthy article on the topic.
                """
                ),
                instructions=[
                    "Given a topic and a list of URLs, first read the article using `get_article_text`."
                    "Then write a high-quality NYT-worthy article on the topic."
                    "The article should be well-structured, informative, and engaging",
                    "Ensure the length is at least as long as a NYT cover story -- at a minimum, 15 paragraphs.",
                    "Ensure you provide a nuanced and balanced opinion, quoting facts where possible.",
                    "Remember: you are writing for the New York Times, so the quality of the article is important.",
                    "Focus on clarity, coherence, and overall quality.",
                    "Never make up facts or plagiarize. Always provide proper attribution.",
                ],
                tools=[NewspaperToolkit()],
                add_datetime_to_instructions=True,
                add_chat_history_to_prompt=True,
                num_history_messages=3,
            )

            editor = Assistant(
                name="Editor",
                llm=OpenAIChat(model="gpt-4o", api_key=openai_api_key),
                team=[searcher, writer],
                description="You are a senior NYT editor. Given a topic, your goal is to write a NYT worthy article.",
                instructions=[
                    "Given a topic, ask the search journalist to search for the most relevant URLs for that topic.",
                    "Then pass a description of the topic and URLs to the writer to get a draft of the article.",
                    "Edit, proofread, and refine the article to ensure it meets the high standards of the New York Times.",
                    "The article should be extremely articulate and well written. "
                    "Focus on clarity, coherence, and overall quality.",
                    "Ensure the article is engaging and informative.",
                    "Remember: you are the final gatekeeper before the article is published.",
                ],
                add_datetime_to_instructions=True,
                markdown=True,
            )

            # Input field for the report query
            query = st.text_input("What do you want the AI journalist to write an Article on?")

            if query:
                with st.spinner("Processing..."):
                    # Get the response from the assistant
                    response = editor.run(query, stream=False)
                    st.write(response)


# Hackernews (phi, openai or llama3, agents)

        import streamlit as st
        from phi.assistant import Assistant
        from phi.tools.hackernews import HackerNews
        from phi.llm.openai import OpenAIChat
        from phi.llm.ollama import Ollama

        # Set up the Streamlit app
        st.title("Multi-Agent AI Researcher 🔍🤖")
        st.caption("This app allows you to research top stories and users on HackerNews and write blogs, reports and social posts.")

    
        openai_api_key = st.text_input("OpenAI API Key", type="password")

        if openai_api_key:
            llm_model = Ollama(model="llama3:instruct", max_tokens=1024)
        else:
            llm_model = OpenAIChat(model="gpt-4o",max_tokens=1024,temperature=0.5,api_key=openai_api_key)

        story_researcher = Assistant(
            name="HackerNews Story Researcher",
            role="Researches hackernews stories and users.",
            tools=[HackerNews()],
            llm= llm_model if openai_api_key else None
        )

        user_researcher = Assistant(
            name="HackerNews User Researcher",
            role="Reads articles from URLs.",
            tools=[HackerNews()],
            llm= llm_model if openai_api_key else None
        )

        hn_assistant = Assistant(
            name="Hackernews Team",
            team=[story_researcher, user_researcher],
            llm=llm_model
        )

        query = st.text_input("Enter your report query")

        if query:
            response = hn_assistant.run(query, stream=False)
            st.write(response)


# Travel Planner (phi=>openai, Serp, several agents)

        from textwrap import dedent
        from phi.assistant import Assistant
        from phi.tools.serpapi_tools import SerpApiTools
        import streamlit as st
        from phi.llm.openai import OpenAIChat

        st.title("AI Travel Planner ✈️")
        st.caption("Plan your next adventure with AI Travel Planner by researching and planning a personalized itinerary on autopilot using GPT-4o")

        openai_api_key = st.text_input("Enter OpenAI API Key to access GPT-4o", type="password")
        serp_api_key = st.text_input("Enter Serp API Key for Search functionality", type="password")

        if openai_api_key and serp_api_key:
            researcher = Assistant(
                name="Researcher",
                role="Searches for travel destinations, activities, and accommodations based on user preferences",
                llm=OpenAIChat(model="gpt-4o", api_key=openai_api_key),
                description=dedent(
                    """\
                You are a world-class travel researcher. Given a travel destination and the number of days the user wants to travel for,
                generate a list of search terms for finding relevant travel activities and accommodations.
                Then search the web for each term, analyze the results, and return the 10 most relevant results.
                """
                ),
                instructions=[
                    "Given a travel destination and the number of days the user wants to travel for, first generate a list of 3 search terms related to that destination and the number of days.",
                    "For each search term, `search_google` and analyze the results."
                    "From the results of all searches, return the 10 most relevant results to the user's preferences.",
                    "Remember: the quality of the results is important.",
                ],
                tools=[SerpApiTools(api_key=serp_api_key)],
                add_datetime_to_instructions=True,
            )
            planner = Assistant(
                name="Planner",
                role="Generates a draft itinerary based on user preferences and research results",
                llm=OpenAIChat(model="gpt-4o", api_key=openai_api_key),
                description=dedent(
                    """\
                You are a senior travel planner. Given a travel destination, the number of days the user wants to travel for, and a list of research results,
                your goal is to generate a draft itinerary that meets the user's needs and preferences.
                """
                ),
                instructions=[
                    "Given a travel destination, the number of days the user wants to travel for, and a list of research results, generate a draft itinerary that includes suggested activities and accommodations.",
                    "Ensure the itinerary is well-structured, informative, and engaging.",
                    "Ensure you provide a nuanced and balanced itinerary, quoting facts where possible.",
                    "Remember: the quality of the itinerary is important.",
                    "Focus on clarity, coherence, and overall quality.",
                    "Never make up facts or plagiarize. Always provide proper attribution.",
                ],
                add_datetime_to_instructions=True,
                add_chat_history_to_prompt=True,
                num_history_messages=3,
            )

            destination = st.text_input("Where do you want to go?")
            num_days = st.number_input("How many days do you want to travel for?", min_value=1, max_value=30, value=7)

            if st.button("Generate Itinerary"):
                with st.spinner("Processing..."):
                    response = planner.run(f"{destination} for {num_days} days", stream=False)
                    st.write(response)


# Talk to pdf 1 (embedchain, llama3 or openai)
        import os
        import tempfile
        import streamlit as st
        from embedchain import App

        # Define the embedchain_bot function
        def embedchain_bot(db_path, api_key=None):
            if api_key:
                return App.from_config(
                    config={
                        "llm": {"provider": "openai", "config": {"api_key": api_key}},
                        "embedder": {"provider": "openai", "config": {"api_key": api_key}},
                        "vectordb": {"provider": "chroma", "config": {"dir": db_path}},
                    }
                )
            else:
                return App.from_config(
                    config={
                        "llm": {"provider": "ollama", "config": {"model": "llama3:instruct", "max_tokens": 250, "temperature": 0.5, "stream": True, "base_url": 'http://localhost:11434'}},
                        "embedder": {"provider": "ollama", "config": {"model": "llama3:instruct", "base_url": 'http://localhost:11434'}},
                        "vectordb": {"provider": "chroma", "config": {"dir": db_path}},
                    }
                )

        st.title("Chat with PDF")
        st.caption("This app allows you to chat with a PDF using Llama3 running locally wiht Ollama!")

        openai_access_token = st.text_input("OpenAI API Key", type="password")

        db_path = tempfile.mkdtemp() # Create a temporary directory to store the PDF file

        if openai_access_token:
            app = embedchain_bot(db_path, openai_access_token) # Create an instance of the embedchain App
        else:
            app = embedchain_bot(db_path) # Create an instance of the embedchain App


        pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

        if pdf_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(pdf_file.getvalue())
                app.add(f.name, data_type="pdf_file")
            os.remove(f.name)
            st.success(f"Added {pdf_file.name} to knowledge base!")

        prompt = st.text_input("Ask a question about the PDF")

        if prompt:
            answer = app.chat(prompt)
            st.write(answer)


# Talk to pdf 2 (langchain, PyPDF2, openai)

        import os
        from PyPDF2 import PdfReader
        import streamlit as st
        from langchain.text_splitter import CharacterTextSplitter
        from langchain.embeddings.openai import OpenAIEmbeddings
        from langchain import FAISS
        from langchain.chains.question_answering import load_qa_chain
        from langchain.llms import OpenAI
        from langchain.callbacks import get_openai_callback

        import langchain
        langchain.verbose = False

        # process text from pdf
        def process_text(text):
            # split the text into chunks using langchain
            text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
            chunks = text_splitter.split_text(text)

            # convert the chunks of text into embeddings to form a knowledge base
            embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

            knowledge_base = FAISS.from_texts(chunks, embeddings)

            return knowledge_base

        st.title("Chat with my PDF")

        pdf = st.file_uploader("Upload your PDF File", type="pdf")

        if pdf is not None:
            pdf_reader = PdfReader(pdf)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            knowledgeBase = process_text(text)

            query = st.text_input('Ask question to PDF...')

            cancel_button = st.button('Cancel')
            if cancel_button:
                st.stop()

            if query:
                docs = knowledgeBase.similarity_search(query)
                llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))
                chain = load_qa_chain(llm, chain_type="stuff")

                with get_openai_callback() as cost:
                    response = chain.invoke(input={"question": query, "input_documents": docs})
                    print(cost)
                    st.write(response["output_text"])


# Talk to Arxiv paper (phi, llama3 or openai)
        import streamlit as st
        from phi.assistant import Assistant
        from phi.llm.ollama import Ollama
        from phi.llm.openai import OpenAIChat
        from phi.tools.arxiv_toolkit import ArxivToolkit

        # Set up the Streamlit app
        st.title("Chat with Research Papers 🔎🤖")
        st.caption("This app allows you to chat with arXiv research papers using OpenAI GPT-4o model or Llama-3 running locally.")

        # Get OpenAI API key from user
        openai_access_token = st.text_input("OpenAI API Key", type="password")

        # Create an instance of the Assistant
        if openai_access_token:    
            assistant = Assistant(
                llm=OpenAIChat(
                    model="gpt-4o",
                    max_tokens=1024,
                    temperature=0.9,
                    api_key=openai_access_token) , 
                tools=[ArxivToolkit()]
            )
        else:
            assistant = Assistant(
                llm=Ollama(
                    model="llama3:instruct"), 
                tools=[ArxivToolkit()], 
                show_tool_calls=True
            )

        # Get the search query from the user
        query= st.text_input("Enter the Search Query", type="default")

        if query:
            # Search the web using the AI Assistant
            response = assistant.run(query, stream=False)
            st.write(response)


# Talk to Youtube video (embedchain)

        import tempfile
        import streamlit as st
        from embedchain import App

        # Define the embedchain_bot function
        def embedchain_bot(db_path, api_key):
            return App.from_config(
                config={
                    "llm": {"provider": "openai", "config": {"model": "gpt-4-turbo", "temperature": 0.5, "api_key": api_key}},
                    "vectordb": {"provider": "chroma", "config": {"dir": db_path}},
                    "embedder": {"provider": "openai", "config": {"api_key": api_key}},
                }
            )

        st.title("Chat with YouTube Video 📺")
        st.caption("This app allows you to chat with a YouTube video using OpenAI API")

        openai_access_token = st.text_input("OpenAI API Key", type="password")

        if openai_access_token:
            db_path = tempfile.mkdtemp() # Create a temporary directory to store the database
            app = embedchain_bot(db_path, openai_access_token) # Create an instance of Embedchain App
            video_url = st.text_input("Enter YouTube Video URL", type="default")
            if video_url:
                app.add(video_url, data_type="youtube_video")
                st.success(f"Added {video_url} to knowledge base!")
                prompt = st.text_input("Ask any question about the YouTube Video")
                if prompt:
                    answer = app.chat(prompt)
                    st.write(answer)


# Personal Finance Agent (phi=> openai, serp, planner + researcher)

        from textwrap import dedent
        from phi.assistant import Assistant
        from phi.tools.serpapi_tools import SerpApiTools
        import streamlit as st
        from phi.llm.openai import OpenAIChat

        st.title("AI Personal Finance Planner 💰")
        st.caption("Manage your finances with AI Personal Finance Manager by creating personalized budgets, investment plans, and savings strategies using GPT-4o")

        openai_api_key = st.text_input("Enter OpenAI API Key to access GPT-4o", type="password")
        serp_api_key = st.text_input("Enter Serp API Key for Search functionality", type="password")

        if openai_api_key and serp_api_key:
            researcher = Assistant(
                name="Researcher",
                role="Searches for financial advice, investment opportunities, and savings strategies based on user preferences",
                llm=OpenAIChat(model="gpt-4o", api_key=openai_api_key),
                description=dedent(
                    """\
                You are a world-class financial researcher. Given a user's financial goals and current financial situation,
                generate a list of search terms for finding relevant financial advice, investment opportunities, and savings strategies.
                Then search the web for each term, analyze the results, and return the 10 most relevant results.
                """
                ),
                instructions=[
                    "Given a user's financial goals and current financial situation, first generate a list of 3 search terms related to those goals.",
                    "For each search term, `search_google` and analyze the results.",
                    "From the results of all searches, return the 10 most relevant results to the user's preferences.",
                    "Remember: the quality of the results is important.",
                ],
                tools=[SerpApiTools(api_key=serp_api_key)],
                add_datetime_to_instructions=True,
            )
            planner = Assistant(
                name="Planner",
                role="Generates a personalized financial plan based on user preferences and research results",
                llm=OpenAIChat(model="gpt-4o", api_key=openai_api_key),
                description=dedent(
                    """\
                You are a senior financial planner. Given a user's financial goals, current financial situation, and a list of research results,
                your goal is to generate a personalized financial plan that meets the user's needs and preferences.
                """
                ),
                instructions=[
                    "Given a user's financial goals, current financial situation, and a list of research results, generate a personalized financial plan that includes suggested budgets, investment plans, and savings strategies.",
                    "Ensure the plan is well-structured, informative, and engaging.",
                    "Ensure you provide a nuanced and balanced plan, quoting facts where possible.",
                    "Remember: the quality of the plan is important.",
                    "Focus on clarity, coherence, and overall quality.",
                    "Never make up facts or plagiarize. Always provide proper attribution.",
                ],
                add_datetime_to_instructions=True,
                add_chat_history_to_prompt=True,
                num_history_messages=3,
            )

            # Input fields for the user's financial goals and current financial situation
            financial_goals = st.text_input("What are your financial goals?")
            current_situation = st.text_area("Describe your current financial situation")

            if st.button("Generate Financial Plan"):
                with st.spinner("Processing..."):
                    # Get the response from the assistant
                    response = planner.run(f"Financial goals: {financial_goals}, Current situation: {current_situation}", stream=False)
                    st.write(response)


# Chatbot (+ talk to image, chatbot layout, gemini or local lm studio with Llama 3)

        import os
        import streamlit as st
        import google.generativeai as genai
        from PIL import Image

        from openai import OpenAI

        st.set_page_config(page_title="Multimodal Chatbot with Gemini Flash", layout="wide")
        st.title("Multimodal Chatbot with Gemini Flash ⚡️")
        st.caption("Chat with Google's Gemini Flash model using image and text input to get lightning fast results. 🌟")
        st.caption("Chat with locally hosted memory-enabled Llama-3 using the LM Studio 💯")

        api_key = st.text_input("Enter Google API Key", type="password")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")


        # Point to the local server setup using LM Studio
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")


        if api_key:
            # Initialize the chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Sidebar for image upload
            with st.sidebar:
                st.title("Chat with Images")
                uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)

            # Main layout
            chat_placeholder = st.container()

            with chat_placeholder:
                # Display the chat history
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            # User input area at the bottom
            prompt = st.chat_input("What do you want to know?")

            if prompt:
                inputs = [prompt]
                
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                # Display user message in chat message container
                with chat_placeholder:
                    with st.chat_message("user"):
                        st.markdown(prompt)
                
                if uploaded_file:
                    inputs.append(image)

                with st.spinner('Generating response...'):
                    response = model.generate_content(inputs)
                    response_text = response.text

                    response = client.chat.completions.create(
                        model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
                        messages=st.session_state.messages, temperature=0.7
                    )
                    response_text = response.choices[0].message.content


                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                # Display assistant response in chat message container

                with chat_placeholder:
                    with st.chat_message("assistant"):
                        st.markdown(response_text)

            if uploaded_file and not prompt:
                st.warning("Please enter a text query to accompany the image.")