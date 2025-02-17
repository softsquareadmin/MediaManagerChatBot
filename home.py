import json
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from streamlit_lottie import st_lottie_spinner
from langchain.memory import ConversationBufferMemory
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL
from streamlit_cookies_controller import CookieController

def render_animation():
    path = "assets/typing_animation.json"
    with open(path,"r") as file: 
        animation_json = json.load(file) 
        return animation_json

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string

st.set_page_config(
    page_title="MM - AI Chatbot",
    page_icon="https://raw.githubusercontent.com/softsquareadmin/chatbotgallery/main/Media%20Manager%203.png",
)

load_dotenv()
openaiModels = st.secrets["OPENAI_MODEL"]
portKeyApi = st.secrets["PORTKEY_API_KEY"]
pinecone_index = st.secrets["PINECONE_INDEX"]

# Load Animation
typing_animation_json = render_animation()
hide_st_style = """ <style>
                    #MainMenu {visibility:hidden;}
                    footer {visibility:hidden;}
                    header {visibility:hidden;}
                    </style>"""
st.markdown(hide_st_style, unsafe_allow_html=True)
st.markdown("""
    <h1 id="chat-header" style="position: fixed;
                   top: 0;
                   left: 0;
                   width: 100%;
                   text-align: center;
                   background-color: #f1f1f1;
                   z-index: 9">
        Media Manager - AI Chatbot
            <p>Media Manager is a Salesforce AppExchange product by Softsquare, <br> designed to streamline and manage your media assets within Salesforce.</p>
    </h1>
""", unsafe_allow_html=True)


st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    ._container_gzau3_1, ._profileContainer_gzau3_53,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* Ensure the entire page uses a flexbox layout */
    .main {
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100vh;
        padding: 0;
        margin: 0;
    }

    /* Chat container styling */
    .stApp {
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100vh;
        padding: 0;
        margin: 0;
    }

    /* Chat input should stick to the bottom */
    .stChatInput {
        position: fixed;
        bottom: 0;
        width: 95%;
        max-width: 800px; /* Adjust based on your design */
        margin: 0 auto;
        margin-bottom: 10px;
        z-index: 9999;
        border-top: 1px solid #ddd;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    /* Hide the span tag by class name */
    .st-emotion-cache-gi0tri.e1nzilvr1 {
        display: none !important;
    }

    /* Optional: Hide the span tag with a specific data-testid */
    [data-testid="stHeaderActionElements"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Hi there, I am your Media Manager Assist. How can I help you today?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'initialPageLoad' not in st.session_state:
    st.session_state['initialPageLoad'] = False

if 'selected_product_type' not in st.session_state:
    st.session_state['selected_product_type'] = 'Media Manager'

if 'prevent_loading' not in st.session_state:
    st.session_state['prevent_loading'] = False

if 'email' not in st.session_state:
    st.session_state['email'] = ''

embeddings = OpenAIEmbeddings()
controller = CookieController()

# with st.sidebar:
#     emailInput = st.text_input("Enter Your Email")
#     if emailInput != '' and emailInput != None:
#         controller.set("email_id",emailInput)

controller.set("email_id","")

email_id = str(controller.get('email_id'))
user_id = controller.get("ajs_anonymous_id")

st.session_state.email = email_id


if email_id != '' and email_id != None and email_id != 'None':
    st.markdown("""
    <style>
        section[data-testid="stSidebar"][aria-expanded="true"]{
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)

portkey_headers = createHeaders(api_key=portKeyApi,provider="openai", metadata={"email_id": email_id, "_user_id" :user_id } )

llm = ChatOpenAI(temperature=0,
                model=openaiModels,
                base_url=PORTKEY_GATEWAY_URL,
                default_headers=portkey_headers
               )

vector_store = PineconeVectorStore(index_name=pinecone_index, embedding=embeddings)

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferMemory(memory_key="chat_history",
                                    max_len=50,
                                    return_messages=True,
                                    output_key='answer')

# Answer the question as truthfully as possible using the provided context, 
# and if the answer is not contained within the text below, say 'I don't know'
general_system_template = r""" 
You are an AI support assistant for Media Manager, an AppExchange product built on the Salesforce platform by Softsquare Solutions. Your primary tools and resources include Salesforce's data model and architecture documentation, along with our product's user and admin manuals. Your role involves:
 
Key Objectives :
    - Understand User Queries: Use Natural Language Processing (NLP) to accurately interpret user questions.
    - Verify User Persona: Determine if the user is an Admin, Consultant, Developer, Business User, or Manager. Tailor your responses to fit their specific context, enhancing the personalized support experience.
    Knowledge Base Integration:
        - Dive into our product's manuals, which has detail installation steps, feature explanations, and use cases on the Salesforce platform.
        - Employ keyword matching and user intent analysis for precise searches within the knowledge base.
        - Grasp the Salesforce standard object model, understand the relationship between standard objects, understanding the architecture and feature sets.
        - Analyse example use cases for insights into problem statements, configurable steps, and their solutions.

Contextual Clarification: 
    - If needed, Ask follow-up questions to fully understand the context before providing an answer.

Conversation Analysis: 
    - Review the conversation to pinpoint keywords, error messages, and referenced features or objects. Leverage this information to formulate precise queries within Salesforce and our product's documentation.

Provide Step-by-Step Guidance: 
    - Offer detailed instructions for configuring and using Media Manager features.

Access Knowledge Base: 
    - Provide answers from pre-existing documentation, FAQs, and knowledge bases.

Troubleshoot Issues: 
    - Offer troubleshooting steps for common problems.

Escalate When Necessary: 
    - Escalate complex issues to the Media Manager support team when needed.

Media Manager Configuration Setup Objectives :
Key Elements of Media Manager Configuration setup:
    Below are the key Elements that you need to generate Media Manager configuration steps and the ways to find the important Media Manager key elements to build an Media Manager configuration from the user query :
        - Configuration Object: The primary object in File Tree tab is the one that the user wants to show or render a list of Files.
        - Render Location: The page where the Media Manager Viewer should be rendered.
.       - Key Features to Use: The Ulitimate goal of Media Manager Product is display files under any record page. Every Media Manager Product feature has unique capabilities to solve user unique problems. Try to understand every feature capabilities, the object relationship usage on those feature and based on user query and Object relationships, Identify the appropriate Media Manager feature that fits to solve the user problem statement and to generate the setup steps of Media Manager configuration. 

Configuration Verification:
        - When the user query is related to Media Manager configuration steps, display record or to render list, breakdown the user query and try to interpret all the key elements (Refer Key Elements for Media Manager Configuration setup) need to build a Media Manager configuration with that identified key elements of Media Manager Configuration. Then start responding with that identified key elements of Media Manager Configuration and provide them the actual Media Manager configuration steps (Refer Media Manager Configuration Setup Steps Response) using those key elements. If you feel hard to interpret any key elements from user query breakdown or feel like any key elements is missing on user query, then ask user for the missing key elements and confirm with them to provide them the Media Manager Configuration steps. User may also correct you with the right key elements on your interpretation to build the proper Media Manager Configuration. So Based on user response for the identified key elements, provide them the actual Media Manager configuration steps (Refer Media Manager Configuration Setup Steps Response) using those key elements. Media Manager Configuration Setup Steps response is the Ultimate goal to respond the user. So never miss to respond that after confirm the key elements from User.

    Examples :
        Below are the few examples of user query, where you can find the user query variations and key elements identified on the user query. Take these as examples of how to interpret the key elements to build Media Manager configuration.

    Example 1 : Display Account Files
                User Query Prompt Variations: "I would like to display Files under Account Object"
                Key Elements :
                    Configuration Primary Object: Account
                    Render Location: Account record detail page
                    Key Features to Use: File Tree(Primary Object as "Account")

    Example 2 : Supported views to preview images
                User Query Prompt Variations: "I need to preview the images using Tile and Slider View under Contact Page"
                Key Elements:
                        Configuration Object: Contact
                        Render Location: Contact
                        Key Features to Use: File Tree(Contact Object), Supported Views on Configuration Settings

    Example 3 : Preview the files of Case object on Account page.
                User Query Prompt Variations: "I want to preview the case files on Account Page with files details"
                Key Attributes:
                    Configuration Object: Account
                    Render Location: Account
                    Object of File to render: Case
                    Relationship: Account(Primary) > Case as child object using "AccountId"
                    Key Features to Use: File Tree, Columns

    Example 4 : Preview the parent files in Child Record Page
                User Query Prompt Variations: "Able to display contact files on child cases"
                Key Attributes:
                    Configuration Object: Account
                    Render Location: Account
                    Object of File to render: Case
                    Relationship: Account(Primary) > Case as child object using "AccountId"
                    Key Features to Use: File Tree, Columns


    Example 5 : Preview the Contact files across all contact records
                User Query Prompt Variations: "Able to display All contact files in single view"
                Key Attributes:
                    File Browser Configuration Object: Contact
                    Render Location: File Browser 
                    Object of File to render: Contact
                    Key Features to Use: File Browser configuration(Object), Supported Views(List, Slider, Tile)

Media Manager Configuration Setup Steps Response:
    - To render a Files, the component should place under any object record page. So your ultimate goal is to explain which object files and configuration setup need to render, providing step-by-step guidance using all the identified key elements also with additional requirements in user query to match with Media Manager features like Supported Views like List View, Tile View, Slider View, File Types, Filtering, File Tree, File Browser. Ensure that the instructions are clear, concise, and comprehensive to facilitate accurate configuration.

Prompting for Clarification:
    - If a user query is unclear to interpret the key elements, ask user to gather more information or clarify their needs. A good practice is to ask questions like, “Can you specify which feature you’re using?” or “Could you describe the issue in more detail?”
 
Overall Objective: 
    - Your aim is to understand the user's issue, find solutions using the appropriate key elements mentioned, and offer valuable assistance, thus resolving their concerns with Media Manager product especially providing Media Manager Configuration steps and Salesforce, and improving their overall experience.
 
DOs:
    - Highlight the bot’s benefits briefly, such as 24/7 support and quicker problem resolution.
    - Personalize responses based on the identified user type, emphasizing adaptability.
    - Clarify the sources of your knowledge, reassuring users of the reliability of the information provided.
 
DON'Ts:
    - Avoid overcomplication; aim for clarity and conciseness.
    - Steer clear of technical jargon not understood by all user types.
 
Response Style:
    - Aim for simple, human-like responses to ensure readability and clarity.
    - Use short paragraphs and bullet points for easy comprehension.

----
{context}
----
"""
general_user_template = "Question:```{question}```"

system_msg_template = SystemMessagePromptTemplate.from_template(template=general_system_template)

human_msg_template = HumanMessagePromptTemplate.from_template(template=general_user_template)
messages = [
            SystemMessagePromptTemplate.from_template(general_system_template),
            HumanMessagePromptTemplate.from_template(general_user_template)
]
qa_prompt = ChatPromptTemplate.from_messages( messages )
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vector_store.as_retriever(search_kwargs={'k': 2}),
    verbose=True,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": qa_prompt},
    rephrase_question = True,
    response_if_no_docs_found = "Sorry, I dont know",
    memory = st.session_state.buffer_memory,
    
)

# container for chat history
response_container = st.container()
textcontainer = st.container()


chat_history = []
with textcontainer:
    st.session_state.initialPageLoad = False
    query = st.chat_input(placeholder="Say something ... ", key="input")
    if query and query != "Menu":
        conversation_string = get_conversation_string()
        with st_lottie_spinner(typing_animation_json, height=50, width=50, speed=3, reverse=True):
            response = qa_chain({'question': query, 'chat_history': chat_history})
            chat_history.append((query, response['answer']))
            print("response:::: ",response)
            st.session_state.requests.append(query)
            st.session_state.responses.append(response['answer'])
    st.session_state.prevent_loading = True



with response_container:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    st.session_state.initialPageLoad = False
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            response = f"<div style='font-size:0.875rem;line-height:1.75;white-space:normal;'>{st.session_state['responses'][i]}</div>"
            message(response,allow_html=True,key=str(i),logo=('https://raw.githubusercontent.com/softsquareadmin/chatbotgallery/main/Media%20Manager%203.png'))
            if i < len(st.session_state['requests']):
                request = f"<meta name='viewport' content='width=device-width, initial-scale=1.0'><div style='font-size:.875rem'>{st.session_state['requests'][i]}</div>"
                message(request, allow_html=True,is_user=True,key=str(i)+ '_user',logo='https://raw.githubusercontent.com/Maniyuvi/SoftsquareChatbot/main/generic-user-icon-13.jpg')


