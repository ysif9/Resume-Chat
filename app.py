import re

import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from spacy.lang.en import English

from htmlTemplates import css, bot_template, user_template


llm = OllamaLLM(model="gemma2:2b")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_text(raw_text)
    if  chunks:
        print(len(chunks))
    return chunks


def get_vector_store(documents):
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(documents=documents, embedding=embedding_model)
    return vectorstore


def analyze_resume(resume, job_description):
    prompt_template = PromptTemplate(
        input_variables=["resume", "job_description"],
        template=("""
                You are an AI assistant specialized in resume analysis and recruitment. Analyze the given resume and compare it with the job description. 

                Example Response Structure:

                **OVERVIEW**:
                - **Match Percentage**: [Calculate overall match percentage between the resume and job description]
                - **Matched Skills**: [List the skills in job description that match the resume]
                - **Unmatched Skills**: [List the skills in the job description that are missing in the resume]

                **DETAILED ANALYSIS**:
                Provide a detailed analysis about:
                1. Overall match percentage between the resume and job description
                2. List of skills from the job description that match the resume
                3. List of skills from the job description that are missing in the resume

                **Additional Comments**:
                Additional comments about the resume and suggestions for the recruiter or HR manager.

                Resume: {resume}
                Job Description: {job_description}

                Analysis:
                """
                  )
    )

    chain = prompt_template | llm
    response = chain.invoke({"resume": resume, "job_description": job_description})

    return response


def get_conversation_chain(vectorstore):

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
        }
    )

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )


    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_retrieval_chain = RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history=st.session_state["chat_history"],
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_retrieval_chain


def handle_user_input(user_question):

    input_data = {
        "input": user_question,
        "chat_history": st.session_state.chat_history,
    }

    response = st.session_state.conversation.invoke(input_data)

    # answer_text = response['answer']
    # st.write(st.session_state.chat_history)

    for i, message in enumerate(st.session_state.chat_history.messages):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def parse_resume(raw_text):
    # try with spaCy
    sections = {
        "Contact Info" : "",
        "Summary" : "",
        "Experience" : "",
        "Education": "",
        "Projects" : "",
        "Skills" : "",
    }

    patterns = {
        "Contact Info": r"(phone|email|linkedin|github|address)",
        "Summary": r"(^|\n)(summary|objective|brief)\b",
        "Experience": r"(^|\n)(experience|employment|work history)\b",
        "Education": r"(^|\n)(education|academic)\b",
        "Projects": r"(^|\n)(projects)\b",
        "Skills": r"(^|\n)(skills|technologies|tools|technical skills|certificates)\b"
    }


    for section, pattern in patterns.items():
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            start = match.start()

            # search for the other patterns to stop unless the pattern is already full
            # would be wrong with multiple skill headings for example, fix later
            other_patterns = [p for s, p in patterns.items() if (s != section and not sections[s])]
            all_other_patterns = "|".join(other_patterns)

            # search for the end
            next_section_match = re.search(rf"(?=(?={all_other_patterns}))", raw_text[start+1:], re.IGNORECASE)
            end = next_section_match.start() + start if next_section_match else len(raw_text)

            # combine text from start to end
            sections[section] = raw_text[start:end].strip()

    return sections

def tokenize_description(description):
    nlp = English()
    nlp.add_pipe("sentencizer")

    text = nlp(description)

    # remove stop words and punctuation
    filtered_tokens = [token.text for token in text if not token.is_stop]
    filtered_text = " ".join(filtered_tokens)
    filtered_text = nlp(filtered_text)

    # extract sentences and clean
    filtered_text_sentences = [sentence.text.strip() for sentence in filtered_text.sents]


    return list(filtered_text_sentences)


def chunk_resume(key_sections):
    chunked_resume = []
    for section, text in key_sections.items():
        if text:
            for chunk in get_text_chunks(text):
                chunked_section = Document(
                    page_content=chunk,
                    metadata={"type": section}
                )
                chunked_resume.append(chunked_section)
    return chunked_resume

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ChatMessageHistory()

    st.header("Chat with your Resume")
    user_question = st.text_input("Ask a question about your resume:")
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Your Resume")
        pdf_docs = st.file_uploader("Upload your Resume here and click on Process", accept_multiple_files=True)
        job_description = st.text_area("Enter the Job Description here")
        if st.button("Process"):
            with st.spinner("Processing"):

                # extract pdf text
                raw_text = get_pdf_text(pdf_docs)

                # Identify key sections
                key_sections = parse_resume(raw_text)
                # st.write(key_sections)

                # tokenize job description
                tokenized_description = tokenize_description(job_description)
                # st.write(tokenized_description)

                # get the text chunks with section metadata
                key_sections_chunks = chunk_resume(key_sections)
                # st.write(key_sections_chunks)


                # create vector store
                vectorstore = get_vector_store(key_sections_chunks)

                # Initial Analyze
                response = analyze_resume(key_sections_chunks,tokenized_description)
                st.write(response)

                # conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)




if __name__ == "__main__":
    main()