def local_llm(chunks, analyze):
    try:
        # Initialize embeddings with error handling
        st.info("Initializing embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Create vector store with error handling
        st.info("Creating vector store...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        split_chunks = []
        for chunk in chunks:
            split_chunks.extend(text_splitter.split_text(chunk))
        
        vectorstores = FAISS.from_texts(split_chunks, embedding=embeddings)
        docs = vectorstores.similarity_search(query=analyze, k=3)
        
        # Get LLM instance
        st.info("Getting LLM instance...")
        llm = initialize_llm()
        if not llm:
            st.error("Failed to initialize LLM")
            return None
        
        # Create and run the chain
        st.info("Running analysis...")
        chain = load_qa_chain(llm=llm, chain_type='stuff')
        response = chain.run(input_documents=docs, question=analyze)
        return response
    except Exception as e:
        st.error(f"Error in LLM processing: {str(e)}")
        return None 