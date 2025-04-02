import time
import numpy as np
import pandas as pd 
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
import os

import warnings
warnings.filterwarnings('ignore')

def initialize_llm():
    """Initialize the local LLM model with optimized parameters for better performance"""
    try:
        model_path = "models/llama-2-7b-chat.Q4_K_M.gguf"
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}")
            return None
            
        st.info("Loading LLM model... This may take a few moments.")
        llm = LlamaCpp(
            model_path=model_path,
            temperature=0.7,
            max_tokens=2000,
            top_p=0.9,
            verbose=True,
            n_ctx=2048,
            n_threads=4,
            n_batch=512,
            n_gpu_layers=0,
            f16_kv=True,
            seed=42
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

def streamlit_config():
    st.set_page_config(page_title='Talent Track By AI', layout="wide")
    page_background_color = """
    <style>
    [data-testid="stHeader"] 
    {
    background: rgba(0,0,0,0);
    }
    </style>
    """
    st.markdown(page_background_color, unsafe_allow_html=True)
    st.markdown(f'<h1 style="text-align: center;">Talent Track By AI</h1>', unsafe_allow_html=True)

def process_resume(pdf):
    if pdf is not None:
        try:
            with st.spinner('Processing...'):
                pdf_chunks = resume_analyzer.pdf_to_chunks(pdf)
                summary_prompt = resume_analyzer.summary_prompt(query_with_chunks=pdf_chunks)
                summary = resume_analyzer.local_llm(chunks=pdf_chunks, analyze=summary_prompt)
                if summary:
                    st.session_state['resume_data'] = {
                        'pdf': pdf,
                        'chunks': pdf_chunks,
                        'summary': summary
                    }
                    return True
        except Exception as e:
            st.markdown(f'<h5 style="text-align: center;color: orange;">{e}</h5>', unsafe_allow_html=True)
    return False

class resume_analyzer:
    def pdf_to_chunks(pdf):
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=200,
            length_function=len)
        chunks = text_splitter.split_text(text=text)
        return chunks

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

    def summary_prompt(query_with_chunks):
        query = f''' need to detailed summarization of below resume and finally conclude them
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    {query_with_chunks}
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    '''
        return query

    def resume_summary():
        with st.form(key='Summary'):
            add_vertical_space(1)
            if 'resume_data' not in st.session_state:
                pdf = st.file_uploader(label='Upload Your Resume', type='pdf')
                add_vertical_space(2)
                submit = st.form_submit_button(label='Submit')
                add_vertical_space(1)
            else:
                st.info("Using previously uploaded resume")
                submit = st.form_submit_button(label='Analyze Again')
                add_vertical_space(1)
        
        add_vertical_space(3)
        if submit:
            if 'resume_data' not in st.session_state:
                if pdf is not None:
                    if process_resume(pdf):
                        st.markdown(f'<h4 style="color: orange;">Summary:</h4>', unsafe_allow_html=True)
                        st.write(st.session_state['resume_data']['summary'])
            else:
                st.markdown(f'<h4 style="color: orange;">Summary:</h4>', unsafe_allow_html=True)
                st.write(st.session_state['resume_data']['summary'])

    def strength_prompt(query_with_chunks):
        query = f'''need to detailed analysis and explain of the strength of below resume and finally conclude them
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    {query_with_chunks}
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    '''
        return query

    def resume_strength():
        with st.form(key='Strength'):
            add_vertical_space(1)
            if 'resume_data' not in st.session_state:
                pdf = st.file_uploader(label='Upload Your Resume', type='pdf')
                add_vertical_space(2)
                submit = st.form_submit_button(label='Submit')
                add_vertical_space(1)
            else:
                st.info("Using previously uploaded resume")
                submit = st.form_submit_button(label='Analyze Again')
                add_vertical_space(1)

        add_vertical_space(3)
        if submit:
            if 'resume_data' not in st.session_state:
                if pdf is not None:
                    if process_resume(pdf):
                        strength_prompt = resume_analyzer.strength_prompt(query_with_chunks=st.session_state['resume_data']['summary'])
                        strength = resume_analyzer.local_llm(chunks=st.session_state['resume_data']['chunks'], analyze=strength_prompt)
                        if strength:
                            st.markdown(f'<h4 style="color: orange;">Strength:</h4>', unsafe_allow_html=True)
                            st.write(strength)
            else:
                strength_prompt = resume_analyzer.strength_prompt(query_with_chunks=st.session_state['resume_data']['summary'])
                strength = resume_analyzer.local_llm(chunks=st.session_state['resume_data']['chunks'], analyze=strength_prompt)
                if strength:
                    st.markdown(f'<h4 style="color: orange;">Strength:</h4>', unsafe_allow_html=True)
                    st.write(strength)

    def weakness_prompt(query_with_chunks):
        query = f'''need to detailed analysis and explain of the weakness of below resume and how to improve make a better resume.
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    {query_with_chunks}
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    '''
        return query

    def resume_weakness():
        with st.form(key='Weakness'):
            add_vertical_space(1)
            if 'resume_data' not in st.session_state:
                pdf = st.file_uploader(label='Upload Your Resume', type='pdf')
                add_vertical_space(2)
                submit = st.form_submit_button(label='Submit')
                add_vertical_space(1)
            else:
                st.info("Using previously uploaded resume")
                submit = st.form_submit_button(label='Analyze Again')
                add_vertical_space(1)
        
        add_vertical_space(3)
        if submit:
            if 'resume_data' not in st.session_state:
                if pdf is not None:
                    if process_resume(pdf):
                        weakness_prompt = resume_analyzer.weakness_prompt(query_with_chunks=st.session_state['resume_data']['summary'])
                        weakness = resume_analyzer.local_llm(chunks=st.session_state['resume_data']['chunks'], analyze=weakness_prompt)
                        if weakness:
                            st.markdown(f'<h4 style="color: orange;">Weakness and Suggestions:</h4>', unsafe_allow_html=True)
                            st.write(weakness)
            else:
                weakness_prompt = resume_analyzer.weakness_prompt(query_with_chunks=st.session_state['resume_data']['summary'])
                weakness = resume_analyzer.local_llm(chunks=st.session_state['resume_data']['chunks'], analyze=weakness_prompt)
                if weakness:
                    st.markdown(f'<h4 style="color: orange;">Weakness and Suggestions:</h4>', unsafe_allow_html=True)
                    st.write(weakness)

    def job_title_prompt(query_with_chunks):
        query = f''' what are the job roles i apply to likedin based on below?
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    {query_with_chunks}
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    '''
        return query

    def job_title_suggestion():
        with st.form(key='Job Titles'):
            add_vertical_space(1)
            if 'resume_data' not in st.session_state:
                pdf = st.file_uploader(label='Upload Your Resume', type='pdf')
                add_vertical_space(2)
                submit = st.form_submit_button(label='Submit')
                add_vertical_space(1)
            else:
                st.info("Using previously uploaded resume")
                submit = st.form_submit_button(label='Analyze Again')
                add_vertical_space(1)

        add_vertical_space(3)
        if submit:
            if 'resume_data' not in st.session_state:
                if pdf is not None:
                    if process_resume(pdf):
                        job_title_prompt = resume_analyzer.job_title_prompt(query_with_chunks=st.session_state['resume_data']['summary'])
                        job_title = resume_analyzer.local_llm(chunks=st.session_state['resume_data']['chunks'], analyze=job_title_prompt)
                        if job_title:
                            st.markdown(f'<h4 style="color: orange;">Job Titles:</h4>', unsafe_allow_html=True)
                            st.write(job_title)
            else:
                job_title_prompt = resume_analyzer.job_title_prompt(query_with_chunks=st.session_state['resume_data']['summary'])
                job_title = resume_analyzer.local_llm(chunks=st.session_state['resume_data']['chunks'], analyze=job_title_prompt)
                if job_title:
                    st.markdown(f'<h4 style="color: orange;">Job Titles:</h4>', unsafe_allow_html=True)
                    st.write(job_title)

class linkedin_scraper:
    @staticmethod
    def webdriver_setup():
        """Set up Chrome webdriver with enhanced anti-detection measures"""
        try:
            options = webdriver.ChromeOptions()
            
            # Basic options
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--disable-extensions')
            options.add_argument('--disable-notifications')
            
            # Window size and display
            options.add_argument('--window-size=1920,1080')
            options.add_argument('--start-maximized')
            
            # Enhanced privacy and security settings
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_argument('--disable-web-security')
            options.add_argument('--allow-running-insecure-content')
            options.add_argument('--ignore-certificate-errors')
            options.add_argument('--ignore-ssl-errors')
            
            # Random user agent
            user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Edge/120.0.0.0'
            ]
            user_agent = np.random.choice(user_agents)
            options.add_argument(f'--user-agent={user_agent}')
            
            # Experimental options
            options.add_experimental_option('excludeSwitches', ['enable-automation', 'enable-logging'])
            options.add_experimental_option('useAutomationExtension', False)
            
            # Create driver
            driver = webdriver.Chrome(options=options)
            
            # Additional JavaScript to avoid detection
            driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": user_agent})
            
            # Modify navigator properties
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            driver.execute_script("Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']})")
            driver.execute_script("Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]})")
            
            # Set viewport and window size
            driver.execute_cdp_cmd('Emulation.setDeviceMetricsOverride', {
                'mobile': False,
                'width': 1920,
                'height': 1080,
                'deviceScaleFactor': 1,
            })
            
            return driver
            
        except Exception as e:
            st.error(f"Failed to initialize Chrome driver: {str(e)}")
            st.info("Please ensure Chrome browser is installed and updated to the latest version")
            return None

    @staticmethod
    def get_userinput():
        """Get job search parameters from user"""
        job_title = st.text_input('Enter Job Titles (comma separated):', 'Data Scientist')
        job_location = st.text_input('Enter Job Location:', 'India')
        job_count = st.number_input('Enter Number of Jobs to Scrape (max 100):', min_value=1, max_value=100, value=2)
        return job_title.split(','), job_location, job_count

    @staticmethod
    def build_url(job_title, job_location):
        """Build LinkedIn search URL"""
        formatted_title = '%20'.join(job_title[0].strip().split())  # Use first job title only
        formatted_location = '%20'.join(job_location.split())
        return f"https://www.linkedin.com/jobs/search?keywords={formatted_title}&location={formatted_location}"

    @staticmethod
    def scroll_page(driver, job_count):
        """Scroll page to load more jobs"""
        try:
            st.info("Scrolling page to load more jobs...")
            # Calculate number of scrolls needed (25 jobs per scroll approximately)
            scrolls = min(job_count // 25 + 1, 4)
            
            for i in range(scrolls):
                st.info(f"Scroll attempt {i+1}/{scrolls}")
                # Scroll to bottom
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(4)  # Wait for content to load
                
                try:
                    # Try to click "Show more" button if present
                    show_more_buttons = driver.find_elements(by=By.CSS_SELECTOR, value=[
                        "button.infinite-scroller__show-more-button",
                        "button.see-more-jobs",
                        "button[data-tracking-control-name='infinite-scroller_show-more']"
                    ])
                    
                    for button in show_more_buttons:
                        if button.is_displayed():
                            driver.execute_script("arguments[0].click();", button)
                            time.sleep(3)  # Wait for new content
                            break
                            
                except Exception as e:
                    st.warning(f"Could not find or click 'Show more' button: {str(e)}")
                    
                # Additional wait after last scroll
                if i == scrolls - 1:
                    time.sleep(5)
                    
        except Exception as e:
            st.warning(f"Error during page scrolling: {str(e)}")

    @staticmethod
    def scrape_jobs(driver, job_count):
        """Scrape job listings from LinkedIn with updated selectors"""
        jobs_data = {
            'company_name': [],
            'job_title': [],
            'location': [],
            'job_url': []
        }

        try:
            # Wait for job cards to load with explicit wait
            st.info("Waiting for page to load...")
            time.sleep(8)  # Increased initial wait time
            
            # Try multiple selectors for job cards
            selectors = [
                "div.job-card-container",
                "li.jobs-search-results__list-item",
                "div.base-card",
                "div.job-search-card",
                "li.jobs-search-results-list__list-item"
            ]
            
            job_cards = []
            for selector in selectors:
                try:
                    job_cards = driver.find_elements(by=By.CSS_SELECTOR, value=selector)
                    if job_cards:
                        st.success(f"Found job cards using selector: {selector}")
                        break
                except:
                    continue
            
            if not job_cards:
                st.error("Could not find any job listings. LinkedIn might have updated their page structure.")
                return pd.DataFrame(jobs_data)

            # Limit to requested number
            job_cards = job_cards[:job_count]
            
            st.info(f"Processing {len(job_cards)} job cards...")
            
            for card in job_cards:
                try:
                    # Company name selectors
                    company_selectors = [
                        ".job-card-container__company-name",
                        ".base-search-card__subtitle",
                        ".company-name",
                        "span[data-tracking-control-name='public_jobs_company_name']",
                        ".job-card-container__primary-description"
                    ]
                    
                    # Job title selectors
                    title_selectors = [
                        ".job-card-container__title",
                        ".base-search-card__title",
                        ".job-card-list__title",
                        "h3.base-search-card__title",
                        ".job-search-card__title"
                    ]
                    
                    # Location selectors
                    location_selectors = [
                        ".job-card-container__metadata-item",
                        ".base-search-card__metadata",
                        ".job-search-card__location",
                        "span[data-tracking-control-name='public_jobs_job-location']",
                        ".job-card-container__metadata-wrapper"
                    ]
                    
                    # Try to find company name
                    company = None
                    for selector in company_selectors:
                        try:
                            element = card.find_element(by=By.CSS_SELECTOR, value=selector)
                            company = element.text.strip()
                            if company:
                                break
                        except:
                            continue
                    
                    # Try to find job title
                    title = None
                    for selector in title_selectors:
                        try:
                            element = card.find_element(by=By.CSS_SELECTOR, value=selector)
                            title = element.text.strip()
                            if title:
                                break
                        except:
                            continue
                    
                    # Try to find location
                    location = None
                    for selector in location_selectors:
                        try:
                            element = card.find_element(by=By.CSS_SELECTOR, value=selector)
                            location = element.text.strip()
                            if location:
                                break
                        except:
                            continue
                    
                    # Try to find URL
                    try:
                        url = card.find_element(by=By.CSS_SELECTOR, value="a").get_attribute("href")
                    except:
                        try:
                            url = card.find_element(by=By.CSS_SELECTOR, value="a.base-card__full-link").get_attribute("href")
                        except:
                            url = None
                    
                    if all([company, title, location, url]):
                        jobs_data['company_name'].append(company)
                        jobs_data['job_title'].append(title)
                        jobs_data['location'].append(location)
                        jobs_data['job_url'].append(url)
                        st.success(f"Successfully scraped job: {title} at {company}")
                    
                except Exception as e:
                    st.warning(f"Failed to scrape a job card: {str(e)}")
                    continue

            if not jobs_data['company_name']:
                st.error("Could not extract any job information. LinkedIn might be blocking automated access.")
                
        except Exception as e:
            st.error(f"Error during job scraping: {str(e)}")

        return pd.DataFrame(jobs_data)

    @staticmethod
    def display_results(df):
        """Display scraped job results"""
        if df.empty:
            st.error("No jobs were found. Please try again with different search parameters.")
            return

        st.markdown('### 📊 Scraped Job Listings')
        
        # Display summary statistics
        st.markdown(f"**Total Jobs Found:** {len(df)}")
        st.markdown(f"**Unique Companies:** {df['company_name'].nunique()}")
        st.markdown(f"**Locations Covered:** {df['location'].nunique()}")
        
        # Display the dataframe
        st.dataframe(df)
        
        # Add download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Results as CSV",
            csv,
            "linkedin_jobs.csv",
            "text/csv",
            key='download-csv'
        )

    def main():
        st.markdown('## 🔍 LinkedIn Job Search')
        
        job_titles, job_location, job_count = linkedin_scraper.get_userinput()
        
        if st.button('Start Scraping'):
            with st.spinner('Scraping LinkedIn jobs...'):
                try:
                    driver = linkedin_scraper.webdriver_setup()
                    if driver is None:
                        return
                        
                    url = linkedin_scraper.build_url(job_titles, job_location)
                    st.info(f"Searching: {url}")
                    
                    driver.get(url)
                    time.sleep(5)  # Increased initial wait time
                    
                    linkedin_scraper.scroll_page(driver, job_count)
                    df = linkedin_scraper.scrape_jobs(driver, job_count)
                    
                    driver.quit()
                    
                    if not df.empty:
                        linkedin_scraper.display_results(df)
                    else:
                        st.error('No jobs found matching your criteria. Try different search terms or location.')
                        
                except Exception as e:
                    st.error(f'An error occurred while scraping: {str(e)}')
                    if 'driver' in locals():
                        driver.quit()

class career_chatbot:
    def initialize_session_state():
        # Initialize session state variables for the chatbot
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "I'm your Career & Resume Assistant! Ask me anything about job searching, resume writing, interview preparation, or career development."}
            ]
        
        if "conversation_memory" not in st.session_state:
            st.session_state.conversation_memory = ConversationBufferMemory(return_messages=True)
        
        if "resume_data" not in st.session_state:
            st.session_state.resume_data = None
    
    def setup_chatbot_ui():
        with st.container():
            st.markdown(f'<h3 style="color: orange; text-align: center;">Career Advisor Chatbot</h3>', unsafe_allow_html=True)
            
            # Option to upload resume to provide context for the chatbot
            with st.expander("Upload Resume for Context (Optional)"):
                pdf = st.file_uploader(label='Upload Resume', type='pdf', key="chatbot_resume")
                if pdf is not None and st.button("Process Resume"):
                    with st.spinner('Processing resume for context...'):
                        try:
                            pdf_chunks = resume_analyzer.pdf_to_chunks(pdf)
                            summary_prompt = resume_analyzer.summary_prompt(query_with_chunks=pdf_chunks)
                            summary = resume_analyzer.local_llm(chunks=pdf_chunks, analyze=summary_prompt)
                            if summary:
                                st.session_state.resume_data = summary
                                st.success("Resume processed successfully! The chatbot now has context from your resume.")
                        except Exception as e:
                            st.error(f"Error processing resume: {e}")
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
    
    def create_system_prompt():
        base_prompt = """You are a specialized career and job-search assistant. Your expertise is limited to:
1. Resume writing, analysis, and improvement
2. Job search strategies and techniques
3. Interview preparation and tips
4. Career development advice
5. LinkedIn profile optimization
6. Professional networking guidance
7. Salary negotiation tactics
8. Professional skill development recommendations

Answer questions ONLY related to these topics. For any off-topic questions, politely redirect the conversation back to career-related topics.
Your responses should be helpful, specific, and actionable. Use bullet points for clarity when appropriate.
"""
        
        # Add resume context if available
        if st.session_state.resume_data:
            resume_context = f"\nThe user has provided a resume with the following information:\n{st.session_state.resume_data}\n\nUse this context to provide personalized advice when relevant."
            return base_prompt + resume_context
        else:
            return base_prompt
    
    def process_user_input():
        # Get user input and clear the input box
        user_input = st.chat_input("Ask me about careers, job search, or resume advice...")
        
        if user_input:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Generate response using the chatbot
            try:
                with st.spinner("Thinking..."):
                    llm = initialize_llm()
                    if not llm:
                        raise Exception("Failed to initialize LLM")
                    
                    # Update conversation memory
                    st.session_state.conversation_memory.chat_memory.add_user_message(user_input)
                    
                    system_prompt = career_chatbot.create_system_prompt()
                    chat_history = st.session_state.conversation_memory.buffer
                    
                    # Format prompt with system instructions and context
                    prompt = f"""
                    {system_prompt}
                    
                    Chat History: {chat_history}
                    
                    Human: {user_input}
                    Assistant:"""
                    
                    response = llm.predict(prompt)
                    
                    # Add assistant response to memory
                    st.session_state.conversation_memory.chat_memory.add_ai_message(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Display assistant response
                    with st.chat_message("assistant"):
                        st.write(response)
            
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": "I'm sorry, I encountered an error. Please try again."})
    
    def main():
        career_chatbot.initialize_session_state()
        career_chatbot.setup_chatbot_ui()
        career_chatbot.process_user_input()

# Streamlit Configuration Setup
streamlit_config()
add_vertical_space(2)

with st.sidebar:
    add_vertical_space(4)
    option = option_menu(menu_title='', options=['Summary', 'Strength', 'Weakness', 'Job Titles', 'Linkedin Jobs', 'Career Chat'],
                         icons=['house-fill', 'database-fill', 'pass-fill', 'list-ul', 'linkedin', 'chat-dots-fill'])

if option == 'Summary':
    resume_analyzer.resume_summary()
elif option == 'Strength':
    resume_analyzer.resume_strength()
elif option == 'Weakness':
    resume_analyzer.resume_weakness()
elif option == 'Job Titles':
    resume_analyzer.job_title_suggestion()
elif option == 'Linkedin Jobs':
    linkedin_scraper.main()
elif option == 'Career Chat':
    career_chatbot.main()