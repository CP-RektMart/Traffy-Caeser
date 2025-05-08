import streamlit as st
from google.cloud import aiplatform
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel


@st.cache_resource
def initialize_vertex_ai():
    """Initialize Vertex AI with GCP credentials from streamlit secrets."""
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )

    project_id = st.secrets["PROJECT_ID"]
    location = "us-central1"  # Change to your preferred region

    # Initialize Vertex AI
    vertexai.init(project=project_id, location=location, credentials=credentials)

    return True


@st.cache_data(ttl=3600)  # Cache for 1 hour
def generate_cluster_summary(comments_list):
    """Generate a summary for a cluster based on sample comments using Gemini model.

    Args:
        comments_list: List of comment texts from the cluster

    Returns:
        A string summary of the cluster
    """
    # Make sure Vertex AI is initialized
    _ = initialize_vertex_ai()

    # Prepare the prompt with the comments
    comments_text = "\n".join([f"- {comment}" for comment in comments_list])
    prompt = f"""จากข้อมูลความคิดเห็นที่ให้มา จงสร้างสรุปสำหรับกลุ่มความคิดเห็นนี้
    โดยเน้นที่รูปแบบที่ปรากฏ สรุปนี้ควรมีความยาวไม่เกิน 2 ประโยค และต้องสื่อถึงใจความที่แสดงในกลุ่มนี้อย่างถูกต้อง
    ให้ตอบเป็นภาษาไทยเท่านั้น ห้ามใช้อีโมจิหรือสัญลักษณ์พิเศษอื่น ๆ
    สรุปควรมีความกระชับและชัดเจน ไม่ต้องมีการทบทวนหรือขึ้นต้นด้วยคำว่า "จากความคิดเห็นที่ให้มา"
    เน้นที่ประเด็นหลักและใจความสำคัญของความคิดเห็นในกลุ่มนี้
    
    ข้อมูลความคิดเห็น:
    {comments_text}
    """

    try:
        # Load the model
        model = GenerativeModel("gemini-2.5-flash-preview-04-17")

        # Generate content
        response = model.generate_content(prompt)

        # Return the generated summary
        return response.text.strip()
    except Exception as e:
        st.warning(f"Failed to generate summary for cluster: {e}")
        return "Summary not available"
