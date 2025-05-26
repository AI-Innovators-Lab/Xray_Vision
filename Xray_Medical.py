import streamlit as st
from PIL import Image as PILImage # Renamed to avoid conflict
import io
import base64
import requests
from openai import OpenAI
import json
import os
from datetime import datetime
from fpdf import FPDF # For PDF Generation
from urllib.parse import quote_plus # For Google Search URL
import re
import tempfile # For temporary file handling


# --- Constants ---
LLAMA_VISION_INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
PALMYRA_BASE_URL = "https://integrate.api.nvidia.com/v1"
LLAMA_VISION_MODEL = "meta/llama-3.2-90b-vision-instruct"
PALMYRA_MODEL = "writer/palmyra-med-70b"
MAX_ENCODED_IMAGE_SIZE_BYTES = 770000 # Approx 770KB for base64
APPROX_ORIGINAL_IMAGE_LIMIT_KB = int(MAX_ENCODED_IMAGE_SIZE_BYTES / (1000 * 1.33)) # Approx 580KB original
APP_NAME = "X-ray AI Insights Navigator"
APP_LOGO_PATH = "logo.png" # Optional: Place your logo.png in the same directory or set to None
REPORT_DISCLAIMER_TEXT = (
    "IMPORTANT DISCLAIMER: This tool uses AI models for image analysis and text generation. "
    "The information provided is for educational and informational purposes ONLY and is NOT a substitute "
    "for professional medical advice, diagnosis, or treatment. AI interpretations can have errors or limitations. "
    "Always consult with a qualified healthcare professional (e.g., your doctor, a radiologist) for any "
    "medical concerns or interpretation of your X-rays. Do not rely on this tool for medical decisions."
)
SAMPLE_IMAGE_FILENAME = "sample_xray.jpg" # Create this file or change path
SAMPLE_IMAGE_PATH = os.path.join(os.path.dirname(__file__), SAMPLE_IMAGE_FILENAME)


# --- Helper Functions ---
def encode_image_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode()

def format_vlm_output_for_display(vlm_raw_text): # For screen display with emojis
    if not vlm_raw_text or "X-RAY IMAGE ANALYSIS OUTPUT:" not in vlm_raw_text:
        return "❌ No analysis data available or format is unexpected."
    lines = vlm_raw_text.splitlines()
    formatted_lines = []
    in_analysis_section = False
    section_emojis = {
        "Anatomical Region": "🔍",
        "Observed Structures & Condition": "🦴",
        "Detailed Radiological Findings": "📋",
        "Overall Impression of Visual Findings": "📊"
    }
    for line in lines:
        stripped_line = line.strip()
        if stripped_line == "**X-RAY IMAGE ANALYSIS OUTPUT:**":
            in_analysis_section = True
            continue
        if not in_analysis_section: continue
        if stripped_line.startswith("*   **") and stripped_line.endswith(":**"):
            section_title_raw = stripped_line.replace("*   **", "").replace(":**", "").strip()
            emoji = section_emojis.get(section_title_raw, "?")
            formatted_lines.append(f"\n#### {emoji} {section_title_raw}\n")
        elif stripped_line.startswith("*   "):
            item_text = stripped_line[len("*   "):].strip()
            if item_text.startswith("**") and "**:" in item_text:
                 formatted_lines.append(f"- {item_text}")
            else:
                 formatted_lines.append(f"- {item_text}")
        elif stripped_line:
            formatted_lines.append(f"{stripped_line}\n")
    return "\n".join(formatted_lines)

def generate_xray_report_pdf(
    xray_image_bytes,
    original_filename,
    raw_vlm_output,
    palmyra_output=None
):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    default_font = "Helvetica" # Using FPDF's core font

    # 1. Header
    page_width = pdf.w - 2 * pdf.l_margin
    logo_displayed = False
    if APP_LOGO_PATH and os.path.exists(APP_LOGO_PATH):
        try:
            with PILImage.open(APP_LOGO_PATH) as _:
                pass
            pdf.image(APP_LOGO_PATH, x=pdf.l_margin, y=10, w=30)
            logo_displayed = True
        except Exception as e:
            print(f"PDF Warning: Error loading or adding logo '{APP_LOGO_PATH}': {e}")

    pdf.set_font(default_font, "B", 16)
    app_name_x_pos = pdf.l_margin + 35 if logo_displayed else pdf.l_margin
    app_name_width_adj = 35 if logo_displayed else 0
    pdf.set_xy(app_name_x_pos, 15)
    # Remove emojis from app name for PDF
    app_name_clean = "X-ray AI Insights Navigator"
    pdf.cell(page_width - app_name_width_adj, 10, app_name_clean, 0, 1, "C")

    pdf.set_font(default_font, "", 10)
    pdf.cell(0, 10, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, "R")
    pdf.ln(10)

    # 2. X-ray Image Section
    pdf.set_font(default_font, "B", 12)
    pdf.cell(0, 10, f"Analyzed Image: {original_filename}", 0, 1, "L")
    temp_image_file_path = None
    try:
        pil_image = PILImage.open(io.BytesIO(xray_image_bytes))
        original_format = pil_image.format
        save_format_pil = "PNG"
        temp_file_suffix = ".png"

        if original_format in ["JPEG", "JPG"]:
            save_format_pil = "JPEG"
            temp_file_suffix = ".jpg"
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
        elif original_format == "PNG":
            save_format_pil = "PNG"
            temp_file_suffix = ".png"
        elif original_format == "GIF": # FPDF supports GIF
            save_format_pil = "GIF"
            temp_file_suffix = ".gif"
        else:
            if pil_image.mode == "RGBA" or pil_image.mode == "P" and "transparency" in pil_image.info:
                 pil_image = pil_image.convert("RGBA")
            else:
                 pil_image = pil_image.convert("RGB")

        with tempfile.NamedTemporaryFile(delete=False, suffix=temp_file_suffix) as tmpfile:
            pil_image.save(tmpfile, format=save_format_pil)
            temp_image_file_path = tmpfile.name
        
        img_width_pil, img_height_pil = pil_image.size
        page_content_width = pdf.w - 2 * pdf.l_margin
        available_width = page_content_width * 0.9
        aspect_ratio = img_height_pil / img_width_pil if img_width_pil > 0 else 1
        display_width = available_width
        display_height = display_width * aspect_ratio
        max_display_height = pdf.h * 0.40
        if display_height > max_display_height:
            display_height = max_display_height
            display_width = display_height / aspect_ratio if aspect_ratio > 0 else display_height
        x_pos = (pdf.w - display_width) / 2
        pdf.image(temp_image_file_path, x=x_pos, w=display_width, h=display_height)
        pdf.ln(5)

    except Exception as e:
        pdf.set_font(default_font, "I", 10)
        error_message = f"(Error displaying X-ray image in PDF. Details: {type(e).__name__} - {str(e)})"
        pdf.multi_cell(0, 5, error_message)
        print(f"PDF Image Processing CRITICAL Error: {type(e).__name__} - {str(e)}")
    finally:
        if temp_image_file_path and os.path.exists(temp_image_file_path):
            try:
                os.remove(temp_image_file_path)
            except Exception as e_del:
                print(f"PDF Warning: Could not remove temporary file {temp_image_file_path}. Error: {e_del}")
    pdf.ln(5)

    def clean_raw_llm_text_for_pdf(raw_text):
        if not raw_text: return ""
        
        # Remove emojis and non-latin-1 characters
        import re
        # Remove emojis and other Unicode characters that aren't in latin-1
        cleaned_text = re.sub(r'[^\x00-\xFF]', '', raw_text)
        
        # Remove markdown formatting
        cleaned_text = cleaned_text.replace("**", "")
        
        processed_lines = []
        if "X-RAY IMAGE ANALYSIS OUTPUT:" in cleaned_text:
            lines = cleaned_text.splitlines()
            in_analysis_section = False
            for line in lines:
                stripped_line = line.strip()
                if stripped_line == "X-RAY IMAGE ANALYSIS OUTPUT:":
                    in_analysis_section = True
                    continue
                if not in_analysis_section and "X-RAY IMAGE ANALYSIS OUTPUT:" in raw_text: 
                    continue
                if stripped_line.startswith("*   "):
                    item_content = stripped_line[len("*   "):].strip()
                    if item_content.endswith(":"):
                        processed_lines.append(f"\n{item_content}")
                    else:
                        processed_lines.append(f"  - {item_content}")
                elif stripped_line:
                    processed_lines.append(stripped_line)
            cleaned_text = "\n".join(processed_lines)
        
        # Final encoding check and replacement
        try:
            # Test if it can be encoded to latin-1
            cleaned_text.encode('latin-1')
            return cleaned_text
        except UnicodeEncodeError:
            # If still fails, use more aggressive cleaning
            return cleaned_text.encode('latin-1', 'replace').decode('latin-1')

    vlm_pdf_text = clean_raw_llm_text_for_pdf(raw_vlm_output)
    pdf.set_font(default_font, "B", 12)
    pdf.cell(0, 10, "AI Visual Analysis (Vision Model)", 0, 1, "L")
    pdf.set_font(default_font, "", 10)
    pdf.multi_cell(0, 5, vlm_pdf_text)
    pdf.ln(5)

    if palmyra_output:
        palmyra_pdf_text = clean_raw_llm_text_for_pdf(palmyra_output)
        pdf.set_font(default_font, "B", 12)
        pdf.cell(0, 10, "Patient-Friendly Explanation & Suggestions", 0, 1, "L")
        pdf.set_font(default_font, "", 10)
        pdf.multi_cell(0, 5, palmyra_pdf_text)
        pdf.ln(5)
        
    # Clean disclaimer text as well
    disclaimer_pdf_text = clean_raw_llm_text_for_pdf(REPORT_DISCLAIMER_TEXT)
    pdf.set_font(default_font, "I", 8)
    pdf.multi_cell(0, 4, disclaimer_pdf_text)
    pdf.ln(5)

    return pdf.output(dest='S').encode('latin-1')


# --- API Call Functions (Cached) ---
@st.cache_data(show_spinner=False)
def call_llama_3_2_90b_vision_cached(_api_key_for_cache, image_b64_string, custom_prompt_text):
    headers = {"Authorization": f"Bearer {_api_key_for_cache}", "Accept": "text/event-stream", "Content-Type": "application/json"}
    image_type = "png"
    if image_b64_string.startswith("/9j/"): image_type = "jpeg"
    elif image_b64_string.startswith("iVBOR"): image_type = "png"
    payload = {"model": LLAMA_VISION_MODEL, "messages": [{"role": "user", "content": f'{custom_prompt_text} <img src="data:image/{image_type};base64,{image_b64_string}" />'}], "max_tokens": 1024, "temperature": 0.50, "top_p": 0.70, "stream": True}
    try:
        response_stream = requests.post(LLAMA_VISION_INVOKE_URL, headers=headers, json=payload, stream=True)
        response_stream.raise_for_status()
        full_response_vlm = ""
        for line in response_stream.iter_lines():
            if line:
                decoded_line = line.decode("utf-8").strip()
                if decoded_line.startswith("data:"):
                    json_data_str = decoded_line[len("data:"):].strip()
                    if json_data_str == "[DONE]": break
                    try:
                        chunk_data = json.loads(json_data_str)
                        if chunk_data.get("choices") and len(chunk_data["choices"]) > 0 and chunk_data["choices"][0].get("delta"):
                            content = chunk_data["choices"][0]["delta"].get("content", "")
                            full_response_vlm += content
                    except json.JSONDecodeError: pass
        return full_response_vlm.strip()
    except requests.exceptions.RequestException as e:
        error_message = f"❌ Error calling Llama 3.2 Vision API: {e}"
        if hasattr(e, 'response') and e.response is not None:
            try: error_message += f"\nAPI Response: {e.response.json()}"
            except ValueError: error_message += f"\nAPI Response (text): {e.response.text}"
        st.session_state.vlm_api_error = error_message
        return None

@st.cache_data(show_spinner=False)
def call_palmyra_med_70b_cached(_api_key_for_cache, previous_llm_output, custom_prompt_text_for_remediation):
    try:
        client = OpenAI(base_url=PALMYRA_BASE_URL, api_key=_api_key_for_cache)
        stream = client.chat.completions.create(model=PALMYRA_MODEL, messages=[{"role": "system", "content": "You are an AI medical information assistant designed to explain X-ray findings to a patient in simple terms. Your tone should be empathetic and informative. You are NOT a doctor and cannot give medical advice or a diagnosis. Your explanations should empower the patient to have a more informed discussion with their healthcare provider. Always include the crucial disclaimer provided."}, {"role": "user", "content": f"{custom_prompt_text_for_remediation}\n\n--- AI X-RAY IMAGE ANALYSIS ---\n{previous_llm_output}\n--- END AI X-RAY IMAGE ANALYSIS ---"}], temperature=0.3, top_p=0.7, max_tokens=1500, stream=True)
        full_response_palmyra = []
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                full_response_palmyra.append(chunk.choices[0].delta.content)
        return "".join(full_response_palmyra).strip()
    except Exception as e:
        st.session_state.palmyra_api_error = f"❌ Error calling Palmyra-Med-70b API (Explanation): {e}"
        return None

@st.cache_data(show_spinner=False)
def call_palmyra_chat_cached(_api_key_for_cache, messages_history):
    """
    Calls Palmyra-Med-70b for general medical chat.
    messages_history should be a list of dicts, e.g.,
    [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    """
    try:
        client = OpenAI(base_url=PALMYRA_BASE_URL, api_key=_api_key_for_cache)
        stream = client.chat.completions.create(
            model=PALMYRA_MODEL,
            messages=messages_history,
            temperature=0.5,
            top_p=0.7,
            max_tokens=1000,
            stream=True
        )
        full_response_palmyra_chat = []
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                full_response_palmyra_chat.append(chunk.choices[0].delta.content)
        return "".join(full_response_palmyra_chat).strip()
    except Exception as e:
        st.session_state.palmyra_chat_api_error = f"❌ Error in Palmyra Chat: {e}" # Specific error state for chat
        return None

# --- Prompts ---
PROMPT_FOR_VLM_IMAGE_ANALYSIS = """
You are an AI assistant specialized in radiological image analysis.
Your task is to meticulously analyze this X-ray image and provide a structured description of your observations in a way that can be understood by another medical AI for further explanation.
 
1.  **Anatomical Region Identification:**
    *   Clearly state the primary anatomical region and view (e.g., "PA Chest X-ray," "AP view of the left knee," "Lateral view of the cervical spine").
 
2.  **Observed Structures & Their Condition:**
    *   Describe the appearance of the key anatomical structures visible (e.g., bones, joints, lungs, heart silhouette).
    *   Note if they appear normal or if any deviations are observed.
 
3.  **Detailed Radiological Findings (if any):**
    *   Meticulously describe any observable abnormalities, lesions, fractures (type, location), dislocations, opacities, lucencies, misalignments, degenerative changes (e.g., osteophytes, joint space narrowing), foreign bodies, or other deviations from normal radiological appearance.
    *   Use precise medical terminology where appropriate. For each finding, try to describe its location, characteristics (size, shape, density, margins if applicable), and extent.
 
4.  **Overall Impression of Visual Findings:**
    *   Based *only* on the visual evidence, provide a concise summary of the most significant radiological observations.
 
**Output Format for this stage (strictly adhere to this for subsequent processing):**
 
**X-RAY IMAGE ANALYSIS OUTPUT:**
*   **Anatomical Region:** [Your identification]
*   **Observed Structures & Condition:**
    *   [Structure 1]: [Description]
    *   [Structure 2]: [Description]
    *   ...
*   **Detailed Radiological Findings:**
    *   [Finding 1: Name of finding, Location, Detailed Description, Characteristics]
    *   [Finding 2: Name of finding, Location, Detailed Description, Characteristics]
    *   ... (List "None observed" if no abnormalities are clearly visible)
*   **Overall Impression of Visual Findings:** [Your summary]
 
**Important:** Your analysis is for preliminary informational purposes based on visual interpretation.
It is NOT a definitive diagnosis.
"""

PROMPT_FOR_PALMYRA_REMEDIATION = """
Based *solely* on the "Detailed Radiological Findings" and "Overall Impression of Visual Findings" provided in the AI X-RAY IMAGE ANALYSIS below, please generate a patient-friendly explanation structured as follows:
 
**1. Understanding What Your X-ray Image Might Show (based on AI analysis):**
    *   Based on the "Anatomical Region" identified by the first AI, briefly explain what this type of X-ray typically looks at.
    *   Explain key medical terms mentioned in the "Detailed Radiological Findings" in simple terms.
 
**2. Summary of Potential Key Observations by the First AI:**
    *   In plain language, summarize the main potential findings described by the first AI.
 
**3. What These Potential AI Observations Might Mean (General Implications):**
    *   Based on the "Overall Impression of Visual Findings" and "Detailed Radiological Findings," explain what these observations *might generally* suggest.
    *   Emphasize that this is an interpretation of visual patterns by an AI and requires confirmation and context from a human doctor.
 
**4. General Remediation Considerations & Next Steps (NOT Medical Advice):**
    *   What are *general* types of approaches often discussed for the kinds of potential issues suggested by the AI's visual findings? (e.g., "If findings like these are confirmed by your doctor, they might discuss pain management, physical therapy, or lifestyle adjustments.").
    *   What kind of specialist might be relevant if these findings are significant?
    *   Stress the absolute necessity of consulting their own doctor for diagnosis and treatment.
**5. Suggested Specialist (if applicable):**
   *   Based on the findings from the AI X-RAY IMAGE ANALYSIS, what type of medical specialist might typically be consulted?
   *   Please state this clearly. For example: "For these types of findings, you might consider consulting an **Orthopedist**."
   *   If findings are general or appear normal according to the AI analysis, suggest consulting "your **General Practitioner** or family doctor."
   *   If a specialist is clearly indicated by the AI's findings, state the specialty.
   *   If it's not clear or the AI cannot determine a specific specialist from the findings, suggest "your **General Practitioner**."
   *   **Crucially, if a specialist is named, please try to use the format: "A specialist such as an **[Name of Specialty]** may be appropriate." or "Consider consulting an **[Name of Specialty]**."**
 
**6. Questions to Ask Your Doctor About the X-ray Image:**
    *   Provide a list of 3-5 sample questions the patient could ask their doctor based on the AI's visual interpretation.
 
**IMPORTANT DISCLAIMER (Include this verbatim at the beginning and end of your ENTIRE response):**
"This information is based on an AI's interpretation of an X-ray image and is for educational purposes only.
It is NOT a medical diagnosis or medical advice and DOES NOT replace examination and interpretation by a qualified radiologist and consultation with your physician.
An AI's visual interpretation can have limitations and errors. Your doctor is the only one who can provide a diagnosis and treatment plan after considering all your medical information.
Always discuss your health with your physician."
"""

CHAT_SYSTEM_PROMPT_PALMYRA = """
You are an AI medical information assistant. Your role is to provide helpful, empathetic, and informative responses to general medical questions.
You are NOT a doctor and CANNOT provide medical advice, diagnosis, or treatment plans.
Your primary goal is to educate and empower users to discuss health concerns with qualified healthcare professionals.

IMPORTANT INSTRUCTIONS:
1.  Be clear, concise, and use language that is easy for a layperson to understand.
2.  Maintain an empathetic and supportive tone.
3.  If a question asks for diagnosis, medical advice, or treatment, you MUST politely decline and firmly state that you cannot provide such information. Instead, recommend consulting a doctor or qualified healthcare provider.
4.  ALWAYS include the following disclaimer at the end of EVERY response you generate:
    "DISCLAIMER: This information is for educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something you have read or heard from this AI assistant."
5.  Do not make up information. If you don't know the answer to something, say so.
"""

# --- Streamlit App Interface ---
st.set_page_config(layout="wide", page_title="🩺 X-ray AI Insights")
st.markdown(f"<h1 style='text-align: center;'>🩺{APP_NAME}</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>✨AI-assisted interpretation for educational purposes. Not a substitute for professional medical advice.</p>", unsafe_allow_html=True)


# Initialize session state variables
if 'vlm_analysis_output' not in st.session_state: st.session_state.vlm_analysis_output = ""
if 'palmyra_output' not in st.session_state: st.session_state.palmyra_output = ""
if 'uploaded_filename' not in st.session_state: st.session_state.uploaded_filename = None
if 'vlm_analysis_running' not in st.session_state: st.session_state.vlm_analysis_running = False
if 'palmyra_running' not in st.session_state: st.session_state.palmyra_running = False
if 'sample_image_loaded' not in st.session_state: st.session_state.sample_image_loaded = False
if 'uploaded_file_bytes' not in st.session_state: st.session_state.uploaded_file_bytes = None
if 'vlm_api_error' not in st.session_state: st.session_state.vlm_api_error = None
if 'palmyra_api_error' not in st.session_state: st.session_state.palmyra_api_error = None
# New session states for chat
if 'palmyra_chat_messages' not in st.session_state: st.session_state.palmyra_chat_messages = []
if 'palmyra_chat_api_error' not in st.session_state: st.session_state.palmyra_chat_api_error = None
if 'show_chat_popover' not in st.session_state: st.session_state.show_chat_popover = False

def reset_all_states():
    st.session_state.vlm_analysis_output = ""
    st.session_state.palmyra_output = ""
    st.session_state.vlm_analysis_running = False
    st.session_state.palmyra_running = False
    st.session_state.uploaded_filename = None
    st.session_state.sample_image_loaded = False
    st.session_state.uploaded_file_bytes = None
    st.session_state.vlm_api_error = None
    st.session_state.palmyra_api_error = None
    # Reset chat states
    st.session_state.palmyra_chat_messages = []
    st.session_state.palmyra_chat_api_error = None
    # Do not reset show_chat_popover here, as it's controlled by its button explicitly.
    # If you want "Start Over" to also close the chat, then uncomment:
    # st.session_state.show_chat_popover = False


# --- Sidebar ---
if APP_LOGO_PATH and os.path.exists(APP_LOGO_PATH):
    st.sidebar.image(APP_LOGO_PATH, width=150)
else:
    st.sidebar.markdown("## 📋 Menu")
st.sidebar.header("🔑 NVIDIA API Access")
api_key_input = st.sidebar.text_input("Enter your NVIDIA API Key:", type="password", help="Your API key is used to access the AI models. It is not stored.")


NVIDIA_API_GUIDE_LINK = "https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html" # <-- **** REPLACE WITH THE BEST OFFICIAL LINK ****
st.sidebar.markdown(
    f"<small>🔑 Don't have an API key? [Learn how to get one from NVIDIA]({NVIDIA_API_GUIDE_LINK})</small>",
    unsafe_allow_html=True
)
st.sidebar.divider()
st.sidebar.subheader("💬 Medical Q&A")
if st.sidebar.button("🩺 Chat about Medical Topics", use_container_width=True, key="toggle_chat_popover_btn"):
    st.session_state.show_chat_popover = not st.session_state.show_chat_popover
    # If opening the popover, ensure chat history doesn't persist across different "sessions" of opening it, unless desired.
    # If you want chat to clear every time popover is opened:
    # if st.session_state.show_chat_popover:
    #     st.session_state.palmyra_chat_messages = []
    #     st.session_state.palmyra_chat_api_error = None


# --- Main Page Content ---
st.warning(f"⚠️ {REPORT_DISCLAIMER_TEXT}")
st.divider()

# --- Chat Popover Logic (conditionally rendered) ---
if st.session_state.show_chat_popover:
    with st.popover("💬 Chat with Medical AI", use_container_width=False): # Set use_container_width to False or adjust width for better popover sizing
        st.markdown("#### 🤖 Ask Medical Assistant")
        st.caption("This is a general medical Q&A. For X-ray analysis, please use the main interface below.")
        st.markdown("---")

        if not api_key_input:
            st.warning("⚠️ Please enter your NVIDIA API Key in the sidebar to use the chat.")
        else:
            # Display existing chat messages
            for msg in st.session_state.palmyra_chat_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"]) # Use markdown for potentially formatted responses

            # Chat input
            if user_chat_prompt := st.chat_input("❓ Ask a medical question... (e.g., 'What are common symptoms of flu?')", key="palmyra_chat_input"):
                st.session_state.palmyra_chat_messages.append({"role": "user", "content": user_chat_prompt})
                with st.chat_message("user"):
                    st.markdown(user_chat_prompt)

                # Prepare messages for API
                # The system prompt is the very first message in the conversation.
                api_messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT_PALMYRA}] + st.session_state.palmyra_chat_messages

                with st.spinner("🤖 Palmyra-Med-70b is thinking..."):
                    st.session_state.palmyra_chat_api_error = None # Clear previous error
                    ai_response = call_palmyra_chat_cached(api_key_input, api_messages)

                if st.session_state.palmyra_chat_api_error:
                    st.error(st.session_state.palmyra_chat_api_error)
                elif ai_response:
                    st.session_state.palmyra_chat_messages.append({"role": "assistant", "content": ai_response})
                    with st.chat_message("assistant"):
                        st.markdown(ai_response) # Use markdown for AI responses
                else:
                    st.error("❌ AI did not return a response for the chat.")
                st.rerun() # Rerun to update message display immediately
            
            if st.session_state.palmyra_chat_api_error and not user_chat_prompt: # Show persistent error if any
                 st.error(st.session_state.palmyra_chat_api_error)

            if st.session_state.palmyra_chat_messages: # Show clear button only if there are messages
                if st.button("🗑️ Clear Chat History", key="clear_chat_hist_btn_popover"):
                    st.session_state.palmyra_chat_messages = []
                    st.session_state.palmyra_chat_api_error = None
                    st.rerun()

# --- X-ray Analysis Workflow ---
# Sample Image Button - moved here to be part of main flow logic if desired, or keep in sidebar
# if st.sidebar.button("?? Use Sample X-ray Image"): # Or place this button in the main area
#     if os.path.exists(SAMPLE_IMAGE_PATH):
#         with open(SAMPLE_IMAGE_PATH, "rb") as f:
#             reset_all_states() # This will also close popover if you enable that line in reset_all_states
#             st.session_state.uploaded_file_bytes = f.read()
#             st.session_state.uploaded_filename = SAMPLE_IMAGE_FILENAME
#             st.session_state.sample_image_loaded = True
#             st.rerun()
#     else:
#         st.sidebar.error(f"Sample image '{SAMPLE_IMAGE_FILENAME}' not found.")
#         st.session_state.sample_image_loaded = False

uploaded_file_obj = st.file_uploader("📁 Upload your X-ray image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key="file_uploader_widget")

current_image_bytes = None
current_filename = None

if uploaded_file_obj is not None:
    if st.session_state.uploaded_filename != uploaded_file_obj.name or st.session_state.sample_image_loaded:
        reset_all_states()
        st.session_state.uploaded_filename = uploaded_file_obj.name
        st.session_state.uploaded_file_bytes = uploaded_file_obj.getvalue()
        st.session_state.sample_image_loaded = False
    current_image_bytes = st.session_state.uploaded_file_bytes
    current_filename = st.session_state.uploaded_filename
elif st.session_state.sample_image_loaded and st.session_state.uploaded_file_bytes: # For sample image
    current_image_bytes = st.session_state.uploaded_file_bytes
    current_filename = st.session_state.uploaded_filename

if current_image_bytes is None:
    st.info("ℹ️ Please upload an X-ray image to begin exploring AI-assisted insights.")
else:
    try:
        image_pil = PILImage.open(io.BytesIO(current_image_bytes))
        st.sidebar.subheader("🩻 Current X-ray Image")
        st.sidebar.image(image_pil, caption=f"Current: {current_filename}", use_container_width=True)
    except Exception as e:
        st.error(f"❌ Could not display uploaded image. It might be corrupted. Error: {e}")
        current_image_bytes = None

    if current_image_bytes:
        if not api_key_input:
            st.error("❌ Please enter your NVIDIA API Key in the sidebar to activate AI analysis.")
        else:
            processing_disabled = st.session_state.vlm_analysis_running or st.session_state.palmyra_running

            if not st.session_state.vlm_analysis_output and not st.session_state.vlm_analysis_running:
                if st.button("🔬 Analyze X-ray (Vision AI)", type="primary", use_container_width=True, disabled=processing_disabled):
                    st.session_state.vlm_analysis_running = True
                    st.session_state.palmyra_output = "" # Clear previous explanation
                    st.session_state.vlm_api_error = None
                    st.session_state.palmyra_api_error = None
                    st.rerun()

            if st.session_state.vlm_analysis_running and not st.session_state.vlm_analysis_output:
                with st.spinner("🔍 AI is performing visual analysis... This may take a moment."):
                    image_b64_string = encode_image_to_base64(current_image_bytes)
                    if len(image_b64_string) >= MAX_ENCODED_IMAGE_SIZE_BYTES:
                        st.error(f"❌ Image too large (>{MAX_ENCODED_IMAGE_SIZE_BYTES/1000:.0f}KB encoded). Max original size approx {APPROX_ORIGINAL_IMAGE_LIMIT_KB}KB.")
                        st.session_state.vlm_analysis_running = False
                    else:
                        full_response_vlm = call_llama_3_2_90b_vision_cached(api_key_input, image_b64_string, PROMPT_FOR_VLM_IMAGE_ANALYSIS)
                        if st.session_state.vlm_api_error:
                            st.error(st.session_state.vlm_api_error)
                        elif full_response_vlm is not None:
                            st.session_state.vlm_analysis_output = full_response_vlm
                            if not st.session_state.vlm_analysis_output: st.error("❌ Received empty analysis from Vision AI.")
                        else: # Should be caught by vlm_api_error, but as a fallback
                            if not st.session_state.vlm_api_error: st.error("❌ Vision AI call failed without specific error.")
                        st.session_state.vlm_analysis_output = st.session_state.vlm_analysis_output or "" # Ensure it's a string
                    st.session_state.vlm_analysis_running = False
                    st.rerun()

            if st.session_state.vlm_analysis_output:
                with st.expander("🔬 AI's Visual Analysis (Vision Model)", expanded=True):
                    formatted_vlm_output_for_screen = format_vlm_output_for_display(st.session_state.vlm_analysis_output)
                    st.markdown(formatted_vlm_output_for_screen)
                    try:
                        pdf_data_vlm = generate_xray_report_pdf(xray_image_bytes=current_image_bytes, original_filename=current_filename, raw_vlm_output=st.session_state.vlm_analysis_output, palmyra_output=None)
                        st.download_button(label="📄 Download Vision AI Analysis (PDF)", data=pdf_data_vlm, file_name=f"vision_ai_report_{current_filename or 'sample'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf", use_container_width=True)
                    except Exception as e: st.error(f"❌ Failed to generate Vision AI PDF report: {e}")

                st.divider()

                if not st.session_state.palmyra_output and not st.session_state.palmyra_running:
                    if st.button("🩺 Get Patient-Friendly Explanation (Medical AI)", type="primary", use_container_width=True, disabled=processing_disabled or not st.session_state.vlm_analysis_output):
                        st.session_state.palmyra_running = True
                        st.session_state.palmyra_api_error = None
                        st.rerun()

                if st.session_state.palmyra_running and not st.session_state.palmyra_output:
                    with st.spinner("🩺 AI (Medical) is generating a patient-friendly explanation..."):
                        full_response_palmyra = call_palmyra_med_70b_cached(api_key_input, st.session_state.vlm_analysis_output, PROMPT_FOR_PALMYRA_REMEDIATION)
                        if st.session_state.palmyra_api_error:
                            st.error(st.session_state.palmyra_api_error)
                        elif full_response_palmyra is not None:
                            st.session_state.palmyra_output = full_response_palmyra
                            if not st.session_state.palmyra_output: st.error("❌ Received empty explanation from Medical AI.")
                        else: # Should be caught by palmyra_api_error
                            if not st.session_state.palmyra_api_error: st.error("❌ Medical AI call failed without specific error.")
                        st.session_state.palmyra_output = st.session_state.palmyra_output or "" # Ensure it's a string
                    st.session_state.palmyra_running = False
                    st.rerun()

                if st.session_state.palmyra_output:
                    with st.expander("👨‍⚕️ Patient-Friendly Explanation & Suggestions (Medical AI)", expanded=True):
                        st.markdown(st.session_state.palmyra_output)
                        try:
                            pdf_data_full = generate_xray_report_pdf(xray_image_bytes=current_image_bytes, original_filename=current_filename, raw_vlm_output=st.session_state.vlm_analysis_output, palmyra_output=st.session_state.palmyra_output)
                            st.download_button(label="📋 Download Full AI Report (PDF)", data=pdf_data_full, file_name=f"full_ai_xray_report_{current_filename or 'sample'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf", use_container_width=True)
                        except Exception as e: st.error(f"❌ Failed to generate full PDF report: {e}")

                    st.divider()
                    st.subheader("🏥 Find a Specialist (via Google Search)")
                    extracted_specialty = "doctor"
                    if st.session_state.palmyra_output:
                        palmyra_text_lower = st.session_state.palmyra_output.lower()
                        patterns_to_check = [
                            r"a specialist such as an \*\*(.*?)\*\*", r"a specialist such as a \*\*(.*?)\*\*",
                            r"consulting an \*\*(.*?)\*\*", r"consulting a \*\*(.*?)\*\*",
                            r"specialist like an \*\*(.*?)\*\*", r"specialist like a \*\*(.*?)\*\*",
                            r"suggested specialist: \*\*(.*?)\*\*", r"suggested specialist: (.*?)",
                        ]
                        found_it = False
                        for pattern in patterns_to_check:
                            match = re.search(pattern, palmyra_text_lower, re.IGNORECASE)
                            if match:
                                potential_specialty = match.group(1).strip().title()
                                if potential_specialty.endswith('.'): potential_specialty = potential_specialty[:-1].strip()
                                if potential_specialty and potential_specialty.lower() not in ["none", "your doctor"]:
                                    extracted_specialty = potential_specialty
                                    found_it = True
                                    break
                        if not found_it:
                            if "general practitioner" in palmyra_text_lower or \
                               "family doctor" in palmyra_text_lower or \
                               "your doctor" in palmyra_text_lower:
                                extracted_specialty = "General Practitioner"

                    st.write(f"Based on the AI's insights, consulting with a **{extracted_specialty}** might be beneficial.")
                    user_location_input = st.text_input(
                        f"📍 Enter your City, Postal Code, or general area to search:",
                        placeholder=f"e.g., New York, NY or 10001",
                        key="doctor_finder_location_input_gs_final_v3"
                    )
                    if user_location_input:
                        search_query = f"{extracted_specialty} near {user_location_input}"
                        google_search_url = f"https://www.google.com/search?q={quote_plus(search_query)}" # Corrected https
                        st.link_button(
                            f"🔍 Search for {extracted_specialty}s in '{user_location_input}' on Google",
                            google_search_url, use_container_width=True, type="primary"
                        )
                        st.caption("This will open Google Search in a new tab.")
                    else:
                        st.info("ℹ️ Enter a location above to enable the 'Find a Specialist' search link.")

            if st.session_state.vlm_analysis_output or st.session_state.palmyra_output:
                if st.button("🔄 Start Over / New Image", use_container_width=True, key="start_over_main_btn"):
                    reset_all_states()
                    st.query_params.clear()
                    st.rerun()

st.divider()
st.markdown("<p style='text-align: center; color: grey;'>🤖 This AI tool is for educational demonstration only. Not for clinical use. Always consult a medical professional.</p>", unsafe_allow_html=True)
