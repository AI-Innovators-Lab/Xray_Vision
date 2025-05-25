# X-ray AI Insights Navigator 

An AI-powered web application for analyzing X-ray images using NVIDIA's advanced vision and medical language models. This tool provides educational insights and patient-friendly explanations of X-ray findings.

## Important Disclaimer

**This tool is for educational and informational purposes ONLY.** It is NOT a substitute for professional medical advice, diagnosis, or treatment. AI interpretations can have errors or limitations. Always consult with a qualified healthcare professional for any medical concerns or interpretation of your X-rays.

## Features

-  **AI Visual Analysis**: Uses Meta's Llama 3.2 90B Vision model for detailed X-ray image analysis
-  **Patient-Friendly Explanations**: Leverages Writer's Palmyra-Med-70B model for easy-to-understand medical explanations
-  **Medical Q&A Chat**: Interactive chat feature for general medical questions
-  **PDF Report Generation**: Download comprehensive reports with analysis and recommendations
-  **Specialist Finder**: Integrated Google search to find relevant medical specialists
-  **Multi-format Support**: Supports JPG, JPEG, and PNG image formats

## Prerequisites

- Python 3.7 or higher
- NVIDIA API key (required for AI model access)

## Installation

1. **Clone or download the repository**
   ```bash
   git clone <repository-url>
   cd Medical_Vision_X-ray
   ```

2. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Get NVIDIA API Key**
   - Visit [NVIDIA's API documentation](https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html)
   - Sign up and obtain your API key

## Usage

1. **Start the application**
   ```bash
   streamlit run Xray_Medical.py
   ```

2. **Access the web interface**
   - Open your browser and navigate to `http://localhost:8501`

3. **Use the application**
   - Enter your NVIDIA API key in the sidebar
   - Upload an X-ray image (JPG, JPEG, or PNG format)
   - Click "Analyze X-ray (Vision AI)" for initial analysis
   - Click "Get Patient-Friendly Explanation" for simplified interpretation
   - Download PDF reports as needed
   - Use the chat feature for general medical questions

## File Structure

```
vlm_demo/
     Xray_Medical.py          # Main application file
     requirements.txt         # Python dependencies
     README.md               # This file
     logo.png                # Optional logo file
     sample_xray.jpg         # Optional sample X-ray image
```

## Dependencies

The application requires the following Python packages:

- **streamlit**: Web application framework
- **Pillow**: Image processing library
- **openai**: OpenAI API client (used for NVIDIA API calls)
- **fpdf2**: PDF generation library
- **requests**: HTTP library for API calls

## Configuration

### Constants (configurable in [`Xray_Medical.py`](Xray_Medical.py))

- `MAX_ENCODED_IMAGE_SIZE_BYTES`: Maximum image size (770KB)
- `APP_NAME`: Application title
- `APP_LOGO_PATH`: Path to logo file (optional)
- `SAMPLE_IMAGE_FILENAME`: Sample X-ray image filename

### API Models Used

- **Vision Analysis**: `meta/llama-3.2-90b-vision-instruct`
- **Medical Explanations**: `writer/palmyra-med-70b`

## Key Functions

### Image Processing
- [`encode_image_to_base64`](Xray_Medical.py): Converts images to base64 for API transmission
- [`format_vlm_output_for_display`](Xray_Medical.py): Formats AI analysis output with emojis

### PDF Generation
- [`generate_xray_report_pdf`](Xray_Medical.py): Creates comprehensive PDF reports with analysis

### API Calls
- [`call_llama_3_2_90b_vision_cached`](Xray_Medical.py): Calls vision model for X-ray analysis
- [`call_palmyra_med_70b_cached`](Xray_Medical.py): Calls medical model for explanations
- [`call_palmyra_chat_cached`](Xray_Medical.py): Handles general medical chat

## Security Notes

- API keys are not stored permanently
- All processing happens locally except for API calls to NVIDIA
- No medical data is permanently stored

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your NVIDIA API key is valid and has access to the required models
2. **Image Size Issues**: Reduce image size if upload fails (max ~580KB original size)
3. **PDF Generation Errors**: Ensure proper image format (JPG, PNG)

### Error Handling

The application includes comprehensive error handling for:
- API connection issues
- Image processing errors
- PDF generation failures
- Invalid file formats

## Contributing

This is an educational demonstration project. For improvements or bug fixes:

1. Ensure all changes maintain the educational/non-clinical nature
2. Test thoroughly with various X-ray image formats
3. Maintain proper error handling and user warnings

## License

This project is for educational purposes only. Please respect NVIDIA's API terms of service and usage guidelines.

## Support

For technical issues:
1. Check the console output for detailed error messages
2. Verify API key validity
3. Ensure all dependencies are properly installed
4. Check image format and size requirements

---

**Remember: This tool is for educational demonstration only. Always consult qualified medical professionals for health-related decisions.**
