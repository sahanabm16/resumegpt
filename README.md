# AI-Powered ATS Resume Checker 🚀

A comprehensive resume optimization tool powered by Google's Gemini AI that helps job seekers create ATS-friendly resumes with intelligent suggestions and automatic improvements.

## Features

### 🎯 Core Functionality
- **AI-Powered Analysis**: Uses Google Gemini AI for intelligent resume evaluation
- **ATS Compatibility Check**: Ensures your resume passes Applicant Tracking Systems
- **Section-wise Analysis**: Detailed evaluation of each resume section
- **Auto-Fix Feature**: One-click improvements for resume sections
- **Real-time Scoring**: Instant feedback with color-coded scores

### 📊 Analysis Categories
- **Contact Information**: Format and completeness check
- **Professional Summary**: Impact and keyword optimization
- **Work Experience**: Achievement quantification and relevance
- **Education**: Proper formatting and relevance
- **Skills**: Industry-relevant keywords and organization

### 🎨 User Interface
- **Clean, Modern Design**: Professional gradient-based UI
- **Responsive Layout**: Works on desktop and mobile
- **Interactive Navigation**: Easy-to-use sidebar menu
- **Visual Feedback**: Color-coded scores and progress indicators
- **Section Highlighting**: Clear identification of improvement areas

### 🔧 Technical Features
- **Multi-format Support**: PDF and DOCX file upload
- **Text Extraction**: Advanced parsing for various resume formats
- **Real-time Processing**: Fast AI analysis with progress indicators
- **Download Functionality**: Export improved resume as text file
- **Error Handling**: Robust error management and user feedback

## Installation

1. **Clone or download** this repository
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key**:
   - Create a `.env` file in the project directory
   - Add your Google Gemini API key:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```
   Or simply double-click `run_app.bat` on Windows

## Usage

### 1. Upload Resume
- Navigate to the "Upload Resume" section
- Drag and drop or browse for your resume file (PDF/DOCX)
- Click "Analyze Resume with AI" to start the evaluation

### 2. Review Analysis
- Check your overall ATS compatibility score
- Review section-wise analysis with detailed feedback
- Identify missing keywords and formatting issues
- See prioritized improvement recommendations

### 3. Apply Improvements
- Click "Auto-Fix" buttons for sections you want to improve
- Review original vs. improved content side-by-side
- Accept or reject AI-generated improvements
- See real-time updates to your resume quality

### 4. Download Results
- Export your improved resume
- View improvement summary statistics
- Save the optimized version for job applications

## Key Benefits

### For Job Seekers
- **Increase Interview Chances**: ATS-optimized resumes get past initial screening
- **Professional Quality**: AI ensures industry-standard formatting and content
- **Time-Saving**: Automatic improvements reduce manual editing time
- **Keyword Optimization**: AI identifies and suggests relevant industry keywords
- **Objective Feedback**: Unbiased analysis of resume effectiveness

### For Career Counselors
- **Scalable Support**: Help multiple clients efficiently
- **Consistent Quality**: Standardized improvement suggestions
- **Educational Tool**: Show clients what makes a resume effective
- **Progress Tracking**: Before/after comparisons for client progress

## Technical Architecture

### AI Engine
- **Google Gemini Pro**: Latest language model for intelligent analysis
- **Custom Prompts**: Specialized prompts for resume evaluation
- **JSON Parsing**: Structured analysis output for consistent processing
- **Error Recovery**: Fallback mechanisms for robust operation

### Frontend
- **Streamlit Framework**: Python-based web application
- **Custom CSS**: Professional styling with gradient themes
- **Component Library**: Reusable UI components for consistency
- **Responsive Design**: Mobile-friendly interface

### File Processing
- **PyPDF2**: PDF text extraction with error handling
- **python-docx**: DOCX document processing
- **Text Cleaning**: Preprocessing for optimal AI analysis
- **Format Preservation**: Maintain original document structure

## Configuration

### Environment Variables
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

### Customization Options
- Modify AI prompts in `analyze_resume_with_gemini()` function
- Adjust scoring criteria and thresholds
- Customize CSS themes in `load_css()` function
- Add new analysis sections as needed

## Troubleshooting

### Common Issues

1. **API Key Error**
   - Ensure your `.env` file is in the correct directory
   - Verify your Gemini API key is valid and active
   - Check API quota and usage limits

2. **File Upload Issues**
   - Ensure file size is under Streamlit's limit (200MB)
   - Check file format (only PDF and DOCX supported)
   - Verify file is not password-protected

3. **Analysis Errors**
   - Check internet connection for API calls
   - Ensure resume text is extractable (not image-based)
   - Try uploading a different resume format

### Performance Tips
- Use well-formatted resumes for best results
- Ensure stable internet connection for AI analysis
- Close other browser tabs to free up memory
- Use recent versions of supported browsers

## Security & Privacy

- **No Data Storage**: Resume content is not saved on servers
- **Secure Processing**: API calls use encrypted connections
- **Local Operation**: All processing happens in your browser session
- **API Compliance**: Follows Google's AI usage policies

## Contributing

Feel free to contribute improvements:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open-source and available under the MIT License.

## Support

For issues or questions:
- Check the troubleshooting section above
- Review error messages in the application
- Ensure all dependencies are correctly installed

---

**Built with ❤️ using Python, Streamlit, and Google Gemini AI**
