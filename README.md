# Plant_Sense

A machine learning application for plant disease detection and classification.

## Overview
Plant_Sense is an AI-powered tool that helps identify plant diseases using image recognition technology. The application uses a trained Keras model to analyze plant images and provide disease diagnosis.

## Features
- Plant disease detection from images
- Machine learning model powered by Keras
- Simple Python-based interface
- Real-time analysis capabilities
- Groq API integration for enhanced inference

## Installation
1. Clone the repository
2. Set up Groq API credentials:
   - Sign up at [Groq Console](https://console.groq.com)
   - Get your API key
   - Set the environment variable:
   ```bash
   set GROQ_API_KEY=your_api_key_here
   ```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the application using:
```bash
python app.py
```

## Project Structure
```
.
├── .env                # Environment variables (including GROQ_API_KEY)
├── .gitignore         # Git ignore file
├── app.py             # Main application file
├── model.keras        # Trained ML model
├── requirements.txt   # Python dependencies
└── imgs/             # Image assets
    ├── logo_plant.png
    └── plant_bot.png
```

## Requirements
- Python 3.8+
- Groq API key
- See `requirements.txt` for a full list of dependencies

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.