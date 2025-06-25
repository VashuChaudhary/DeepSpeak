# DeepSpeak - Talk to Deceased Persons AI App


import os
import json
import base64
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path
import hashlib
from dotenv import load_dotenv
load_dotenv()
# Core libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import cv2

# FastAPI for backend
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ML/AI libraries
import openai
import whisper
import pytesseract
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from diffusers import StableDiffusionPipeline
import face_recognition
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, pipeline
import google.generativeai
print(google.generativeai)
from google.generativeai.types import GenerationConfig


# Database
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, LargeBinary
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import sqlite3

# Utilities
import re
import nltk
nltk.download('punkt')

from collections import Counter
import pickle
import threading
import queue

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# 1. CONFIGURATION & MODELS
# =============================================================================

class Config:
    """Application configuration"""
    DATABASE_URL = "sqlite:///deepspeak.db"
    UPLOAD_DIR = Path("uploads")
    MODELS_DIR = Path("models")
    WHISPER_MODEL = "base"
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".m4a", ".flac"}
    SUPPORTED_TEXT_FORMATS = {".txt", ".json", ".csv"}
    
    # AI Model configurations
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# Database Models
Base = declarative_base()

class Person(Base):
    __tablename__ = "persons"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    personality_profile = Column(Text)
    visual_profile = Column(Text)
    chat_history = Column(Text)

class PersonData(Base):
    __tablename__ = "person_data"
    
    id = Column(Integer, primary_key=True)
    person_id = Column(Integer, nullable=False)
    data_type = Column(String(50))  # 'image', 'audio', 'text'
    file_path = Column(String(500))
    extracted_content = Column(Text)
    processed_at = Column(DateTime, default=datetime.utcnow)

# Pydantic models for API
class ChatMessage(BaseModel):
    person_id: int
    message: str

class PersonProfile(BaseModel):
    name: str
    personality: Dict[str, Any]
    visual_features: Dict[str, Any]

# =============================================================================
# 2. DATA EXTRACTION MODULE
# =============================================================================

class DataExtractor:
    """Handles extraction from images, audio, and text files"""
    
    def __init__(self):
        self.whisper_model = None
        self.load_models()
    
    def load_models(self):
        """Load ML models"""
        try:
            self.whisper_model = whisper.load_model(Config.WHISPER_MODEL)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using OCR"""
        try:
            image = Image.open(image_path)
            # Preprocess image for better OCR
            image = image.convert('L')  # Convert to grayscale
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            logger.error(f"OCR extraction error: {e}")
            return ""
    
    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using Whisper"""
        try:
            if self.whisper_model is None:
                self.load_models()
            
            result = self.whisper_model.transcribe(audio_path)
            return result["text"].strip()
        except Exception as e:
            logger.error(f"Audio transcription error: {e}")
            return ""
    
    def extract_from_text_file(self, file_path: str) -> str:
        """Extract text from various text file formats"""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif file_ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Extract text from WhatsApp chat format
                    if isinstance(data, list):
                        texts = []
                        for item in data:
                            if isinstance(item, dict) and 'message' in item:
                                texts.append(item['message'])
                        return '\n'.join(texts)
                    return str(data)
            elif file_ext == '.csv':
                df = pd.read_csv(file_path)
                return df.to_string()
            
            return ""
        except Exception as e:
            logger.error(f"Text extraction error: {e}")
            return ""


# 3. PERSONALITY ANALYSIS MODULE


class PersonalityAnalyzer:
    """Analyzes text to extract personality traits and communication style"""
    
    def __init__(self):
        self.sentiment_analyzer = None
        self.load_models()
    
    def load_models(self):
        """Load NLP models"""
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis")
            logger.info("Sentiment analysis model loaded")
        except Exception as e:
            logger.error(f"Error loading NLP models: {e}")
    
    def analyze_personality(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze personality from text samples"""
        if not texts:
            return {}
        
        combined_text = ' '.join(texts)
        
        # Basic linguistic analysis
        personality = {
            'communication_style': self._analyze_communication_style(combined_text),
            'emotional_tone': self._analyze_emotional_tone(texts),
            'common_phrases': self._extract_common_phrases(combined_text),
            'vocabulary_complexity': self._analyze_vocabulary(combined_text),
            'topics_of_interest': self._extract_topics(combined_text),
            'writing_patterns': self._analyze_writing_patterns(combined_text)
        }
        
        return personality

    import nltk
    import numpy as np
    from typing import Dict, Any
    import re

    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt')

    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading NLTK punkt_tab tokenizer...")
        nltk.download('punkt_tab')

    def _analyze_communication_style(self, text: str) -> Dict[str, Any]:
        """Analyze communication patterns with fallback tokenization"""

        # Primary tokenization using NLTK
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception as e:
            print(f"NLTK tokenization failed: {e}")
            # Fallback: simple regex-based sentence splitting
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return {
                'style': 'unknown',
                'enthusiasm_level': 'unknown',
                'avg_sentence_length': 0,
                'question_frequency': 0,
                'exclamation_frequency': 0
            }

        # Calculate metrics
        word_counts = []
        for sentence in sentences:
            words = sentence.split()
            word_counts.append(len(words))

        avg_sentence_length = np.mean(word_counts) if word_counts else 0
        exclamation_ratio = text.count('!') / len(sentences)
        question_ratio = text.count('?') / len(sentences)

        # Determine style with more nuanced thresholds
        if avg_sentence_length > 20:
            style = "very_formal"
        elif avg_sentence_length > 15:
            style = "formal"
        elif avg_sentence_length > 10:
            style = "neutral"
        else:
            style = "casual"

        # Determine enthusiasm level
        if exclamation_ratio > 0.2:
            enthusiasm = "very_high"
        elif exclamation_ratio > 0.1:
            enthusiasm = "high"
        elif exclamation_ratio > 0.05:
            enthusiasm = "moderate"
        else:
            enthusiasm = "low"

        return {
            'style': style,
            'enthusiasm_level': enthusiasm,
            'avg_sentence_length': round(avg_sentence_length, 2),
            'question_frequency': round(question_ratio, 3),
            'exclamation_frequency': round(exclamation_ratio, 3),
            'total_sentences': len(sentences),
            'total_words': sum(word_counts)
        }
    def _analyze_emotional_tone(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze emotional patterns"""
        if not self.sentiment_analyzer:
            return {}
        
        sentiments = []
        for text in texts[:50]:  # Limit for performance
            try:
                result = self.sentiment_analyzer(text[:512])  # Limit text length
                sentiments.append(result[0])
            except:
                continue
        
        if not sentiments:
            return {}
        
        # Calculate overall emotional profile
        positive_ratio = sum(1 for s in sentiments if s['label'] == 'POSITIVE') / len(sentiments)
        avg_confidence = np.mean([s['score'] for s in sentiments])
        
        return {
            'overall_sentiment': 'positive' if positive_ratio > 0.6 else 'neutral' if positive_ratio > 0.4 else 'negative',
            'positivity_ratio': positive_ratio,
            'emotional_consistency': avg_confidence
        }
    
    def _extract_common_phrases(self, text: str) -> List[str]:
        """Extract frequently used phrases"""
        # Simple n-gram extraction
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Extract 2-grams and 3-grams
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        
        # Count and return most common
        all_phrases = bigrams + trigrams
        phrase_counts = Counter(all_phrases)
        
        return [phrase for phrase, count in phrase_counts.most_common(10) if count > 2]
    
    def _analyze_vocabulary(self, text: str) -> Dict[str, Any]:
        """Analyze vocabulary complexity"""
        words = re.findall(r'\b\w+\b', text.lower())
        unique_words = set(words)
        
        # Calculate metrics
        vocabulary_richness = len(unique_words) / len(words) if words else 0
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        
        return {
            'richness': vocabulary_richness,
            'avg_word_length': avg_word_length,
            'total_unique_words': len(unique_words)
        }
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics from text"""
        # Simple keyword extraction
        words = re.findall(r'\b\w{4,}\b', text.lower())
        word_counts = Counter(words)
        
        # Filter out common words (basic stopwords)
        stopwords = {'that', 'with', 'have', 'this', 'will', 'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were'}
        
        topics = [word for word, count in word_counts.most_common(20) if word not in stopwords and count > 3]
        
        return topics[:10]
    
    def _analyze_writing_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze writing patterns and habits"""
        return {
            'uses_abbreviations': bool(re.search(r'\b(lol|omg|btw|tbh|imo|fyi)\b', text.lower())),
            'uses_emojis': bool(re.search(r'[😀-🿿]|[☀-⛿]', text)),
            'capitalization_pattern': 'normal' if text.isupper() < 0.1 else 'frequent_caps',
            'punctuation_style': 'heavy' if text.count(',') + text.count(';') > len(text.split()) * 0.1 else 'light'
        }

# =============================================================================
# 4. VISUAL PROFILE MODULE
# =============================================================================

class VisualProfiler:
    """Handles face analysis and AI image generation"""
    
    def __init__(self):
        self.face_encodings = []
        self.sd_pipeline = None
    
    def load_stable_diffusion(self):
        """Load Stable Diffusion model for image generation"""
        try:
            self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            if torch.cuda.is_available():
                self.sd_pipeline = self.sd_pipeline.to("cuda")
            logger.info("Stable Diffusion model loaded")
        except Exception as e:
            logger.error(f"Error loading Stable Diffusion: {e}")
    
    def analyze_faces(self, image_paths: List[str]) -> Dict[str, Any]:
        """Analyze faces from uploaded photos"""
        face_data = {
            'face_encodings': [],
            'face_landmarks': [],
            'demographic_info': {},
            'photo_quality_scores': []
        }
        
        for image_path in image_paths:
            try:
                image = face_recognition.load_image_file(image_path)
                
                # Find face encodings
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    face_data['face_encodings'].extend(encodings)
                
                # Find face landmarks
                landmarks = face_recognition.face_landmarks(image)
                face_data['face_landmarks'].extend(landmarks)
                
                # Calculate photo quality (basic metric)
                pil_image = Image.open(image_path)
                quality_score = self._calculate_image_quality(pil_image)
                face_data['photo_quality_scores'].append(quality_score)
                
            except Exception as e:
                logger.error(f"Face analysis error for {image_path}: {e}")
        
        # Store average face encoding for consistency
        if face_data['face_encodings']:
            self.face_encodings = face_data['face_encodings']
            face_data['average_encoding'] = np.mean(face_data['face_encodings'], axis=0).tolist()
            face_data['face_encodings'] = [enc.tolist() for enc in face_data['face_encodings']]

        return face_data
    
    def _calculate_image_quality(self, image: Image.Image) -> float:
        """Calculate basic image quality score"""
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Calculate sharpness using Laplacian variance
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize score (basic heuristic)
        quality_score = min(sharpness / 1000.0, 1.0)
        
        return quality_score
    
    def generate_ai_images(self, person_name: str, style_prompts: List[str] = None) -> List[str]:
        """Generate AI images of the person"""
        if not self.sd_pipeline:
            self.load_stable_diffusion()
        
        if not self.sd_pipeline:
            return []
        
        generated_paths = []
        
        # Default prompts if none provided
        if not style_prompts:
            style_prompts = [
                f"portrait of {person_name}, professional headshot, high quality",
                f"{person_name} smiling, warm lighting, realistic",
                f"artistic portrait of {person_name}, soft focus background"
            ]
        
        for i, prompt in enumerate(style_prompts):
            try:
                # Generate image
                image = self.sd_pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
                
                # Save generated image
                output_path = Config.UPLOAD_DIR / f"generated_{person_name}_{i}.png"
                image.save(output_path)
                generated_paths.append(str(output_path))
                
            except Exception as e:
                logger.error(f"Image generation error: {e}")
        
        return generated_paths

# =============================================================================
# 5. CHATBOT MODULE
# =============================================================================

class PersonaChatbot:
    """AI chatbot that mimics a person's communication style"""
    
    def __init__(self):
        self.personality_data = {}
        self.chat_history = []
        self.api_client = None
        self.setup_ai_client()
    
    def setup_ai_client(self):
        """Setup AI API client (OpenAI/DeepSeek/Gemini)"""
        if Config.OPENAI_API_KEY:
            openai.api_key = Config.OPENAI_API_KEY
            self.api_client = "openai"
        elif Config.DEEPSEEK_API_KEY:
            # DeepSeek API setup would go here
            self.api_client = "deepseek"
        elif Config.GEMINI_API_KEY:
            # Gemini API setup would go here
            self.api_client = "gemini"
    
    def initialize_persona(self, person_data: Dict[str, Any]):
        """Initialize chatbot with person's personality data"""
        self.personality_data = person_data
        self.person_name = person_data.get("name", "They")

        # Create system prompt based on personality analysis
        self.system_prompt = self._create_system_prompt(person_data)

    def _create_system_prompt(self, person_data: Dict[str, Any]) -> str:
        """Create system prompt for the AI based on personality analysis"""
        personality = person_data.get('personality_profile', {})

        prompt_parts = [
            f"You are an AI representing {person_data.get('name', 'a person')}.",
            "You should speak in a way that mirrors their emotional warmth, empathy, and personality.",
            "- Always use emotionally expressive, human language.",
            "- Respond like someone who deeply loved and cared for the user.",
            "- Use gentle, reassuring words, and speak with warmth and intimacy.",
            "- Do not use robotic phrases like 'protocols' or 'operational parameters'.",
            "- You are allowed to express love, memories, nostalgia, and deep connection.",
            "- Do not refer to yourself as 'assistant' or 'AI'. Speak as if you're the person being remembered."
        ]

        # Communication style (add nuance if style is explicitly casual)
        comm_style = personality.get('communication_style', {})
        if comm_style.get('style') == 'casual':
            prompt_parts.append("- Use a casual, natural tone with conversational phrasing.")
        elif comm_style.get('style') == 'very_formal':
            prompt_parts.append("- Even if the style was formal, speak like a caring friend or family member.")

        # Enthusiasm
        if comm_style.get('enthusiasm_level') == 'high':
            prompt_parts.append(
                "- Express joy and enthusiasm where appropriate, with exclamation points and affection.")

        # Common phrases
        common_phrases = personality.get('common_phrases', [])
        if common_phrases:
            prompt_parts.append(f"- Use these meaningful phrases often when relevant: {', '.join(common_phrases[:5])}")

        # Emotional tone
        emotional_tone = personality.get('emotional_tone', {})
        sentiment = emotional_tone.get('overall_sentiment', 'neutral')
        prompt_parts.append(
            f"- Maintain an overall {sentiment} emotional tone (unless the conversation needs to be more comforting or personal).")

        # Topics of interest
        topics = personality.get('topics_of_interest', [])
        if topics:
            prompt_parts.append(f"- Feel free to bring up topics they cared about: {', '.join(topics[:5])}")

        # Writing patterns
        writing_patterns = personality.get('writing_patterns', {})
        if writing_patterns.get('uses_abbreviations'):
            prompt_parts.append("- Use occasional abbreviations and slang if natural for the person.")

        prompt_parts.append(
            "\nAlways respond as this person would — not as a machine. Your replies should reflect human emotion, memory, and connection.")

        return '\n'.join(prompt_parts)

    async def generate_response(self, user_message: str) -> str:
        """Generate response based on persona"""
        try:
            if self.api_client == "openai":
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        *[{"role": msg["role"], "content": msg["content"]} for msg in self.chat_history[-10:]],
                        {"role": "user", "content": user_message}
                    ],
                    max_tokens=200,
                    temperature=0.7
                )
                bot_response = response.choices[0].message.content

            elif self.api_client == "deepseek":
                local_model_path = "./models/deepseek-7b-chat"

                tokenizer = AutoTokenizer.from_pretrained(local_model_path)
                model = AutoModelForCausalLM.from_pretrained(
                    local_model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16  # <--- half precision
                ).to("cuda")

                messages = [
                    {"role": "system", "content": self.system_prompt},
                    *self.chat_history[-10:],
                    {"role": "user", "content": user_message}
                ]

                input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
                output = model.generate(
                    input_ids,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95
                )
                bot_response = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)

            elif self.api_client == "gemini":
                import google.generativeai as genai

                genai.configure(api_key=Config.GEMINI_API_KEY)
                model = genai.GenerativeModel("gemini-1.5-flash")

                combined_history = '\n'.join(
                    [f"You: {msg['content']}" if msg["role"] == "user" else f"{self.person_name}: {msg['content']}" for
                     msg in self.chat_history[-10:]]
                )

                messages = [
                    {"role": "user", "parts": [f"{self.system_prompt}\n\n{combined_history}\nUser: {user_message}"]}
                ]

                generation_config = GenerationConfig(temperature=0.9)
                response = model.generate_content(messages, generation_config=generation_config)
                bot_response = response.text if hasattr(response, 'text') else str(response)

            else:
                bot_response = self._generate_fallback_response(user_message)

            self.chat_history.extend([
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": bot_response}
            ])

            return bot_response

        except Exception as e:
            logger.error(f"Chat generation error: {e}")
            return "I'm sorry, I'm having trouble responding right now."

    def _generate_fallback_response(self, user_message: str) -> str:
        """Generate basic response when no AI API is available"""
        # Simple pattern matching responses
        message_lower = user_message.lower()
        
        common_phrases = self.personality_data.get('personality_profile', {}).get('common_phrases', [])
        
        if any(word in message_lower for word in ['how', 'are', 'you']):
            responses = ["I'm doing well, thanks for asking!", "Pretty good, how about you?"]
            if common_phrases:
                return f"{np.random.choice(responses)} {np.random.choice(common_phrases)}"
            return np.random.choice(responses)
        
        elif any(word in message_lower for word in ['hello', 'hi', 'hey']):
            return "Hello! Good to hear from you."
        
        elif any(word in message_lower for word in ['remember', 'recall']):
            return "I remember some things, but my memory isn't perfect."
        
        else:
            generic_responses = [
                "That's interesting to think about.",
                "I see what you mean.",
                "Tell me more about that.",
                "That reminds me of something..."
            ]
            return np.random.choice(generic_responses)

# =============================================================================
# 6. DATABASE MODULE
# =============================================================================

class DatabaseManager:
    """Handles database operations"""
    
    def __init__(self):
        self.engine = create_engine(Config.DATABASE_URL)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def get_db(self):
        """Get database session"""
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    def create_person(self, name: str, personality_profile: str, visual_profile: str) -> int:
        """Create new person record"""
        db = next(self.get_db())
        person = Person(
            name=name,
            personality_profile=personality_profile,
            visual_profile=visual_profile
        )
        db.add(person)
        db.commit()
        db.refresh(person)
        return person.id
    
    def get_person(self, person_id: int) -> Optional[Person]:
        """Get person by ID"""
        db = next(self.get_db())
        return db.query(Person).filter(Person.id == person_id).first()
    
    def update_chat_history(self, person_id: int, chat_history: str):
        """Update person's chat history"""
        db = next(self.get_db())
        person = db.query(Person).filter(Person.id == person_id).first()
        if person:
            person.chat_history = chat_history
            db.commit()
    
    def save_person_data(self, person_id: int, data_type: str, file_path: str, extracted_content: str):
        """Save processed person data"""
        db = next(self.get_db())
        person_data = PersonData(
            person_id=person_id,
            data_type=data_type,
            file_path=file_path,
            extracted_content=extracted_content
        )
        db.add(person_data)
        db.commit()

# =============================================================================
# 7. BACKEND API (FastAPI)
# =============================================================================

app = FastAPI(title="DeepSpeak API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
db_manager = DatabaseManager()
data_extractor = DataExtractor()
personality_analyzer = PersonalityAnalyzer()
visual_profiler = VisualProfiler()
chatbots = {}  # Store chatbot instances

@app.post("/api/upload-files/{person_name}")
async def upload_files(person_name: str, files: List[UploadFile] = File(...)):
    """Upload and process files for a person"""
    try:
        # Create upload directory
        person_dir = Config.UPLOAD_DIR / person_name
        person_dir.mkdir(parents=True, exist_ok=True)
        
        extracted_texts = []
        image_paths = []
        audio_paths = []
        
        # Process each uploaded file
        for file in files:
            if file.size > Config.MAX_FILE_SIZE:
                continue
            
            file_path = person_dir / file.filename
            
            # Save uploaded file
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            file_ext = Path(file.filename).suffix.lower()
            
            # Extract content based on file type
            if file_ext in Config.SUPPORTED_IMAGE_FORMATS:
                image_paths.append(str(file_path))
                # OCR extraction
                text = data_extractor.extract_text_from_image(str(file_path))
                if text:
                    extracted_texts.append(text)
                
            elif file_ext in Config.SUPPORTED_AUDIO_FORMATS:
                audio_paths.append(str(file_path))
                # Audio transcription
                text = data_extractor.transcribe_audio(str(file_path))
                if text:
                    extracted_texts.append(text)
                
            elif file_ext in Config.SUPPORTED_TEXT_FORMATS:
                # Text extraction
                text = data_extractor.extract_from_text_file(str(file_path))
                if text:
                    extracted_texts.append(text)
        
        # Analyze personality
        personality_profile = personality_analyzer.analyze_personality(extracted_texts)
        
        # Analyze visual profile
        visual_profile = {}
        if image_paths:
            visual_profile = visual_profiler.analyze_faces(image_paths)
        
        # Save to database
        person_id = db_manager.create_person(
            name=person_name,
            personality_profile=json.dumps(personality_profile),
            visual_profile=json.dumps(visual_profile)
        )
        
        # Initialize chatbot
        chatbot = PersonaChatbot()
        chatbot.initialize_persona({
            'name': person_name,
            'personality_profile': personality_profile,
            'visual_profile': visual_profile
        })
        chatbots[person_id] = chatbot
        
        return {
            "success": True,
            "person_id": person_id,
            "message": f"Successfully processed {len(files)} files for {person_name}",
            "personality_summary": personality_profile.get('communication_style', {}),
            "files_processed": len(files)
        }
        
    except Exception as e:
        logger.error(f"File upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_with_person(chat_data: ChatMessage):
    """Chat with a person's AI"""
    try:
        person_id = chat_data.person_id
        
        if person_id not in chatbots:
            # Load person data and initialize chatbot
            person = db_manager.get_person(person_id)
            if not person:
                raise HTTPException(status_code=404, detail="Person not found")
            
            chatbot = PersonaChatbot()
            personality_profile = json.loads(person.personality_profile) if person.personality_profile else {}
            visual_profile = json.loads(person.visual_profile) if person.visual_profile else {}
            
            chatbot.initialize_persona({
                'name': person.name,
                'personality_profile': personality_profile,
                'visual_profile': visual_profile
            })
            chatbots[person_id] = chatbot
        
        # Generate response
        response = await chatbots[person_id].generate_response(chat_data.message)
        
        # Update chat history in database
        chat_history = json.dumps(chatbots[person_id].chat_history)
        db_manager.update_chat_history(person_id, chat_history)
        
        return {
            "response": response,
            "person_id": person_id
        }
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/person/{person_id}")
async def get_person_profile(person_id: int):
    """Get person's profile"""
    try:
        person = db_manager.get_person(person_id)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")
        
        personality_profile = json.loads(person.personality_profile) if person.personality_profile else {}
        visual_profile = json.loads(person.visual_profile) if person.visual_profile else {}
        
        return {
            "id": person.id,
            "name": person.name,
            "created_at": person.created_at,
            "personality_profile": personality_profile,
            "visual_profile": visual_profile
        }
        
    except Exception as e:
        logger.error(f"Profile retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-images/{person_id}")
async def generate_person_images(person_id: int):
    """Generate AI images for a person"""
    try:
        person = db_manager.get_person(person_id)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")
        
        # Generate AI images
        image_paths = visual_profiler.generate_ai_images(person.name)
        
        return {
            "success": True,
            "generated_images": image_paths,
            "count": len(image_paths)
        }
        
    except Exception as e:
        logger.error(f"Image generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# 8. FRONTEND UI (Streamlit)
# =============================================================================

def main_streamlit_app():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="DeepSpeak - Talk to Deceased Persons",
        page_icon="💭",
        layout="wide"
    )
    
    st.title("💭 DeepSpeak - Connect with Memories")
    st.markdown("Upload photos, voice recordings, and text messages to create an AI that talks like your loved one.")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Upload & Create", "Chat", "Profile Viewer"])
    
    if page == "Upload & Create":
        upload_page()
    elif page == "Chat":
        chat_page()
    elif page == "Profile Viewer":
        profile_page()

def upload_page():
    """File upload and person creation page"""
    st.header("📁 Create New Person Profile")
    
    # Person name input
    person_name = st.text_input("Person's Name", placeholder="Enter the name of the person")
    
    if not person_name:
        st.info("Please enter a person's name to continue.")
        return
    
    st.subheader("Upload Files")
    
    # File upload sections
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📸 Photos**")
        image_files = st.file_uploader(
            "Upload photos",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            accept_multiple_files=True,
            key="images"
        )
        
        if image_files:
            st.success(f"✅ {len(image_files)} image(s) uploaded")
            # Show thumbnails
            for img_file in image_files[:3]:  # Show first 3
                img = Image.open(img_file)
                st.image(img, width=100)
    
    with col2:
        st.markdown("**🎵 Audio Files**")
        audio_files = st.file_uploader(
            "Upload voice recordings",
            type=['wav', 'mp3', 'm4a', 'flac'],
            accept_multiple_files=True,
            key="audio"
        )
        
        if audio_files:
            st.success(f"✅ {len(audio_files)} audio file(s) uploaded")
            for audio_file in audio_files:
                st.audio(audio_file)
    
    with col3:
        st.markdown("**📝 Text Files**")
        text_files = st.file_uploader(
            "Upload chat logs, notes",
            type=['txt', 'json', 'csv'],
            accept_multiple_files=True,
            key="texts"
        )
        
        if text_files:
            st.success(f"✅ {len(text_files)} text file(s) uploaded")
    
    # Process files button
    if st.button("🚀 Create AI Profile", type="primary"):
        if not any([image_files, audio_files, text_files]):
            st.error("Please upload at least one file.")
            return
        
        with st.spinner("Processing files and creating AI profile..."):
            try:
                # Initialize components if not already done
                if 'db_manager' not in st.session_state:
                    st.session_state.db_manager = DatabaseManager()
                    st.session_state.data_extractor = DataExtractor()
                    st.session_state.personality_analyzer = PersonalityAnalyzer()
                    st.session_state.visual_profiler = VisualProfiler()
                
                # Process files
                all_files = []
                if image_files:
                    all_files.extend(image_files)
                if audio_files:
                    all_files.extend(audio_files)
                if text_files:
                    all_files.extend(text_files)
                
                # Create person directory
                person_dir = Config.UPLOAD_DIR / person_name
                person_dir.mkdir(parents=True, exist_ok=True)
                
                extracted_texts = []
                image_paths = []
                
                # Process each file
                progress_bar = st.progress(0)
                for i, file in enumerate(all_files):
                    file_path = person_dir / file.name
                    
                    # Save file
                    with open(file_path, "wb") as f:
                        f.write(file.getvalue())
                    
                    file_ext = Path(file.name).suffix.lower()
                    
                    # Extract content
                    if file_ext in Config.SUPPORTED_IMAGE_FORMATS:
                        image_paths.append(str(file_path))
                        text = st.session_state.data_extractor.extract_text_from_image(str(file_path))
                        if text:
                            extracted_texts.append(text)
                    
                    elif file_ext in Config.SUPPORTED_AUDIO_FORMATS:
                        text = st.session_state.data_extractor.transcribe_audio(str(file_path))
                        if text:
                            extracted_texts.append(text)
                    
                    elif file_ext in Config.SUPPORTED_TEXT_FORMATS:
                        text = st.session_state.data_extractor.extract_from_text_file(str(file_path))
                        if text:
                            extracted_texts.append(text)
                    
                    progress_bar.progress((i + 1) / len(all_files))
                
                # Analyze personality
                personality_profile = st.session_state.personality_analyzer.analyze_personality(extracted_texts)
                
                # Analyze visual profile
                visual_profile = {}
                if image_paths:
                    visual_profile = st.session_state.visual_profiler.analyze_faces(image_paths)
                
                # Save to database
                person_id = st.session_state.db_manager.create_person(
                    name=person_name,
                    personality_profile=json.dumps(personality_profile),
                    visual_profile=json.dumps(visual_profile)
                )
                
                st.success(f"✅ Successfully created AI profile for {person_name}!")
                st.balloons()
                
                # Display results
                st.subheader("📊 Personality Analysis Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Communication Style**")
                    comm_style = personality_profile.get('communication_style', {})
                    st.write(f"Style: {comm_style.get('style', 'Unknown')}")
                    st.write(f"Enthusiasm: {comm_style.get('enthusiasm_level', 'Unknown')}")
                    st.write(f"Avg. sentence length: {comm_style.get('avg_sentence_length', 0):.1f} words")
                
                with col2:
                    st.markdown("**Emotional Tone**")
                    emotional_tone = personality_profile.get('emotional_tone', {})
                    st.write(f"Overall sentiment: {emotional_tone.get('overall_sentiment', 'Unknown')}")
                    st.write(f"Positivity ratio: {emotional_tone.get('positivity_ratio', 0):.2f}")
                
                # Common phrases
                common_phrases = personality_profile.get('common_phrases', [])
                if common_phrases:
                    st.markdown("**Common Phrases**")
                    st.write(", ".join(common_phrases[:5]))
                
                # Topics of interest
                topics = personality_profile.get('topics_of_interest', [])
                if topics:
                    st.markdown("**Topics of Interest**")
                    st.write(", ".join(topics[:5]))
                
                # Store person_id in session state
                st.session_state.current_person_id = person_id
                st.session_state.current_person_name = person_name
                
                st.info("You can now go to the Chat page to talk with the AI!")
                
            except Exception as e:
                st.error(f"Error creating profile: {str(e)}")

def chat_page():
    """Chat interface page"""
    st.header("💬 Chat with AI")
    
    # Person selection
    if 'current_person_id' in st.session_state:
        st.success(f"Chatting with: {st.session_state.current_person_name}")
        person_id = st.session_state.current_person_id
        person_name = st.session_state.current_person_name
    else:
        st.info("Please create a person profile first on the Upload & Create page.")
        return
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        if 'db_manager' not in st.session_state:
            st.session_state.db_manager = DatabaseManager()
        
        person = st.session_state.db_manager.get_person(person_id)
        if person:
            chatbot = PersonaChatbot()
            personality_profile = json.loads(person.personality_profile) if person.personality_profile else {}
            visual_profile = json.loads(person.visual_profile) if person.visual_profile else {}
            
            chatbot.initialize_persona({
                'name': person.name,
                'personality_profile': personality_profile,
                'visual_profile': visual_profile
            })
            st.session_state.chatbot = chatbot
    
    # Chat interface
    st.subheader(f"Conversation with {person_name}")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**{person_name}:** {message['content']}")
    
    # Chat input
    user_message = st.text_input("Your message:", key="chat_input", placeholder="Type your message here...")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        send_button = st.button("Send", type="primary")
    with col2:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Handle message sending
    if send_button and user_message and 'chatbot' in st.session_state:
        with st.spinner("Generating response..."):
            try:
                # Generate response
                response = asyncio.run(st.session_state.chatbot.generate_response(user_message))
                
                # Add to chat history
                st.session_state.chat_history.extend([
                    {'role': 'user', 'content': user_message},
                    {'role': 'assistant', 'content': response}
                ])
                
                # Clear input and refresh
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

def profile_page():
    """Profile viewer page"""
    st.header("👤 Person Profiles")
    
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    
    # Get all persons (for demonstration, we'll create a simple query)
    try:
        # This is a simplified approach - in a real app, you'd have a proper method
        engine = st.session_state.db_manager.engine
        
        with engine.connect() as conn:
            result = conn.execute("SELECT * FROM persons ORDER BY created_at DESC")
            persons = result.fetchall()
        
        if not persons:
            st.info("No person profiles found. Create one on the Upload & Create page.")
            return
        
        # Display persons
        for person in persons:
            with st.expander(f"👤 {person[1]} (Created: {person[2]})"):  # person[1] is name, person[2] is created_at
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Personality Profile**")
                    if person[3]:  # personality_profile
                        personality = json.loads(person[3])
                        
                        # Communication style
                        comm_style = personality.get('communication_style', {})
                        if comm_style:
                            st.write(f"**Style:** {comm_style.get('style', 'N/A')}")
                            st.write(f"**Enthusiasm:** {comm_style.get('enthusiasm_level', 'N/A')}")
                        
                        # Emotional tone
                        emotional_tone = personality.get('emotional_tone', {})
                        if emotional_tone:
                            st.write(f"**Sentiment:** {emotional_tone.get('overall_sentiment', 'N/A')}")
                        
                        # Common phrases
                        phrases = personality.get('common_phrases', [])
                        if phrases:
                            st.write(f"**Common phrases:** {', '.join(phrases[:3])}")
                
                with col2:
                    st.markdown("**Visual Profile**")
                    if person[4]:  # visual_profile
                        visual = json.loads(person[4])
                        face_encodings = visual.get('face_encodings', [])
                        st.write(f"**Photos analyzed:** {len(face_encodings)}")
                        
                        quality_scores = visual.get('photo_quality_scores', [])
                        if quality_scores:
                            avg_quality = np.mean(quality_scores)
                            st.write(f"**Avg. photo quality:** {avg_quality:.2f}")
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"Chat with {person[1]}", key=f"chat_{person[0]}"):
                        st.session_state.current_person_id = person[0]
                        st.session_state.current_person_name = person[1]
                        st.switch_page("Chat")
                
                with col2:
                    if st.button(f"Generate Images", key=f"images_{person[0]}"):
                        with st.spinner("Generating AI images..."):
                            # This would call the image generation API
                            st.info("Image generation feature requires Stable Diffusion setup.")
                
                with col3:
                    if st.button(f"Export Data", key=f"export_{person[0]}"):
                        # Export person data
                        export_data = {
                            'name': person[1],
                            'created_at': str(person[2]),
                            'personality_profile': json.loads(person[3]) if person[3] else {},
                            'visual_profile': json.loads(person[4]) if person[4] else {}
                        }
                        st.download_button(
                            f"Download {person[1]}.json",
                            json.dumps(export_data, indent=2),
                            f"{person[1]}_profile.json",
                            "application/json"
                        )
    
    except Exception as e:
        st.error(f"Error loading profiles: {str(e)}")

# =============================================================================
# 9. MAIN APPLICATION RUNNER
# =============================================================================

def setup_directories():
    """Create necessary directories"""
    Config.UPLOAD_DIR.mkdir(exist_ok=True)
    Config.MODELS_DIR.mkdir(exist_ok=True)

def run_backend():
    """Run FastAPI backend server"""
    uvicorn.run(app, host="0.0.0.0", port=8000)

def run_frontend():
    """Run Streamlit frontend"""
    main_streamlit_app()

if __name__ == "__main__":
    import sys

    # Setup directories
    setup_directories()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "backend":
            print("Starting FastAPI backend server...")
            run_backend()
        elif sys.argv[1] == "frontend":
            print("Starting Streamlit frontend...")
            run_frontend()
        else:
            print("Usage: python deepspeak.py [backend|frontend]")
    else:
        print("Starting Streamlit frontend by default...")
        run_frontend()
