import os
import time
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from rag_engine import RAGEngine
from db import (
    init_db,
    create_user,
    verify_user,
    save_chat,
    get_chat_history,
    save_rating,
)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-secret-key")

# Admin credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# Helper function to get text in current language
def get_text_for_lang(key, lang='en'):
    """Get translated text for a specific language"""
    TRANSLATIONS = {
        'en': {
            'disclaimer': '⚕️ This is an AI assistant for informational purposes only. Always consult healthcare professionals for medical advice.',
            'unauthorized': 'Unauthorized',
            'please_type': 'Please type a question.',
        },
        'hi': {
            'disclaimer': '⚕️ यह केवल सूचनात्मक उद्देश्यों के लिए एक AI असिस्टेंट है। चिकित्सा सलाह के लिए हमेशा स्वास्थ्य देखभाल पेशेवरों से परामर्श लें।',
            'unauthorized': 'अनधिकृत',
            'please_type': 'कृपया एक प्रश्न लिखें।',
        }
    }
    return TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, key)

# Supported languages
LANGUAGES = {
    'en': 'English',
    'hi': 'हिन्दी (Hindi)',
    'bn': 'বাংলা (Bengali)',
    'te': 'తెలుగు (Telugu)',
    'ta': 'தமிழ் (Tamil)',
    'mr': 'मराठी (Marathi)',
    'gu': 'ગુજરાતી (Gujarati)',
    'kn': 'ಕನ್ನಡ (Kannada)',
    'ml': 'മലയാളം (Malayalam)',
    'pa': 'ਪੰਜਾਬੀ (Punjabi)',
    'es': 'Español (Spanish)',
    'fr': 'Français (French)',
    'de': 'Deutsch (German)',
    'zh': '中文 (Chinese)',
    'ja': '日本語 (Japanese)',
    'ru': 'Русский (Russian)',
    'ar': 'العربية (Arabic)'
}

# Translation dictionaries for static content
TRANSLATIONS = {
    'en': {
        # Navigation
        'home': 'Home',
        'chat': 'Chat',
        'login': 'Login',
        'register': 'Register',
        'logout': 'Logout',
        'profile': 'Profile',
        
        # Chat interface
        'medical_assistant': 'MediRAG Assistant',
        'type_message': 'Type your message...',
        'send': 'Send',
        'voice_input': 'Voice Input',
        'text_to_speech': 'Text-to-Speech',
        'stop_speaking': 'Stop Speaking',
        'listening': 'Listening...',
        'offline': 'Offline',
        'not_supported': 'Not supported',
        'error': 'Error',
        'thinking': 'Thinking',
        'you': 'You',
        'assistant': 'MediRAG Assistant',
        'listen': 'Listen',
        'helpful': 'Helpful',
        'not_helpful': 'Not Helpful',
        'sources': 'Sources',
        'latency': 'Latency',
        
        # Welcome message
        'welcome_message': '👋 Hello! I\'m your MediRAG medical assistant. Ask me about symptoms, treatments, medications, or any health-related questions. I\'ll provide information based on trusted medical sources.',
        
        # Disclaimer
        'disclaimer': '⚕️ This is an AI assistant for informational purposes only. Always consult healthcare professionals for medical advice.',
        
        # Feedback
        'feedback_thanks': '✅ Thank you for your feedback!',
        'feedback_error': '❌ Could not save feedback',
        
        # Errors
        'unauthorized': 'Unauthorized',
        'please_type': 'Please type a question.',
        'ui_error': 'Sorry, I encountered an error',
        
        # Landing page
        'tagline': 'Your AI-Powered Medical Assistant',
        'hero_text': 'Get instant, reliable answers to your medical questions using advanced RAG technology. Powered by trusted medical knowledge sources.',
        'get_started': 'Get Started Free',
        'learn_more': 'Learn More',
        'why_choose': 'Why Choose MediRAG?',
        'features_subtitle': 'Advanced features to help you understand medical information better',
        'how_it_works': 'How It Works',
        'how_it_works_subtitle': 'Get medical information in three simple steps',
        'step1_title': 'Ask Your Question',
        'step1_desc': 'Type or speak your medical query. Ask about symptoms, treatments, or medications.',
        'step2_title': 'AI Analysis',
        'step2_desc': 'Our RAG engine searches trusted medical sources and generates accurate responses.',
        'step3_title': 'Get Answers',
        'step3_desc': 'Receive comprehensive answers with sources. Rate responses to help improve accuracy.',
        'ready_to_start': 'Ready to Get Started?',
        'ready_text': 'Join thousands of users who trust MediRAG for their medical information needs',
        'create_account': 'Create Free Account',
        'footer_text': '© 2024 MediRAG - AI Medical Assistant. Not a substitute for professional medical advice.',
        
        # Feature cards
        'feature_rag': 'RAG Technology',
        'feature_rag_desc': 'Retrieval-Augmented Generation ensures accurate, context-aware responses from trusted medical sources.',
        'feature_voice': 'Voice Input',
        'feature_voice_desc': 'Ask questions naturally with speech-to-text support. Just click the mic and speak.',
        'feature_tts': 'Text-to-Speech',
        'feature_tts_desc': 'Listen to responses with our speech synthesis feature. Great for accessibility.',
        'feature_rating': 'Rate Responses',
        'feature_rating_desc': 'Help us improve by rating answers. Your feedback makes the system smarter.',
        'feature_history': 'Chat History',
        'feature_history_desc': 'Access your previous conversations anytime. Never lose important information.',
        'feature_security': 'Secure & Private',
        'feature_security_desc': 'Your conversations are encrypted and private. We prioritize your data security.',
        
        # Stats
        'queries_answered': 'Medical Queries Answered',
        'satisfaction': 'User Satisfaction',
        'availability': '24/7 Availability',
        
        # Login/Register
        'welcome_back': 'Welcome Back',
        'sign_in_to': 'Sign in to continue to your medical assistant',
        'username': 'Username',
        'password': 'Password',
        'enter_username': 'Enter your username',
        'enter_password': 'Enter your password',
        'no_account': 'Don\'t have an account?',
        'create_one': 'Create one',
        'admin_access': 'Admin Access',
        'admin_note': 'Use username admin with password admin123',
        'back_to_home': '← Back to Home',
        'create_account_title': 'Create Account',
        'join_text': 'Join MediRAG for instant medical information',
        'choose_username': 'Choose a username',
        'create_password': 'Create a password',
        'already_account': 'Already have an account?',
        'sign_in': 'Sign in',
        'username_min': 'Minimum 3 characters',
        'password_min': 'Minimum 4 characters',
        'username_reserved': 'This username is reserved. Please choose another.',
        'invalid_credentials': 'Invalid username or password',
    },
    'hi': {
        # Navigation
        'home': 'होम',
        'chat': 'चैट',
        'login': 'लॉगिन',
        'register': 'रजिस्टर',
        'logout': 'लॉगआउट',
        'profile': 'प्रोफाइल',
        
        # Chat interface
        'medical_assistant': 'मेडीआरएजी असिस्टेंट',
        'type_message': 'अपना संदेश लिखें...',
        'send': 'भेजें',
        'voice_input': 'वॉइस इनपुट',
        'text_to_speech': 'टेक्स्ट टू स्पीच',
        'stop_speaking': 'बोलना बंद करें',
        'listening': 'सुन रहा है...',
        'offline': 'ऑफलाइन',
        'not_supported': 'समर्थित नहीं',
        'error': 'त्रुटि',
        'thinking': 'सोच रहा है',
        'you': 'आप',
        'assistant': 'मेडीआरएजी असिस्टेंट',
        'listen': 'सुनें',
        'helpful': 'उपयोगी',
        'not_helpful': 'उपयोगी नहीं',
        'sources': 'स्रोत',
        'latency': 'विलंबता',
        
        # Welcome message
        'welcome_message': '👋 नमस्ते! मैं आपका मेडीआरएजी मेडिकल असिस्टेंट हूं। मुझसे लक्षणों, उपचारों, दवाओं या स्वास्थ्य संबंधी किसी भी प्रश्न के बारे में पूछें। मैं विश्वसनीय चिकित्सा स्रोतों के आधार पर जानकारी प्रदान करूंगा।',
        
        # Disclaimer
        'disclaimer': '⚕️ यह केवल सूचनात्मक उद्देश्यों के लिए एक AI असिस्टेंट है। चिकित्सा सलाह के लिए हमेशा स्वास्थ्य देखभाल पेशेवरों से परामर्श लें।',
        
        # Feedback
        'feedback_thanks': '✅ आपकी प्रतिक्रिया के लिए धन्यवाद!',
        'feedback_error': '❌ प्रतिक्रिया सहेजी नहीं जा सकी',
        
        # Errors
        'unauthorized': 'अनधिकृत',
        'please_type': 'कृपया एक प्रश्न लिखें।',
        'ui_error': 'क्षमा करें, एक त्रुटि हुई',
        
        # Landing page
        'tagline': 'आपका AI-पावर्ड मेडिकल असिस्टेंट',
        'hero_text': 'उन्नत RAG तकनीक का उपयोग करके अपने चिकित्सा प्रश्नों के तुरंत, विश्वसनीय उत्तर प्राप्त करें। विश्वसनीय चिकित्सा ज्ञान स्रोतों द्वारा संचालित।',
        'get_started': 'मुफ्त शुरू करें',
        'learn_more': 'और जानें',
        'why_choose': 'मेडीआरएजी क्यों चुनें?',
        'features_subtitle': 'चिकित्सा जानकारी को बेहतर समझने में मदद करने वाली उन्नत सुविधाएं',
        'how_it_works': 'यह कैसे काम करता है',
        'how_it_works_subtitle': 'तीन सरल चरणों में चिकित्सा जानकारी प्राप्त करें',
        'step1_title': 'अपना प्रश्न पूछें',
        'step1_desc': 'अपना चिकित्सा प्रश्न टाइप करें या बोलें। लक्षणों, उपचारों या दवाओं के बारे में पूछें।',
        'step2_title': 'AI विश्लेषण',
        'step2_desc': 'हमारा RAG इंजन विश्वसनीय चिकित्सा स्रोतों की खोज करता है और सटीक उत्तर उत्पन्न करता है।',
        'step3_title': 'उत्तर प्राप्त करें',
        'step3_desc': 'स्रोतों के साथ व्यापक उत्तर प्राप्त करें। सटीकता बढ़ाने में मदद के लिए प्रतिक्रियाओं को रेट करें।',
        'ready_to_start': 'शुरू करने के लिए तैयार हैं?',
        'ready_text': 'हजारों उपयोगकर्ताओं से जुड़ें जो अपनी चिकित्सा जानकारी की जरूरतों के लिए मेडीआरएजी पर भरोसा करते हैं',
        'create_account': 'मुफ्त खाता बनाएं',
        'footer_text': '© 2024 मेडीआरएजी - AI मेडिकल असिस्टेंट। पेशेवर चिकित्सा सलाह का विकल्प नहीं।',
        
        # Feature cards
        'feature_rag': 'RAG तकनीक',
        'feature_rag_desc': 'रिट्रीवल-ऑग्मेंटेड जेनरेशन विश्वसनीय चिकित्सा स्रोतों से सटीक, संदर्भ-जागरूक प्रतिक्रियाएं सुनिश्चित करता है।',
        'feature_voice': 'वॉइस इनपुट',
        'feature_voice_desc': 'स्पीच-टू-टेक्स्ट समर्थन के साथ स्वाभाविक रूप से प्रश्न पूछें। बस माइक पर क्लिक करें और बोलें।',
        'feature_tts': 'टेक्स्ट-टू-स्पीच',
        'feature_tts_desc': 'हमारी स्पीच सिंथेसिस सुविधा के साथ प्रतिक्रियाएं सुनें। पहुंच के लिए बढ़िया।',
        'feature_rating': 'प्रतिक्रियाओं को रेट करें',
        'feature_rating_desc': 'उत्तरों को रेट करके हमें सुधारने में मदद करें। आपकी प्रतिक्रिया सिस्टम को स्मार्ट बनाती है।',
        'feature_history': 'चैट इतिहास',
        'feature_history_desc': 'कभी भी अपनी पिछली बातचीत तक पहुंचें। महत्वपूर्ण जानकारी कभी न खोएं।',
        'feature_security': 'सुरक्षित और निजी',
        'feature_security_desc': 'आपकी बातचीत एन्क्रिप्टेड और निजी है। हम आपके डेटा सुरक्षा को प्राथमिकता देते हैं।',
        
        # Stats
        'queries_answered': 'चिकित्सा प्रश्नों के उत्तर दिए',
        'satisfaction': 'उपयोगकर्ता संतुष्टि',
        'availability': '24/7 उपलब्धता',
        
        # Login/Register
        'welcome_back': 'वापस स्वागत है',
        'sign_in_to': 'अपने मेडिकल असिस्टेंट में जारी रखने के लिए साइन इन करें',
        'username': 'उपयोगकर्ता नाम',
        'password': 'पासवर्ड',
        'enter_username': 'अपना उपयोगकर्ता नाम दर्ज करें',
        'enter_password': 'अपना पासवर्ड दर्ज करें',
        'no_account': 'खाता नहीं है?',
        'create_one': 'बनाएं',
        'admin_access': 'एडमिन एक्सेस',
        'admin_note': 'उपयोगकर्ता नाम admin और पासवर्ड admin123 का उपयोग करें',
        'back_to_home': '← होम पर वापस',
        'create_account_title': 'खाता बनाएं',
        'join_text': 'त्वरित चिकित्सा जानकारी के लिए मेडीआरएजी से जुड़ें',
        'choose_username': 'उपयोगकर्ता नाम चुनें',
        'create_password': 'पासवर्ड बनाएं',
        'already_account': 'पहले से खाता है?',
        'sign_in': 'साइन इन करें',
        'username_min': 'न्यूनतम 3 अक्षर',
        'password_min': 'न्यूनतम 4 अक्षर',
        'username_reserved': 'यह उपयोगकर्ता नाम आरक्षित है। कृपया दूसरा चुनें।',
        'invalid_credentials': 'गलत उपयोगकर्ता नाम या पासवर्ड',
    }
}

rag = RAGEngine(top_k=4)

init_db()

@app.context_processor
def inject_language():
    """Inject language and translations into all templates"""
    if 'language' not in session:
        session['language'] = 'en'
    
    def get_text(key):
        lang = session.get('language', 'en')
        return TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, key)
    
    return dict(
        get_text=get_text,
        current_language=session.get('language', 'en'),
        languages=LANGUAGES
    )

@app.route('/set-language/<lang>')
def set_language(lang):
    """Set user's preferred language"""
    if lang in LANGUAGES:
        session['language'] = lang
    return redirect(request.referrer or url_for('home'))

@app.route("/")
def home():
    """Landing page"""
    if "username" in session:
        return redirect(url_for("chat"))
    return render_template("landing.html", languages=LANGUAGES)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if username == ADMIN_USERNAME:
            error_msg = TRANSLATIONS.get(session.get('language', 'en'), TRANSLATIONS['en']).get('username_reserved', 'This username is reserved')
            return render_template("register.html", error=error_msg)

        ok, msg = create_user(username, password)
        if ok:
            session["username"] = username
            session["is_admin"] = False
            return redirect(url_for("chat"))
        return render_template("register.html", error=msg)

    return render_template("register.html", error=None)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if username == ADMIN_USERNAME:
            if password == ADMIN_PASSWORD:
                session["username"] = username
                session["is_admin"] = True
                return redirect(url_for("chat"))
            error_msg = TRANSLATIONS.get(session.get('language', 'en'), TRANSLATIONS['en']).get('invalid_credentials', 'Invalid credentials')
            return render_template("login.html", error=error_msg)

        if verify_user(username, password):
            session["username"] = username
            session["is_admin"] = False
            return redirect(url_for("chat"))

        error_msg = TRANSLATIONS.get(session.get('language', 'en'), TRANSLATIONS['en']).get('invalid_credentials', 'Invalid credentials')
        return render_template("login.html", error=error_msg)

    return render_template("login.html", error=None)

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/chat")
def chat():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template(
        "chat.html",
        username=session["username"],
        is_admin=bool(session.get("is_admin", False)),
        languages=LANGUAGES
    )

@app.route("/api/history", methods=["GET"])
def api_history():
    if "username" not in session:
        error_msg = TRANSLATIONS.get(session.get('language', 'en'), TRANSLATIONS['en']).get('unauthorized', 'Unauthorized')
        return jsonify({"error": error_msg}), 401
    history = get_chat_history(session["username"], limit=200)
    return jsonify({"history": history})

@app.route("/api/chat", methods=["POST"])
def api_chat():
    if "username" not in session:
        error_msg = TRANSLATIONS.get(session.get('language', 'en'), TRANSLATIONS['en']).get('unauthorized', 'Unauthorized')
        return jsonify({"error": error_msg}), 401

    data = request.get_json(force=True)
    msg = (data.get("message") or "").strip()
    if not msg:
        please_type_msg = TRANSLATIONS.get(session.get('language', 'en'), TRANSLATIONS['en']).get('please_type', 'Please type a question.')
        return jsonify({"reply": please_type_msg})

    # persist user message
    save_chat(session["username"], "user", msg)

    t0 = time.perf_counter()
    try:
        reply, sources = rag.answer(msg)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        
        # persist bot reply
        save_chat(session["username"], "bot", reply, latency_ms=latency_ms)
        
        # server-side logging
        print(f"[CHAT] user={session['username']} latency_ms={latency_ms} msg_len={len(msg)}")
        
        # Get disclaimer in user's language
        disclaimer_text = TRANSLATIONS.get(session.get('language', 'en'), TRANSLATIONS['en']).get('disclaimer', '⚕️ This is an AI assistant for informational purposes only.')
        disclaimer = f"\n\n{disclaimer_text}"
        
        return jsonify({
            "reply": reply + disclaimer,
            "sources": sources,
            "latency_ms": latency_ms,
            "exchange": {"user_message": msg, "bot_reply": reply, "latency_ms": latency_ms}
        })
    except Exception as e:
        print(f"[ERROR] in api_chat: {str(e)}")
        error_msg = TRANSLATIONS.get(session.get('language', 'en'), TRANSLATIONS['en']).get('ui_error', 'Sorry, I encountered an error')
        return jsonify({
            "error": f"{error_msg}: {str(e)}",
            "reply": error_msg,
            "sources": [],
            "latency_ms": 0
        }), 500

@app.route("/api/rate", methods=["POST"])
def api_rate():
    if "username" not in session:
        error_msg = TRANSLATIONS.get(session.get('language', 'en'), TRANSLATIONS['en']).get('unauthorized', 'Unauthorized')
        return jsonify({"error": error_msg}), 401

    data = request.get_json(force=True)
    helpful = data.get("helpful")
    user_message = (data.get("user_message") or "").strip()
    bot_reply = (data.get("bot_reply") or "").strip()
    latency_ms = data.get("latency_ms", None)

    if helpful is None or not user_message or not bot_reply:
        return jsonify({"error": "Missing fields"}), 400

    save_rating(session["username"], user_message, bot_reply, 1 if helpful else 0, latency_ms)
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)