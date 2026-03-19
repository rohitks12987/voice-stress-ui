const faqData = {
    // Greetings
    "hii": "Hello! I'm your AI Wellness Officer. How can I help you today?",
    "hello": "Hi there! Ready to check your stress levels?",
    "hey": "Hey! How are you feeling right now?",
    "how are you": "I'm doing great! My systems are ready to analyze your wellness.",
    
    // About the System
    "what is this": "This is an AI-powered Mental Stress Detection system that analyzes your voice tone.",
    "how does it work": "We analyze vocal patterns like pitch and jitter to predict stress levels.",
    "is it accurate": "It provides a probability based on vocal biomarkers, but it's not a medical diagnosis.",
    "who made this": "This is an official portal managed by the Ministry of Mental Health & Wellness.",
    
    // Help & Usage
    "how to record": "Go to the 'Record' page and click the microphone icon to start speaking.",
    "how long to speak": "For best results, speak naturally for about 10 to 15 seconds.",
    "where is my history": "You can view all your past results in the 'History' tab.",
    "can i delete history": "Currently, history is stored for your wellness tracking, but you can contact support for data requests.",
    
    // Stress & Wellness Tips
    "what should i do": "You can start by recording your voice to get a stress analysis, or ask me for a breathing exercise.",
    "i feel sad": "I'm sorry to hear that. Acknowledge your feelings, take a deep breath, and consider talking to a friend or professional.",
    "help me relax": "Try closing your eyes and taking three deep breaths. Inhale for 4, hold for 4, exhale for 4.",
    "i am stressed": "I'm sorry to hear that. Please try a short walk or some water. You can also click the EMERGENCY button if it's too much.",
    "tips for anxiety": "Focus on 5 things you can see, 4 things you can touch, and 3 things you can hear.",
    "breathing exercise": "Follow the 4-7-8 rule: Breathe in 4s, hold 7s, exhale 8s.",
    "music for stress": "Lo-fi beats or classical music are scientifically proven to lower cortisol.",
    
    // App Features & Custom Points
    "how to change profile photo": "Go to Settings on the sidebar, click 'Edit Profile', and use the camera icon to upload a new photo.",
    "is my data safe": "Yes, your data is securely stored locally and only shared according to your privacy settings.",

    // Emergency
    "emergency": "If you are in danger, click the RED SOS button or call 1-800-HEALTH-123 immediately.",
    "suicide": "Please reach out to the National Suicide Prevention Lifeline at 988. You are not alone.",
    "sos": "Click the EMERGENCY button at the bottom of the screen to send your location to contacts.",

    // Common Closings
    "thank you": "You're very welcome! Take care of yourself.",
    "thanks": "No problem! I'm always here to help.",
    "bye": "Goodbye! Have a peaceful day.",
    "goodnight": "Rest well. Sleep is essential for mental health!"
};

// Default responses for unknown queries
const defaultResponses = [
    "I am tuned to analyze stress and wellness. Could you ask about your stress levels, breathing exercises, or how to use the app?",
    "I'm here to support your mental wellness. You can ask me 'How do I relax?' or 'How does this work?'.",
    "For specific medical advice, please consult a doctor. However, I can help you with stress management techniques."
];