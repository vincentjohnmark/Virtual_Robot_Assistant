# assistant.py

import os
from dotenv import load_dotenv
import speech_recognition as sr
import pyttsx3
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Load Hugging Face token
load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not token:
    raise ValueError("Missing HUGGINGFACEHUB_API_TOKEN in .env")

# Initialize text-to-speech
engine = pyttsx3.init()

def speak(text):
    print(f"🤖 Robot: {text}")
    engine.say(text)
    engine.runAndWait()

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n🎤 Listening...")
        r.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=6)
        except sr.WaitTimeoutError:
            print("⌛ Timeout: No speech detected.")
            return ""
    try:
        command = r.recognize_google(audio)
        print(f"You: {command}")
        return command.lower()
    except sr.UnknownValueError:
        speak("Sorry, I didn't catch that.")
        return ""

# Load DeepSeek Model
print("🧠 Loading DeepSeek model...")
model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", use_auth_token=token)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Generate response
def ask_deepseek(prompt):
    print("🤖 Thinking...")
    response = generator(prompt, max_new_tokens=300, temperature=0.7, do_sample=True)[0]['generated_text']
    return response.replace(prompt, "").strip()

# Main loop
if __name__ == "__main__":
    speak("Virtual Robot Assistant ready. Using DeepSeek AI for answers.")
    mode = "smart"

    while True:
        query = listen()
        if not query:
            continue
        if "exit" in query or "quit" in query:
            speak("Goodbye! Shutting down.")
            break
        elif "use chat" in query:
            mode = "chat"
            speak("Chat mode is same in DeepSeek. Ready!")
            continue
        elif "be smart" in query:
            mode = "smart"
            speak("Smart Assistant Mode activated.")
            continue

        prompt = f"User: {query}\nAssistant:"
        reply = ask_deepseek(prompt)
        speak(reply)
