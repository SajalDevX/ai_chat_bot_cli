# test_providers.py
"""Test both Gemini and OpenAI providers."""
from ai_chat_bot import GroqClient  # Add import

from ai_chat_bot import (
    GeminiClient,
    Conversation,
    get_settings,
)

def test_provider(client_class, name):
    """Test a single provider."""
    print(f"\n{'=' * 50}")
    print(f"Testing {name}")
    print('=' * 50)
    
    try:
        with client_class() as client:
            print(f"✅ Provider: {client.PROVIDER_NAME}")
            print(f"✅ Base class: {client.__class__.__bases__[0].__name__}")
            
            # Create conversation
            conv = Conversation()
            conv.set_system_prompt("You are helpful. Reply in 10 words or less.")
            conv.add_user_message("What is Python?")
            
            # Test regular chat
            print("\n📤 Sending message...")
            response = client.chat(conv)
            
            print(f"📥 Response: {response.content}")
            print(f"📊 Tokens: {response.usage}")
            print(f"💰 Cost: ${response.cost:.6f}")
            
            # Test streaming
            print("\n🌊 Testing streaming...")
            conv2 = Conversation()
            conv2.add_user_message("Count 1 to 3.")
            
            print("   Stream: ", end="")
            for chunk in client.chat_stream(conv2):
                print(chunk, end="", flush=True)
            print()
            
            print(f"\n✅ {name} passed all tests!")
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    settings = get_settings()
    print("Available providers:", settings.available_providers())
    
    results = {}
    
    # Test Gemini
    if settings.has_gemini():
        results["Gemini"] = test_provider(GeminiClient, "Gemini")
    else:
        print("\n⚠️ Gemini: Skipped (no API key)")
        results["Gemini"] = None
    

    if settings.has_groq():
        results["Groq"] = test_provider(GroqClient, "Groq")
    else:
        print("\n⚠️ Groq: Skipped (no API key)")
        results["Groq"] = None
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for provider, passed in results.items():
        if passed is None:
            print(f"  {provider}: ⚠️ Skipped")
        elif passed:
            print(f"  {provider}: ✅ Passed")
        else:
            print(f"  {provider}: ❌ Failed")


if __name__ == "__main__":
    main()