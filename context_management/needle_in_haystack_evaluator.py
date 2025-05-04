import json
import openai
from dotenv import load_dotenv
import os
from openai.error import OpenAIError, RateLimitError
import statistics

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
API_KEY = os.getenv("OPENAI_API_KEY")

# --- Configuration ---
SUMMARIZER_MODEL = "gpt-4o"
EVALUATOR_MODEL = "gpt-4o-mini"

# --- Summarization Strategies (Prompts) ---
SUMMARIZATION_STRATEGIES = {
    "Standard Concise": {
        "system_prompt": "You are an expert summarizer. Condense the following text, focusing on the main points and key information, while maintaining clarity and coherence.",
        "user_prompt_template": "Please summarize the following text concisely:\n\n---\n{text}\n---"
    },
    "Key Bullet Points": {
        "system_prompt": "You are an expert summarizer. Extract the key bullet points or main takeaways from the following text.",
        "user_prompt_template": "Extract the key bullet points from this text:\n\n---\n{text}\n---"
    },
    "Abstractive TL;DR": {
        "system_prompt": "You are an expert at creating very short, high-level summaries (like a TL;DR). Capture the absolute essence of the text in 1-2 sentences.",
        "user_prompt_template": "Provide a 1-2 sentence TL;DR summary for this text:\n\n---\n{text}\n---"
    },
    "Question-Focused (Who, What, Why)": {
        "system_prompt": "You are an expert summarizer. Summarize the text by primarily answering the questions: Who was involved? What happened? Why did it happen?",
        "user_prompt_template": "Summarize this text, focusing on Who, What, and Why:\n\n---\n{text}\n---"
    }
}

# --- Helper Functions ---

def get_openai_client(api_key):
    """Initializes and returns the OpenAI client."""
    if not api_key:
        raise ValueError("API Key is missing!")
    try:
        openai.api_key = api_key
        return openai
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenAI client: {e}")


def generate_summary(_client, original_text, strategy_name, max_tokens=500):
    """Calls OpenAI API to generate a summary using a specific strategy."""
    if not _client or not original_text or strategy_name not in SUMMARIZATION_STRATEGIES:
        raise ValueError("Invalid input for summarization.")

    strategy = SUMMARIZATION_STRATEGIES[strategy_name]
    system_prompt = strategy["system_prompt"]
    user_prompt = strategy["user_prompt_template"].format(text=original_text)

    try:
        response = openai.ChatCompletion.create(
            model=SUMMARIZER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.5
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except RateLimitError:
        raise RuntimeError("OpenAI API rate limit exceeded. Please try again later.")
    except OpenAIError as e:
        raise RuntimeError(f"OpenAI API error: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during summarization: {e}")


def evaluate_needle_in_haystack(_client, original_text, summary_text, key_information):
    """Evaluates if the summary contains the key information (needle in a haystack)."""
    if not _client or not original_text or not summary_text or not key_information:
        raise ValueError("Invalid input for evaluation.")

    # More flexible detection of key information
    results = {
        "found_information": [],
        "missing_information": [],
        "partially_found_information": []
    }
    
    for info in key_information:
        # Check if the key information is present exactly
        if info.lower() in summary_text.lower():
            results["found_information"].append({
                "info": info,
                "status": "exact match"
            })
        else:
            # Check for partial matches using key terms
            key_terms = [term for term in info.lower().split() if len(term) > 3]  # Only use significant words
            matched_terms = [term for term in key_terms if term.lower() in summary_text.lower()]
            
            # If more than half of the key terms are found, consider it a partial match
            if matched_terms and len(matched_terms) >= len(key_terms) / 2:
                results["partially_found_information"].append({
                    "info": info,
                    "matched_terms": matched_terms,
                    "status": "partial match"
                })
            else:
                results["missing_information"].append({
                    "info": info,
                    "status": "not found"
                })
    
    # Determine overall success based on found and partially found information
    success = len(results["found_information"]) + len(results["partially_found_information"]) >= len(key_information) / 2
    
    # Calculate a score from 0-100
    total_info = len(key_information)
    exact_matches = len(results["found_information"])
    partial_matches = len(results["partially_found_information"])
    score = int((exact_matches * 1.0 + partial_matches * 0.5) / total_info * 100)
    
    return {
        "success": success,
        "results": results,
        "score": score,
        "message": "Summary evaluation complete." if success else "Summary is missing key information."
    }


def main(api_key, conversations):
    """Main function to evaluate all summarization strategies across multiple conversations."""
    client = get_openai_client(api_key)
    
    all_results = {}
    strategy_scores = {strategy: [] for strategy in SUMMARIZATION_STRATEGIES.keys()}
    
    for idx, conv in enumerate(conversations):
        print(f"Processing conversation {idx+1}: {conv['name']}")
        conv_results = {}
        for strategy_name in SUMMARIZATION_STRATEGIES.keys():
            try:
                summary = generate_summary(client, conv["text"], strategy_name)
                evaluation = evaluate_needle_in_haystack(client, conv["text"], summary, conv["key_information"])
                
                # Store the score for averaging later
                strategy_scores[strategy_name].append(evaluation["score"])
                
                conv_results[strategy_name] = {
                    "summary": summary,
                    "evaluation": evaluation
                }
                print(f"  - {strategy_name}: Score {evaluation['score']}/100")
            except Exception as e:
                conv_results[strategy_name] = {
                    "error": str(e)
                }
                print(f"  - {strategy_name}: Error - {str(e)}")
        
        all_results[f"conversation_{idx+1}"] = conv_results
    
    # Calculate average scores for each strategy
    average_scores = {}
    for strategy, scores in strategy_scores.items():
        if scores:  # Check if we have valid scores
            average_scores[strategy] = {
                "average_score": statistics.mean(scores),
                "median_score": statistics.median(scores),
                "min_score": min(scores),
                "max_score": max(scores)
            }
    
    return {
        "individual_results": all_results,
        "strategy_performance": average_scores
    }

# Example conversations
EXAMPLE_CONVERSATIONS = [
    {
        "name": "Travel Planning Conversation",
        "text": """
        User: I need help planning a trip to Japan next spring.
        
        Agent: I'd be happy to help you plan your trip to Japan! When exactly in spring are you thinking of going, and how long will your trip be?
        
        User: We're thinking of going for 2 weeks in April to see the cherry blossoms. There will be 4 of us - me, my wife, and our two teenage kids.
        
        Agent: April is a wonderful time to visit Japan for cherry blossoms! The peak bloom varies by region, typically starting in late March in the south and reaching Tokyo and Kyoto in early April.
        
        For a 2-week itinerary with your family, I'd recommend:
        
        1. Start in Tokyo (4-5 days)
        2. Take a day trip to Hakone or Nikko
        3. Travel to Kyoto (3-4 days)
        4. Visit nearby Nara and Osaka
        5. Consider Hiroshima and Miyajima (2 days)
        6. End in Tokyo (1-2 days)
        
        User: That sounds great. We definitely want to see Tokyo and Kyoto. What are the must-see spots in Tokyo for teenagers?
        
        Agent: For teenagers in Tokyo, I recommend: Shibuya Crossing, Harajuku's Takeshita Street, Akihabara for electronics and anime, TeamLab Borderless digital art museum, Tokyo Disneyland, Shinjuku for neon lights and karaoke, and Odaiba entertainment island. The Ghibli Museum is also great if they like those films, but book tickets starting on the 10th of each month at 10:00 AM Japan time.
        
        User: Those sound perfect. What about hotels? We'd prefer to stay somewhere central in Tokyo.
        
        Agent: For central Tokyo hotels good for families, consider mid-range options like Shinjuku Prince Hotel or Shibuya Excel Hotel Tokyu. For upscale, try Park Hyatt Tokyo or Hilton Tokyo. Budget options include UNPLAN Shinjuku or Centurion Hotel Grand Akasaka. Look for "family rooms" as regular rooms in Japan tend to be small. Book 3-4 months in advance for cherry blossom season.
        
        User: I think we'll go with the Shinjuku Prince Hotel. What's the best way to get from Narita Airport to Shinjuku?
        
        Agent: From Narita Airport to Shinjuku, your best options are:
        
        1. Narita Express (N'EX): Direct train taking 80-90 minutes, costs 3,270 yen per person.
        
        2. Airport Limousine Bus: Direct to Shinjuku Prince Hotel, 90-120 minutes depending on traffic, costs 3,200 yen per person.
        
        3. Keisei Skyliner + JR Line: Take Skyliner to Nippori then transfer to JR Yamanote Line to Shinjuku, total cost about 2,670 yen.
        
        4. Taxi: Most expensive (20,000-25,000 yen) but most convenient.
        
        For a family with luggage, I recommend the Airport Limousine Bus or Narita Express.
        """,
        "key_information": [
            "April is cherry blossom season in Japan",
            "Ghibli Museum tickets open on the 10th of each month",
            "Airport Limousine Bus costs about 3,200 yen per person",
            "Narita Express takes 80-90 minutes to Shinjuku",
            "TeamLab Borderless is recommended for teenagers"
        ]
    },
    {
        "name": "Programming Help Conversation",
        "text": """
        User: I'm trying to build a React component that fetches data from an API but I'm getting a weird error about hooks.
        
        Agent: I'd be happy to help with your React hooks error. These errors usually occur due to specific rules of using hooks. Could you share your component code and the exact error message you're seeing?
        
        User: Here's the component:
        ```javascript
        function DataFetcher(props) {
          const [data, setData] = useState([]);
          const [loading, setLoading] = useState(true);
          
          if (props.filter) {
            const [filteredData, setFilteredData] = useState([]);
          }
          
          useEffect(() => {
            async function fetchData() {
              setLoading(true);
              try {
                const response = await fetch('https://api.example.com/data');
                const json = await response.json();
                setData(json);
                if (props.filter) {
                  setFilteredData(json.filter(props.filter));
                }
              } catch (error) {
                console.error('Error fetching data:', error);
              } finally {
                setLoading(false);
              }
            }
            
            fetchData();
          }, [props.filter]);
          
          return (
            <div>
              {loading ? (
                <p>Loading...</p>
              ) : (
                <ul>
                  {(props.filter ? filteredData : data).map(item => (
                    <li key={item.id}>{item.name}</li>
                  ))}
                </ul>
              )}
            </div>
          );
        }
        ```
        
        And the error is: "React Hook "useState" is called conditionally. React Hooks must be called in the exact same order in every component render."
        
        Agent: The issue is that you're calling useState conditionally:
        
        ```javascript
        if (props.filter) {
          const [filteredData, setFilteredData] = useState([]);
        }
        ```
        
        React hooks must be called in the same order every render. Here's the fix:
        
        ```javascript
        function DataFetcher(props) {
          const [data, setData] = useState([]);
          const [filteredData, setFilteredData] = useState([]); // Always define this
          const [loading, setLoading] = useState(true);
          
          useEffect(() => {
            async function fetchData() {
              setLoading(true);
              try {
                const response = await fetch('https://api.example.com/data');
                const json = await response.json();
                setData(json);
                if (props.filter) {
                  setFilteredData(json.filter(props.filter));
                } else {
                  setFilteredData([]);
                }
              } catch (error) {
                console.error('Error fetching data:', error);
              } finally {
                setLoading(false);
              }
            }
            
            fetchData();
          }, [props.filter]);
          
          const displayData = props.filter ? filteredData : data;
          
          return (
            <div>
              {loading ? (
                <p>Loading...</p>
              ) : (
                <ul>
                  {displayData.map(item => (
                    <li key={item.id}>{item.name}</li>
                  ))}
                </ul>
              )}
            </div>
          );
        }
        ```
        
        User: That worked! Thanks. One more question - sometimes the API returns an error and I want to show that to the user. What's the best way to do that?
        
        Agent: Add an error state to show API errors to users:
        
        ```javascript
        function DataFetcher(props) {
          const [data, setData] = useState([]);
          const [filteredData, setFilteredData] = useState([]);
          const [loading, setLoading] = useState(true);
          const [error, setError] = useState(null); // New error state
          
          useEffect(() => {
            async function fetchData() {
              setLoading(true);
              setError(null); // Reset error state
              
              try {
                const response = await fetch('https://api.example.com/data');
                
                if (!response.ok) {
                  throw new Error(`API error: ${response.status}`);
                }
                
                const json = await response.json();
                setData(json);
                if (props.filter) {
                  setFilteredData(json.filter(props.filter));
                } else {
                  setFilteredData([]);
                }
              } catch (error) {
                console.error('Error fetching data:', error);
                setError(error.message || 'An unknown error occurred');
                setData([]);
                setFilteredData([]);
              } finally {
                setLoading(false);
              }
            }
            
            fetchData();
          }, [props.filter]);
          
          // Show error message if there's an error
          if (error) {
            return (
              <div className="error-container">
                <p className="error-message">Error: {error}</p>
                <button onClick={() => window.location.reload()}>Retry</button>
              </div>
            );
          }
          
          // Rest of component remains the same
          // ...
        }
        ```
        
        User: This is great! One last thing - how do I add a loading spinner instead of the "Loading..." text?
        
        Agent: You can add a loading spinner using the react-spinners library:
        
        ```
        npm install react-spinners
        ```
        
        Then update your component:
        
        ```javascript
        import { ClipLoader } from "react-spinners";
        
        function DataFetcher(props) {
          // ...existing code...
          
          return (
            <div>
              {loading ? (
                <div className="loading-container" style={{ textAlign: 'center', padding: '20px' }}>
                  <ClipLoader 
                    color={"#36D7B7"} 
                    loading={loading} 
                    size={50} 
                  />
                  <p style={{ marginTop: '10px' }}>Loading data...</p>
                </div>
              ) : (
                <ul>
                  {displayData.map(item => (
                    <li key={item.id}>{item.name}</li>
                  ))}
                </ul>
              )}
            </div>
          );
        }
        ```
        
        The library offers many spinner styles like BeatLoader, PulseLoader, RingLoader, etc.
        """,
        "key_information": [
            "React hooks must be called in the same order on every render",
            "useState cannot be called conditionally",
            "ClipLoader is from react-spinners library",
            "Use response.ok to check for API errors",
            "Set displayData = props.filter ? filteredData : data"
        ]
    },
    {
        "name": "Customer Support Conversation",
        "text": """
        User: My subscription was just charged but I can't access my premium account. My account email is john.smith@example.com.
        
        Agent: I'm sorry to hear about your premium access issue. I've looked up your account using john.smith@example.com and see your $49.99 payment was processed yesterday, but there's a flag preventing activation. Let me remove that flag. Please wait 2 minutes and refresh your account page.
        
        User: I waited but it's still not working. When I log in it says "Free tier account" at the top.
        
        Agent: I apologize for the continued issue. The payment was mistakenly assigned to an older subscription ID. I've manually updated your account to Premium status and merged the records. Please:
        
        1. Log out completely
        2. Clear your browser cache (or use incognito mode)
        3. Log back in with your email john.smith@example.com
        
        If using our mobile app, force close and reopen it.
        
        User: That worked! Thank you. How can I make sure this doesn't happen next month? My subscription is annual but I worried it might not renew properly.
        
        Agent: I'm glad it's working! Your annual subscription for $49.99 was just renewed yesterday and is secured until May 15, 2024. I've added a note to your account and applied a special flag to ensure proper renewal processing next year.
        
        For added precaution:
        
        1. Bookmark our customer support line: 1-800-555-7890
        2. Save your confirmation ID: PRE-2023051501A
        3. Keep your payment information up to date
        
        If you have future issues, reference that confirmation ID for quicker support.
        
        User: No, that's all I needed. Thanks for your help!
        
        Agent: You're welcome! I'm happy I could help. To summarize:
        
        1. Fixed your premium access
        2. Confirmed your subscription is active until May 15, 2024
        3. Added protection against future issues
        4. Provided your confirmation ID (PRE-2023051501A)
        
        Our support team is available 24/7 if you need anything else. Enjoy your premium features!
        """,
        "key_information": [
            "Account email is john.smith@example.com",
            "Annual subscription cost is $49.99",
            "Subscription renewal date is May 15, 2024",
            "Customer support line is 1-800-555-7890",
            "Confirmation ID is PRE-2023051501A"
        ]
    }
]

# Run the evaluation
if __name__ == "__main__":
    print("Starting evaluation of summarization strategies...")
    results = main(API_KEY, EXAMPLE_CONVERSATIONS)
    print("\n--- AVERAGE PERFORMANCE ACROSS ALL CONVERSATIONS ---")
    print(json.dumps(results["strategy_performance"], indent=2)) 