from fastapi import FastAPI, HTTPException, Query  # Make sure Query is imported
from fastapi.responses import JSONResponse
from webscout import WEBS, transcriber, LLM
from typing import Optional, List, Dict, Union # Import List, Dict, Union
from fastapi.encoders import jsonable_encoder
from bs4 import BeautifulSoup
import requests
import urllib.parse

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "API documentation can be found at /docs"}

@app.get("/health")
async def health_check():
    return {"status": "OK"}

@app.get("/api/search")
async def search(
    q: str,
    max_results: int = 10,
    timelimit: Optional[str] = None,
    safesearch: str = "moderate",
    region: str = "wt-wt",
    backend: str = "api"
):
    """Perform a text search."""
    try:
        with WEBS() as webs:
            results = webs.text(keywords=q, region=region, safesearch=safesearch, timelimit=timelimit, backend=backend, max_results=max_results)
            return JSONResponse(content=jsonable_encoder(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during search: {e}")

@app.get("/api/images")
async def images(
    q: str,
    max_results: int = 10,
    safesearch: str = "moderate",
    region: str = "wt-wt",
    timelimit: Optional[str] = None,
    size: Optional[str] = None,
    color: Optional[str] = None,
    type_image: Optional[str] = None,
    layout: Optional[str] = None,
    license_image: Optional[str] = None
):
    """Perform an image search."""
    try:
        with WEBS() as webs:
            results = webs.images(keywords=q, region=region, safesearch=safesearch, timelimit=timelimit, size=size, color=color, type_image=type_image, layout=layout, license_image=license_image, max_results=max_results)
            return JSONResponse(content=jsonable_encoder(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during image search: {e}")

@app.get("/api/videos")
async def videos(
    q: str,
    max_results: int = 10,
    safesearch: str = "moderate",
    region: str = "wt-wt",
    timelimit: Optional[str] = None,
    resolution: Optional[str] = None,
    duration: Optional[str] = None,
    license_videos: Optional[str] = None
):
    """Perform a video search."""
    try:
        with WEBS() as webs:
            results = webs.videos(keywords=q, region=region, safesearch=safesearch, timelimit=timelimit, resolution=resolution, duration=duration, license_videos=license_videos, max_results=max_results)
            return JSONResponse(content=jsonable_encoder(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during video search: {e}")

@app.get("/api/news")
async def news(
    q: str,
    max_results: int = 10,
    safesearch: str = "moderate",
    region: str = "wt-wt",
    timelimit: Optional[str] = None
):
    """Perform a news search."""
    try:
        with WEBS() as webs:
            results = webs.news(keywords=q, region=region, safesearch=safesearch, timelimit=timelimit, max_results=max_results)
            return JSONResponse(content=jsonable_encoder(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during news search: {e}")

@app.get("/api/llm")
async def llm_chat(
    model: str,
    message: str,
    system_prompt: str = Query(None, description="Optional custom system prompt")
):
    """Interact with a specified large language model with an optional system prompt."""
    try:
        messages = [{"role": "user", "content": message}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})  # Add system message at the beginning

        llm = LLM(model=model) 
        response = llm.chat(messages=messages)
        return JSONResponse(content={"response": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during LLM chat: {e}")


@app.get("/api/answers")
async def answers(q: str):
    """Get instant answers for a query."""
    try:
        with WEBS() as webs:
            results = webs.answers(keywords=q)
            return JSONResponse(content=jsonable_encoder(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting instant answers: {e}")

@app.get("/api/suggestions")
async def suggestions(q: str, region: str = "wt-wt"):
    """Get search suggestions for a query."""
    try:
        with WEBS() as webs:
            results = webs.suggestions(keywords=q, region=region)
            return JSONResponse(content=jsonable_encoder(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting search suggestions: {e}")

@app.get("/api/chat")
async def chat(
    q: str,
    model: str = "gpt-3.5"
):
    """Perform a text search."""
    try:
        with WEBS() as webs:
            results = webs.chat(keywords=q, model=model)
            return JSONResponse(content=jsonable_encoder(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting chat results: {e}")

def extract_text_from_webpage(html_content):
    """Extracts visible text from HTML content using BeautifulSoup."""
    soup = BeautifulSoup(html_content, "html.parser")
    # Remove unwanted tags
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.extract()
    # Get the remaining visible text
    visible_text = soup.get_text(strip=True)
    return visible_text

@app.get("/api/web_extract")
async def web_extract(
    url: str,
    max_chars: int = 12000,  # Adjust based on token limit
):
    """Extracts text from a given URL."""
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0"})
        response.raise_for_status()
        visible_text = extract_text_from_webpage(response.text)
        if len(visible_text) > max_chars:
            visible_text = visible_text[:max_chars] + "..."
        return {"url": url, "text": visible_text}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching or processing URL: {e}")

@app.get("/api/search-and-extract")
async def web_search_and_extract(
    q: str,
    max_results: int = 3,
    timelimit: Optional[str] = None,
    safesearch: str = "moderate",
    region: str = "wt-wt",
    backend: str = "api",
    max_chars: int = 6000,
    extract_only: bool = False
):
    """
    Searches using WEBS, extracts text from the top results, and returns both.
    """
    try:
        with WEBS() as webs:
            # Perform WEBS search
            search_results = webs.text(keywords=q, region=region, safesearch=safesearch,
                                     timelimit=timelimit, backend=backend, max_results=max_results)

            # Extract text from each result's link
            extracted_results = []
            for result in search_results:
                if 'href' in result:
                    link = result['href']
                    try:
                        response = requests.get(link, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0"})
                        response.raise_for_status()
                        visible_text = extract_text_from_webpage(response.text)
                        if len(visible_text) > max_chars:
                            visible_text = visible_text[:max_chars] + "..."
                        extracted_results.append({"link": link, "text": visible_text})
                    except requests.exceptions.RequestException as e:
                        print(f"Error fetching or processing {link}: {e}")
                        extracted_results.append({"link": link, "text": None})
                else:
                    extracted_results.append({"link": None, "text": None})
            if extract_only:
                return JSONResponse(content=jsonable_encoder({extracted_results}))
            else:
                return JSONResponse(content=jsonable_encoder({"search_results": search_results, "extracted_results": extracted_results}))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during search and extraction: {e}")

@app.get("/api/adv_web_search")
async def adv_web_search(
    q: str,
    model: str = "gpt-3.5",
    max_results: int = 3,  
    timelimit: Optional[str] = None,
    safesearch: str = "moderate",
    region: str = "wt-wt",
    backend: str = "api",
    max_chars: int = 6000,  
    system_prompt: str = "You are Most Advanced and Powerful Ai chatbot, User ask you questions and you have to answer that, You are also provided with Google Search Results, To increase your accuracy and providing real time data. Your task is to answer in best way to user."
):
    """
    Combines web search, web extraction, and LLM chat for advanced search.
    """
    try:
        with WEBS() as webs:
            # 1. Perform the web search
            search_results = webs.text(keywords=q, region=region, 
                                     safesearch=safesearch,
                                     timelimit=timelimit, backend=backend, 
                                     max_results=max_results)

            # 2. Extract text from top search result URLs 
            extracted_text = ""
            for result in search_results:
                if 'href' in result:
                    link = result['href']
                    try:
                        response = requests.get(link, headers={"User-Agent": "Mozilla/5.0"})
                        response.raise_for_status()
                        visible_text = extract_text_from_webpage(response.text)
                        if len(visible_text) > max_chars:
                            visible_text = visible_text[:max_chars] + "..."
                        extracted_text += f"## Content from: {link}\n\n{visible_text}\n\n"
                    except requests.exceptions.RequestException as e:
                        print(f"Error fetching or processing {link}: {e}")
                else:
                   pass

        # 3. Construct the prompt for the LLM
        llm_prompt = f"Query by user: {q} , Answer the query asked by user in detail. Now, You are provided with Google Search Results, To increase your accuracy and providing real time data. SEarch Result: {extracted_text}"

        # 4. Get the LLM's response using LLM class (similar to /api/llm)
        messages = [{"role": "user", "content": llm_prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        llm = LLM(model=model)
        llm_response = llm.chat(messages=messages)

        # 5. Return the results
        return JSONResponse(content=jsonable_encoder({ "llm_response": llm_response }))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during advanced search: {e}")

        
@app.get("/api/website_summarizer")
async def website_summarizer(url: str):
    """Summarizes the content of a given URL using a chat model."""
    try:
        # Extract text from the given URL
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0"})
        response.raise_for_status()
        visible_text = extract_text_from_webpage(response.text)
        if len(visible_text) > 7500:  # Adjust max_chars based on your needs
            visible_text = visible_text[:7500] + "..."

        # Use chat model to summarize the extracted text
        with WEBS() as webs:
            summary_prompt = f"Summarize this in detail in Paragraph: {visible_text}"
            summary_result = webs.chat(keywords=summary_prompt, model="gpt-3.5")

        # Return the summary result
        return JSONResponse(content=jsonable_encoder({summary_result}))

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching or processing URL: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during summarization: {e}")

@app.get("/api/ask_website")
async def ask_website(url: str, question: str, model: str = "llama-3-70b"):
    """
    Asks a question about the content of a given website.
    """
    try:
        # Extract text from the given URL
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0"})
        response.raise_for_status()
        visible_text = extract_text_from_webpage(response.text)
        if len(visible_text) > 7500:  # Adjust max_chars based on your needs
            visible_text = visible_text[:7500] + "..."

        # Construct a prompt for the chat model
        prompt = f"Based on the following text, answer this question in Paragraph: [QUESTION] {question} [TEXT] {visible_text}"

        # Use chat model to get the answer
        with WEBS() as webs:
            answer_result = webs.chat(keywords=prompt, model=model)

        # Return the answer result
        return JSONResponse(content=jsonable_encoder({answer_result}))

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching or processing URL: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during question answering: {e}")
        
@app.get("/api/maps")
async def maps(
    q: str,
    place: Optional[str] = None,
    street: Optional[str] = None,
    city: Optional[str] = None,
    county: Optional[str] = None,
    state: Optional[str] = None,
    country: Optional[str] = None,
    postalcode: Optional[str] = None,
    latitude: Optional[str] = None,
    longitude: Optional[str] = None,
    radius: int = 0,
    max_results: int = 10
):
    """Perform a maps search."""
    try:
        with WEBS() as webs:
            results = webs.maps(keywords=q, place=place, street=street, city=city, county=county, state=state, country=country, postalcode=postalcode, latitude=latitude, longitude=longitude, radius=radius, max_results=max_results)
            return JSONResponse(content=jsonable_encoder(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during maps search: {e}")

@app.get("/api/translate")
async def translate(
    q: str,
    from_: Optional[str] = None,
    to: str = "en"
):
    """Translate text."""
    try:
        with WEBS() as webs:
            results = webs.translate(keywords=q, from_=from_, to=to)
            return JSONResponse(content=jsonable_encoder(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during translation: {e}")

@app.get("/api/youtube/transcript")
async def youtube_transcript(
    video_id: str,
    languages: str = "en",
    preserve_formatting: bool = False
):
    """Get the transcript of a YouTube video."""
    try:
        languages_list = languages.split(",")
        transcript = transcriber.get_transcript(video_id, languages=languages_list, preserve_formatting=preserve_formatting)
        return JSONResponse(content=jsonable_encoder(transcript))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting YouTube transcript: {e}")
        
import requests
@app.get("/weather/json/{location}")
def get_weather_json(location: str):
    url = f"https://wttr.in/{location}?format=j1"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Unable to fetch weather data. Status code: {response.status_code}"}

@app.get("/weather/ascii/{location}")
def get_ascii_weather(location: str):
    url = f"https://wttr.in/{location}"
    response = requests.get(url, headers={'User-Agent': 'curl'})
    if response.status_code == 200:
        return response.text
    else:
        return {"error": f"Unable to fetch weather data. Status code: {response.status_code}"}

# Run the API server if this script is executed
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8083)
