from fastapi import FastAPI, HTTPException, Query, Response
import requests

app = FastAPI()

@app.get("/proxy-image")
def proxy_image(url: str = Query(..., description="The image URL to proxy")):
    if not url.startswith("https://"):
        raise HTTPException(status_code=400, detail="Invalid URL")
    
    # Use headers that mimic a browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/105.0.0.0 Safari/537.36",
        "Referer": "https://www.instagram.com/",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
    }
    
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            content_type = r.headers.get("Content-Type", "image/jpeg")
            return Response(content=r.content, media_type=content_type)
        else:
            raise HTTPException(status_code=r.status_code, detail="Failed to fetch image")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
