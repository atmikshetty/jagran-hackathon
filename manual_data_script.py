import json
import os
import time
from instagrapi import Client

SESSION_FILE = "settings.json"
DATA_FILE = "influencer_data_10.json"

def login():
    """Logs into Instagram and returns a Client instance."""
    cl = Client()
    
    if os.path.exists(SESSION_FILE):
        print("Loading session...")
        cl.load_settings(SESSION_FILE)
    else:
        print("Logging in...")
        cl.login("chadonly1234", "insta@1234")
        cl.dump_settings(SESSION_FILE)
    
    return cl

def fetch_influencer_data(username, post_count=50):
    """Fetches influencer data for a given username and saves it to a JSON file."""
    cl = login()
    
    try:
        user_id = cl.user_id_from_username(username)
        posts = cl.user_medias(user_id, post_count)
        influencer_posts = []

        for post in posts:
            post_details = {
                "id": post.pk,
                "caption": post.caption_text,
                "like_count": post.like_count,
                "comments_count": post.comment_count,
                "post_url": f"https://www.instagram.com/p/{post.code}/",
                "thumbnail_url": str(post.thumbnail_url) if post.thumbnail_url else None,  
                "video_url": str(post.video_url) if post.video_url else None,
                "comments": []
            }
            
            # Fetch comments with a short delay to avoid detection
            time.sleep(2)
            comments = cl.media_comments(post.pk, amount=20)
            for comment in comments:
                post_details["comments"].append({
                    "user": comment.user.username,
                    "text": comment.text
                })
            
            influencer_posts.append(post_details)
        
        # Load existing data if available
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r") as f:
                all_data = json.load(f)
        else:
            all_data = {}
        
        all_data[username] = influencer_posts
        
        # Save the collected data
        with open(DATA_FILE, "w") as f:
            json.dump(all_data, f, indent=4)

        print(f"✅ Data collection for {username} done and saved.")
    
    except Exception as e:
        print(f"❌ Error fetching data for {username}: {e}")
    
    finally:
        cl.logout()

if __name__ == "__main__":
    username = "thepaayaljain"
    fetch_influencer_data(username)
