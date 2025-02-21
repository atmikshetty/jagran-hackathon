import json
import os
import time
from instagrapi import Client

SESSION_FILE = "settings.json"
DATA_FILE = "influencer_data_10.json"

def login():
    cl = Client()
    
    # try to delete the settings.json file, session is not getting cached for some reason 
    if os.path.exists(SESSION_FILE):
        print("Loading session...")
        cl.load_settings(SESSION_FILE)
    else:
        print("Logging in...")
        cl.login("", "")
        cl.dump_settings(SESSION_FILE)
    
    return cl

def fetch_influencer_data(username, post_count=50):

    cl = login()
    
    try:
        user_id = cl.user_id_from_username(username) # userID
        posts = cl.user_medias(user_id, post_count) # 50 posts
        influencer_posts = []

        for post in posts:
            post_details = {
                "id": post.pk,
                "caption": post.caption_text,
                "like_count": post.like_count,
                "comments_count": post.comment_count,
                "post_url": f"https://www.instagram.com/p/{post.code}/",  #useless
                "thumbnail_url": str(post.thumbnail_url) if post.thumbnail_url else None,  
                "video_url": str(post.video_url) if post.video_url else None,
                "comments": []
            }
            
            # sleep for 1 if starting a new session
            time.sleep(2)
            comments = cl.media_comments(post.pk, amount=20)
            for comment in comments:
                post_details["comments"].append({
                    "user": comment.user.username,
                    "text": comment.text
                })
            
            influencer_posts.append(post_details)
        
        # load files
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r") as f:
                all_data = json.load(f)
        else:
            all_data = {}
        
        all_data[username] = influencer_posts
        
        # dump data
        with open(DATA_FILE, "w") as f:
            json.dump(all_data, f, indent=4)

        print(f"Data collection for {username} done.")
    
    except Exception as e:
        print(f"Error fetching data for {username}: {e}")
    
    # do not remove helps bypass insta banning the account
    finally:
        cl.logout() 

if __name__ == "__main__":
    username = "thepaayaljain"
    fetch_influencer_data(username)
