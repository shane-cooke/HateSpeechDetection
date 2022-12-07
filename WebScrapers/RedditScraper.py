import praw
from praw.models import MoreComments
import pandas as pd

#Details for access to the Reddit API
reddit = praw.Reddit(
    client_id="cMz_QytAB0q_HIOoc3oIJg", 
    client_secret="dsphKErUd_dr_Cs7rbrnIoze0G4ZHg", 
    user_agent="Detecting Hate Speech by Shane Cooke"
)

cont_users = []
cont_comments = []
url = "https://www.reddit.com/r/Anarcho_Capitalism/comments/s6ovro/black_guy_harassing_asian_woman_in_nyc/"

submission = reddit.submission(url=url)

submission.comment_sort = "controversial"

for top_level_comment in submission.comments:
    if isinstance(top_level_comment, MoreComments):
        continue
    cont_users.append(top_level_comment.author)
    cont_comments.append(top_level_comment.body)
    
posts = pd.DataFrame()

posts["User"] = cont_users
posts["Comment"] = cont_comments

posts.to_csv('desktop/RedditComments.csv')
