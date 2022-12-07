import twint
import nest_asyncio
import pandas as pd
nest_asyncio.apply()

c = twint.Config()

c.Search = ['insane']
c.Limit = 200
c.Store_csv = True
c.Output = "desktop/twitterData.csv"

twint.run.Search(c)

df = pd.read_csv('desktop/twitterData.csv')
df_refined = df[['tweet', 'username']]
df_refined.to_csv('desktop/twitterData_refined.csv')