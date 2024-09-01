"""Prepare data for Google Sheets."""

import gspread
from gspread_dataframe import set_with_dataframe
# from gspread_formatting import *

import numpy as np
import pandas as pd
import time

path = "/us-2024/"
sheetkey = "1osHcYaKPesurmX5kjMnzdMD2T2Ih75gaAWPqCP9T4YA"

# connect to GSheet
gc = gspread.service_account()
sh = gc.open_by_key(sheetkey)

# Read in data
furl = "https://projects.fivethirtyeight.com/polls/data/president_polls_historical.csv"
df = pd.read_csv(furl)

# Filters
# >= 
df['start_date'] = pd.to_datetime(df['start_date'])
df['end_date'] = pd.to_datetime(df['end_date'])
df['middle_date'] = df['start_date'] + (df['end_date'] - df['start_date']) / 2
df0 = df[df["start_date"] >= "2000-01-01"]

# national polls only race_id == 8914
df0['race_id'].unique()
df0['state'].unique()
# df0 = df0[df0["race_id"] == 8914]

# fill nan for state by 'US' and by 'None' for sponsors
df0['state'] = df0['state'].fillna('US')
df0 = df0.fillna('None')

for state in df0['state'].unique():
  df0_state = df0[(df0['state'] == state) | (pd.isna(df0['state']) & pd.isna(state))]

  # filter out polls/questions with just 2 candidates and 100% of votes
  pt1 = pd.pivot_table(df0_state, values="pct", index=["poll_id", "question_id"], aggfunc=["sum", "count"], fill_value=0)
  pt1.columns = ["sum", "count"]
  pt1 = pt1.reset_index()
  # select doubles poll_id and question_id, where there are 2 candidates // and 100% of votes
  # polls_include = pt1[~((pt1["count"] == 2) & (pt1["sum"] >= 99.9))][["poll_id", "question_id"]]
  # polls_include2 = pt1[((pt1["count"] == 2) & (pt1["sum"] >= 99.9))][["poll_id", "question_id"]]
  polls_include = pt1[~((pt1["count"] == 2))][["poll_id", "question_id"]]
  polls_include2 = pt1[((pt1["count"] == 2))][["poll_id", "question_id"]]
  df2 = polls_include.merge(df0, on=["poll_id", "question_id"], how="left")
  df2_2 = polls_include2.merge(df0, on=["poll_id", "question_id"], how="left")

  # select only Harris, Trump, Kennedy, West and Stein
  df3 = df2[df2["answer"].isin(['Biden', 'Trump', 'Jorgensen', 'Sanders', 'Warren'])]
  df3_2 = df2_2[df2_2["answer"].isin(["Biden", "Trump"])]

  # pivot table, fill all indexes
  cols_index = ["start_date", "end_date", "middle_date", "pollster", "sponsors", "population", "sample_size", "numeric_grade", "url", "pollscore", "transparency_score"]
  pt2 = pd.pivot_table(df3, values="pct", index=cols_index, columns=["answer"], aggfunc="mean")
  pt2_2 = pd.pivot_table(df3_2, values="pct", index=cols_index, columns=["answer"], aggfunc="mean")

  pt2 = pt2.reset_index()
  pt2_2 = pt2_2.reset_index()

  pt2['number_of_candidates'] = 'Field'
  pt2_2['number_of_candidates'] = 'H2H'

  # merge into a single table
  pt2all = pd.concat([pt2, pt2_2], ignore_index=True)

  # sort by middle_date, last_date
  pt2all = pt2all.sort_values(['middle_date', 'end_date'], ascending=[False, False])

  # average values for number_of_candidates (field, h2h) over the same polls (start_date, end_date, pollster, sponsors)
  cols_to_average = ['Biden', 'Trump', 'Jorgensen', 'Sanders', 'Warren']
  # add columns to pt2all if needed
  for col in cols_to_average:
    if col not in pt2all.columns:
      pt2all[col] = np.nan
  # keep _2 for wide format
  pt2all_2 = pt2all.copy()

  cols_for_grouping = cols_index + ['days to elections']
  if len(pt2all) > 0:
    pt2all_grouped = pt2all.groupby(cols_index)[cols_to_average].mean().reset_index()
  else:
    pt2all_grouped = pd.DataFrame(columns=cols_index + cols_to_average)

  # Determine if there are multiple rows for each group and set 'number_of_candidates' accordingly
  def set_number_of_candidates(group):
    if len(group) > 1:
      return "H2H/Field"
    else:
      return group['number_of_candidates'].iloc[0]

  # Apply the function to set 'number_of_candidates'
  pt2all_grouped['number_of_candidates'] = pt2all.groupby(cols_index).apply(set_number_of_candidates).values

  # add columns
  election_date = pd.to_datetime('2020-11-03')
  pt2all_grouped['days to elections'] = (election_date - pd.to_datetime(pt2all_grouped['middle_date'])).dt.days
  pt2all_2['days to elections'] = (election_date - pd.to_datetime(pt2all_2['middle_date'])).dt.days

  # filter out in case of the same polls (start_date, end_date, pollster, sponsors)
  # keep in order: population: lv > rv > a
  pt2all_grouped['population_points'] = pt2all_grouped['population'].map({'lv': 3, 'rv': 2, 'a': 1})
  pt2all_grouped = pt2all_grouped.sort_values(by='population_points', ascending=False)
  pt2all_grouped = pt2all_grouped.drop_duplicates(subset=['start_date', 'end_date', 'pollster', 'sponsors'], keep='first')

  # calculate Harris - Trump
  pt2all_grouped['Biden - Trump'] = pt2all_grouped['Biden'] - pt2all_grouped['Trump']

  # columns
  cols_short = ['pollster', 'sponsors', 'start_date', 'end_date', 'middle_date', 'days to elections', 'Biden', 'Trump', 'Biden - Trump', 'population', 'number_of_candidates']
  cols_wide = ['pollster', 'sponsors', 'numeric_grade', 'pollscore', 'transparency_score', 'start_date', 'end_date', 'middle_date', 'sample_size', 'population', 'number_of_candidates', 'url', 'Biden', 'Trump', 'Jorgensen', 'Sanders', 'Warren']
  us_short = pt2all_grouped.loc[:, cols_short]
  us_wide = pt2all_2.loc[:, cols_wide]
  # sort
  us_short = us_short.sort_values(['middle_date', 'end_date'], ascending=[False, False])
  us_wide = us_wide.sort_values(['middle_date', 'end_date'], ascending=[False, False])
  # save
  if not(state):
    state = 'US'
    
  # us_short.to_csv(path + f"polls_{state}_short.csv", index=False)
  # us_wide.to_csv(path + f"polls_{state}_wide.csv", index=False)

  # save short to GSheet
  # open worksheet, if not exists, create it
  if len(us_short) > 0:
    try:
      worksheet = sh.worksheet(state)
    except:
      worksheet = sh.add_worksheet(title=state, rows="100", cols="20")
      # freeze first row
      worksheet.freeze(rows=1)

    # clear worksheet
    worksheet.clear()
    # update worksheet
    set_with_dataframe(worksheet, us_short)
    print(f"{state} updated with {len(us_short)} rows.")
    time.sleep(5)
  