"""Prepare data for Google Sheets."""

import gspread
from gspread_dataframe import set_with_dataframe
# import gspread_formatting

import numpy as np
import pandas as pd
import time

path = "us-2024/"
sheetkey = "1zgP2VGpMXmOmC5QlxevDroIsQ4UILiBPmdNzhrpI-VE"

# connect to GSheet
gc = gspread.service_account()
sh = gc.open_by_key(sheetkey)

# Read in data
furl = "https://projects.fivethirtyeight.com/polls/data/senate_polls.csv"
df = pd.read_csv(furl)

# Filters
# >= 2024-07-21 only
df['start_date'] = pd.to_datetime(df['start_date'])
df['end_date'] = pd.to_datetime(df['end_date'])
df['middle_date'] = df['start_date'] + (df['end_date'] - df['start_date']) / 2
df0 = df[df["start_date"] >= "2024-01-01"]

# test
df0['race_id'].unique()
df0['state'].unique()
# states
states = list(df0.pivot_table(index='state', values='pct', aggfunc='count', fill_value=0).sort_values('pct', ascending=False).index)

# for each state
for state in states:
  df0_state = df0[(df0['state'] == state) | (pd.isna(df0['state']) & pd.isna(state))]

  # filter out polls/questions with just 2 candidates and 100% of votes
  pt1 = pd.pivot_table(df0_state, values="pct", index=["poll_id", "question_id"], aggfunc=["sum", "count"], fill_value=0)
  pt1.columns = ["sum", "count"]
  pt1 = pt1.reset_index()

  # Two main candidates
  ptname = pd.pivot_table(df0_state, values="pct", index=["answer"], aggfunc=["sum", "count"], fill_value=0)
  ptname.columns = ["sum", "count"]
  answers2 = ptname.sort_values("sum", ascending=False).index[:2]

  # reorder, DEM first
  if df0_state[df0_state["answer"] == answers2[0]].iloc[0]['party'] == 'DEM':
    pass
  else:
    answers2 = answers2[::-1]


  # select doubles poll_id and question_id, where there are 2 candidates // and 100% of votes
  # polls_include = pt1[~((pt1["count"] == 2) & (pt1["sum"] >= 99.9))][["poll_id", "question_id"]]
  # polls_include2 = pt1[((pt1["count"] == 2) & (pt1["sum"] >= 99.9))][["poll_id", "question_id"]]
  polls_include = pt1[~((pt1["count"] == 2))][["poll_id", "question_id"]]
  polls_include2 = pt1[((pt1["count"] == 2))][["poll_id", "question_id"]]
  df2 = polls_include.merge(df0, on=["poll_id", "question_id"], how="left")
  df2_2 = polls_include2.merge(df0, on=["poll_id", "question_id"], how="left")

  # select only Harris, Trump, Kennedy, West and Stein
  df3 = df2[df2["answer"].isin(answers2)]
  df3_2 = df2_2[df2_2["answer"].isin(answers2)]

  # pivot table, fill all indexes
  cols_index = ["start_date", "end_date", "middle_date", "pollster", "sponsors", "population", "sample_size", "numeric_grade", "url", "pollscore", "transparency_score"]
  pt2 = pd.pivot_table(df3, values="pct", index=cols_index, columns=["answer"], aggfunc="mean").reset_index()
  pt2_2 = pd.pivot_table(df3_2, values="pct", index=cols_index, columns=["answer"]).reset_index()

  pt2['number_of_candidates'] = 'Field'
  pt2_2['number_of_candidates'] = 'H2H'

  # merge into a single table
  # one of them may be empty
  if len(pt2) == 0:
    pt2all = pt2_2
  elif len(pt2_2) == 0:
    pt2all = pt2
  else:
    pt2all = pd.concat([pt2, pt2_2], ignore_index=True)
  
  if len(pt2all) == 0:
    print(f"{state} has no data.")
    continue

  # sort by middle_date, last_date
  pt2all = pt2all.sort_values(['middle_date', 'end_date'], ascending=[False, False])

  # average values for number_of_candidates (field, h2h) over the same polls (start_date, end_date, pollster, sponsors)
  cols_to_average = answers2
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
  election_date = pd.to_datetime('2024-11-05')
  pt2all_grouped['days to elections'] = (election_date - pd.to_datetime(pt2all_grouped['middle_date'])).dt.days
  pt2all_2['days to elections'] = (election_date - pd.to_datetime(pt2all_2['middle_date'])).dt.days

  # filter out in case of the same polls (start_date, end_date, pollster, sponsors)
  # keep in order: population: lv > rv > a
  pt2all_grouped['population_points'] = pt2all_grouped['population'].map({'lv': 3, 'rv': 2, 'a': 1})
  pt2all_grouped = pt2all_grouped.sort_values(by='population_points', ascending=False)
  pt2all_grouped = pt2all_grouped.drop_duplicates(subset=['start_date', 'end_date', 'pollster', 'sponsors'], keep='first')

  # calculate DEM - REP

  pt2all_grouped['DEM - REP'] = pt2all_grouped[answers2[0]] - pt2all_grouped[answers2[1]]

  # columns
  cols_short = ['pollster', 'sponsors', 'start_date', 'end_date', 'middle_date', 'days to elections', answers2[0], answers2[1], 'DEM - REP', 'population', 'number_of_candidates']
  cols_wide = ['pollster', 'sponsors', 'numeric_grade', 'pollscore', 'transparency_score', 'start_date', 'end_date', 'middle_date', 'sample_size', 'population', 'number_of_candidates', 'url', answers2[0], answers2[1]]
  us_short = pt2all_grouped.loc[:, cols_short]
  us_wide = pt2all_2.loc[:, cols_wide]
  # sort
  us_short = us_short.sort_values(['middle_date', 'end_date'], ascending=[False, False])
  us_wide = us_wide.sort_values(['middle_date', 'end_date'], ascending=[False, False])
  # save
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
    # add column last_updated with current data and time in the first row
    us_short['last_updated'] = np.nan
    us_short.iloc[0, us_short.columns.get_loc('last_updated')] = pd.Timestamp.now()
    # update worksheet
    set_with_dataframe(worksheet, us_short)
    print(f"{state} updated with {len(us_short)} rows.")
    time.sleep(5)