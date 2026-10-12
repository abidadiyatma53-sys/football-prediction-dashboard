import requests
import json

url = 'https://understat.com/getLeagueData/EPL/2023'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest',
    'Referer': 'https://understat.com/league/EPL/2023'
}

r = requests.get(url, headers=headers)
data = r.json()

teams = data.get('teams', {})
team_ids = list(teams.keys())
print("Number of teams:", len(team_ids))
sample_team = teams[team_ids[0]]
print("Sample team title:", sample_team.get('title'))
print("Sample team keys:", list(sample_team.keys()))
history = sample_team.get('history', [])
if history:
    print("History length:", len(history))
    print("Sample match in team history:")
    print(json.dumps(history[0], indent=2))
