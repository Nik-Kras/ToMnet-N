"""
Function extract_features()

Reads a game from `data/complete_games`
And returns game in suitable for ToMnet-N format

It could be either Pydantic BaseModel or Pandas DataFrame
That keeps Map, Trajectory, Consumed Goal and Agent Type
"""