import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def main():
    # week1 = pd.read_csv("data/week1.csv")
    # plays_data = pd.read_csv("data/plays.csv")
    # players_data = pd.read_csv("data/players.csv")
    # players_data['abbreviatedName'] = (players_data['displayName'].str[0] + "." + players_data['displayName'].str.split().str[-1])
    # name_to_id_map = dict(zip(players_data['abbreviatedName'], players_data['nflId']))

    # plays = week1.groupby(["gameId", "playId"])
    
    # first_game_id, first_play_id = list(plays.groups)[0]
    # first_play = week1[(week1["gameId"] == first_game_id) & (week1["playId"] == first_play_id)]
    # game_data = plays_data[plays_data["gameId"] == first_game_id]
    # play_data = game_data[game_data["playId"] == first_play_id]

    # animate_play(first_play, play_data, name_to_id_map)
    process_single_play(1, 2021090900, 187)

def process_single_play(week, game_id, play_id):
    # plays = week.groupby(["gameId", "playId"])
    # first_game_id, first_play_id = list(plays.groups)[0]


    week = pd.read_csv(f"data/week{week}.csv")
    plays_data = pd.read_csv("data/plays.csv")
    players_data = pd.read_csv("data/players.csv")
    players_data['abbreviatedName'] = (players_data['displayName'].str[0] + "." + players_data['displayName'].str.split().str[-1])
    name_to_id_map = dict(zip(players_data['abbreviatedName'], players_data['nflId']))

    first_play = week[(week["gameId"] == game_id) & (week["playId"] == play_id)]
    game_data = plays_data[plays_data["gameId"] == game_id]
    play_data = game_data[game_data["playId"] == play_id]

    animate_play(first_play, play_data, name_to_id_map)
    
def extract_play_info(play_data, name_to_id_map):
    offense = play_data["possessionTeam"].values[0]
    defense = play_data["defensiveTeam"].values[0]

    description = play_data["playDescription"].values[0]
    interception_split_description = description.split("INTERCEPTED by ")
    interception_name = "none" if (len(interception_split_description) < 2) else interception_split_description[1].split(" ")[0].strip(".")
    interception_id = -1 if interception_name == "none" else name_to_id_map[interception_name]

    target_split_description = description.split(" to ")
    target_name = "none" if ((len(target_split_description) < 2) & (interception_name == "none")) else target_split_description[1].split(" ")[0].strip(".")
    target_id = -1 if target_name == "none" else name_to_id_map[target_name]

    return offense, defense, interception_id, target_id, target_name

def animate_play(tracking_data, play_data, name_to_id_map):
    frames = tracking_data['frameId']
    offense, defense, interception_id, target_id, target_name = extract_play_info(play_data, name_to_id_map)
    description = play_data["playDescription"].values[0]

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Define the animation function
    def animate(i):
        frame_data = tracking_data[tracking_data['frameId'] == i]
        football_data = frame_data[frame_data['team'] == 'football']
        target_data = frame_data[frame_data['nflId'] == target_id]
        off_data = frame_data[(frame_data['team'] == offense) & (frame_data['nflId'] != target_id)]
        def_data = frame_data[(frame_data['team'] != offense) & (frame_data['team'] != 'football')]
        ax.clear()
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 53.3)
        ax.scatter(football_data['x'], football_data['y'], c='brown', s=100)
        ax.scatter(target_data['x'], target_data['y'], c='yellow', s=100)
        ax.scatter(off_data['x'], off_data['y'], c='blue')
        ax.scatter(def_data['x'], def_data['y'], c='red')
        ax.set_title(f'{offense} vs {defense}, {description} (final target: {target_name}), Frame {i}')

    # Create the animation
    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=100, repeat=False)

    plt.show()

if __name__ == "__main__":
    main()