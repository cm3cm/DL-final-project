import pandas as pd
import numpy as np


def main():
    inputs, labels = process_data()
    print("inputs:\n", inputs.head())
    print("labels:\n", labels.head())


name_id_overrides = {
    46093: "Dj.Moore",
    53536: "Mi.Carter",
    53541: "A.St",
    46137: "Ju.Reid",
    53514: "Am.Rodgers",
    52458: "Ja.Johnson",
    41712: "Da.Williams",
    46097: "Te.Edmunds",
    46105: "D.Leonard",
    47964: "Jaq.Johnson",
    46276: "E.St",
}


def process_data():
    plays_data = pd.read_csv("data/plays.csv")
    players_data = pd.read_csv("data/players.csv")
    players_data["abbreviatedName"] = (
        players_data["displayName"].str[0]
        + "."
        + players_data["displayName"].str.split().str[-1]
    )
    name_to_id_map = dict(zip(players_data["nflId"], players_data["abbreviatedName"]))
    name_to_id_map.update(name_id_overrides)
    name_to_id_map = {v: k for k, v in name_to_id_map.items()}  # flip keys and values

    all_inputs = []
    all_labels = []

    for week in range(0, 8):
        inputs, labels = process_week(week + 1, plays_data, name_to_id_map)
        all_inputs.append(inputs)
        all_labels.append(labels)
    return pd.concat(all_inputs), pd.concat(all_labels)


def process_week(week, plays_data, name_to_id_map):
    week = pd.read_csv(f"data/week{week}.csv")
    release_snapshots = week[week["event"] == "pass_forward"].groupby(
        ["gameId", "playId"]
    )
    ids = release_snapshots.first().reset_index()[["gameId", "playId"]]
    labels = create_labels(ids, plays_data, name_to_id_map)

    inputs = []
    for name, group in release_snapshots:
        combined_id = name[0] * 100000 + name[1]
        if combined_id not in labels.index:
            continue

        group["sx"] = group["s"] * np.cos(group["dir"])
        group["sy"] = group["s"] * np.sin(group["dir"])
        group["ax"] = group["a"] * np.cos(group["dir"])
        group["ay"] = group["a"] * np.sin(group["dir"])
        relevant_fields = group.reset_index()[["x", "y", "sx", "sy", "ax", "ay", "o", "dir", "nflId"]]
        metadata = plays_data[
            (plays_data["gameId"] == name[0]) & (plays_data["playId"] == name[1])
        ][["down", "yardsToGo", "absoluteYardlineNumber", "pff_playAction"]].iloc[0]
        flattened = flatten_tracking_data(relevant_fields, metadata)

        if not flattened.empty:
            flattened.index = [combined_id]
            inputs.append(flattened)
    inputs = pd.concat(inputs)

    # remove errant pass_forward events or labels that don't have a corresponding pass_forward event
    # inputs = inputs[inputs.index.isin(labels.index)]
    labels = labels[labels.index.isin(inputs.index)]
    assert labels.index.equals(inputs.index)

    return inputs, labels


def create_labels(ids, plays_data, name_to_id_map):
    pass_plays = plays_data[
        (plays_data["passResult"] == "C")
        | (plays_data["passResult"] == "I")
        | (plays_data["passResult"] == "IN")
    ]
    plays = pass_plays.merge(ids, on=["gameId", "playId"])

    (
        plays["offense"],
        plays["defense"],
        plays["interception_id"],
        plays["target_id"],
        plays["target_name"],
    ) = zip(*plays.apply(lambda row: extract_play_info(row, name_to_id_map), axis=1))
    labels = plays[
        ["offense", "defense", "interception_id", "target_id", "target_name"]
    ]
    labels.index = plays["gameId"] * 100000 + plays["playId"]
    return labels


def extract_play_info(play_data, name_to_id_map):
    offense = play_data["possessionTeam"]
    defense = play_data["defensiveTeam"]

    description = play_data["playDescription"].split(" pass ")[1]
    # ignore anything before pass (e.g. fumbled snap)
    # print(description)
    interception_split_description = description.split("INTERCEPTED by ")
    interception_name = (
        "none"
        if (len(interception_split_description) < 2)
        else interception_split_description[1].split(" ")[0].strip(".")
    )
    interception_id = (
        -1 if interception_name == "none" else name_to_id_map[interception_name]
    )

    target_split_description = description.split(" to ")
    target_name = (
        "none"
        if ((len(target_split_description) < 2) | (interception_name != "none"))
        else target_split_description[1].split(" ")[0].strip(".")
    )
    target_id = -1 if target_name == "none" else name_to_id_map[target_name]

    return offense, defense, interception_id, target_id, target_name


def flatten_tracking_data(
    tracking_data: pd.DataFrame, metadata: pd.Series
) -> pd.DataFrame:
    # some plays have 2 "pass_forward" events, which are annoying to de-duplicate so we'll just skip them
    if tracking_data.shape[0] != 23:
        return pd.DataFrame()

    flattened = tracking_data.stack().swaplevel()
    flattened.index = flattened.index.map("{0[0]}_{0[1]}".format)
    flattened = flattened.to_frame().T

    metadata = metadata.to_frame().T.reset_index()
    flattened = pd.concat([flattened, metadata], axis=1).drop("index", axis=1)

    return flattened


if __name__ == "__main__":
    main()
