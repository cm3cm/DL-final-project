import pandas as pd
import numpy as np


def main():
    inputs, labels = process_data()
    print("inputs:\n", inputs.head())
    print("labels:\n", labels.head())
    save_data(inputs, labels)


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


def get_data():
    if (
        pd.read_csv("data/inputs.csv").shape[0] > 0
        and pd.read_csv("data/labels.csv").shape[0] > 0
    ):
        return load_data()
    else:
        inputs, labels = process_data()
        save_data(inputs, labels)
        return inputs, labels


def save_data(inputs, labels):
    inputs.to_csv("data/inputs.csv")
    labels.to_csv("data/labels.csv")


def load_data():
    inputs = pd.read_csv("data/inputs.csv")
    labels = pd.read_csv("data/labels.csv")
    return inputs, labels


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

    id_to_pos_map = dict(zip(players_data["nflId"], players_data["officialPosition"]))

    all_inputs = []
    all_labels = []

    for week in range(0, 8):
        inputs, labels = process_week(
            week + 1, plays_data, name_to_id_map, id_to_pos_map
        )
        all_inputs.append(inputs)
        all_labels.append(labels)
    inputs, labels = pd.concat(all_inputs), pd.concat(all_labels)
    labels = labels.drop(columns=["offense", "defense", "target_name"])
    return inputs, labels


def process_week(week, plays_data, name_to_id_map, id_to_pos_map):
    week = pd.read_csv(f"data/week{week}.csv")
    release_snapshots = week[week["event"] == "pass_forward"].groupby(
        ["gameId", "playId"]
    )
    ids = release_snapshots.first().reset_index()[["gameId", "playId"]]
    labels = create_labels(ids, plays_data, name_to_id_map)

    inputs = []
    receiver_labels = []
    for name, group in release_snapshots:
        combined_id = name[0] * 100000 + name[1]
        if combined_id not in labels.index:
            continue

        group["sx"] = group["s"] * np.cos(group["dir"])
        group["sy"] = group["s"] * np.sin(group["dir"])
        group["ax"] = group["a"] * np.cos(group["dir"])
        group["ay"] = group["a"] * np.sin(group["dir"])
        group = normalize_direction(group)
        relevant_fields = group.reset_index()[
            ["x", "y", "sx", "sy", "ax", "ay", "o", "nflId", "team"]
        ]

        # mirrored = relevant_fields.copy()
        # mirrored["x"] = 120 - mirrored["x"]
        # mirrored["sx"] = -mirrored["sx"]
        # mirrored["ax"] = -mirrored["ax"]
        # mirrored["o"] = (mirrored["o"] + np.pi) % (2 * np.pi)

        metadata = plays_data[
            (plays_data["gameId"] == name[0]) & (plays_data["playId"] == name[1])
        ][["down", "yardsToGo", "absoluteYardlineNumber", "pff_playAction"]].iloc[0]
        flattened, receiver_ids = flatten_tracking_data(
            relevant_fields, metadata, id_to_pos_map, combined_id
        )
        # flattened_mirrored, receiver_ids_mirrored = flatten_tracking_data(
        #     mirrored, metadata, id_to_pos_map, combined_id
        # )

        if not flattened.empty:
            flattened.index = [combined_id]
            # flattened_mirrored.index = [combined_id]
            receiver_ids.index = [combined_id]
            inputs.append(flattened)
            # inputs.append(flattened_mirrored)
            receiver_labels.append(receiver_ids)
            # receiver_labels.append(receiver_ids_mirrored)

    inputs = pd.concat(inputs)
    receiver_labels = pd.concat(receiver_labels)

    # remove errant pass_forward events or labels that don't have a corresponding pass_forward event
    # inputs = inputs[inputs.index.isin(labels.index)]
    labels = labels[labels.index.isin(inputs.index)]
    assert labels.index.equals(inputs.index)

    # print(labels.head())
    # print(receiver_labels.head())
    labels = labels.merge(receiver_labels, left_index=True, right_index=True)

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
    labels = labels[labels["target_id"] != -1]
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
    tracking_data: pd.DataFrame,
    metadata: pd.Series,
    id_to_pos_map: dict,
    combined_id: int,
) -> pd.DataFrame:
    # some plays have 2 "pass_forward" events, which are annoying to de-duplicate so we'll just skip them
    if tracking_data.shape[0] != 23:
        return pd.DataFrame(), pd.DataFrame()

    qb_data = (
        tracking_data[tracking_data["nflId"].map(id_to_pos_map) == "QB"]
        .reset_index()
        .drop("index", axis=1)
    )
    # qb_data = qb_data.drop(columns=["o"])
    # print(qb_data.head())

    receiver_data = (
        tracking_data[
            tracking_data["nflId"].map(id_to_pos_map).isin(["WR", "TE", "RB"])
        ]
        .reset_index()
        .drop("index", axis=1)
    )
    receiver_data["rel_x"] = receiver_data["x"] - qb_data["x"].values[0]
    receiver_data["rel_y"] = receiver_data["y"] - qb_data["y"].values[0]
    receiver_data["position"] = (
        receiver_data["nflId"].map(id_to_pos_map).map({"WR": 0, "TE": 1, "RB": 2})
    )
    receiver_data = receiver_data.sort_values("y").reset_index(drop=True)
    # print(receiver_data.head())

    remaining_data = (
        tracking_data[
            ~tracking_data["nflId"].map(id_to_pos_map).isin(["QB", "WR", "TE", "RB"])
        ]
        .reset_index()
        .drop("index", axis=1)
    )
    remaining_offense = remaining_data[
        remaining_data["team"] == qb_data["team"].iloc[0]
    ]
    remaining_defense = remaining_data[
        remaining_data["team"] != qb_data["team"].iloc[0]
    ]
    remaining_defense = (
        remaining_defense[remaining_defense["team"] != "football"]
        .reset_index()
        .drop("index", axis=1)
    )
    # assert(remaining_defense.shape[0] == 11)

    def get_closest_defenders(receiver_data, remaining_defense):
        distances = np.sqrt(
            (receiver_data["x"].values[:, np.newaxis] - remaining_defense["x"].values)
            ** 2
            + (receiver_data["y"].values[:, np.newaxis] - remaining_defense["y"].values)
            ** 2
        )
        closest_defenders_indices = np.argsort(distances, axis=1)[:, :2]
        closest_defenders = remaining_defense.iloc[
            closest_defenders_indices.flatten()
        ].reset_index(drop=True)

        # Calculate relative x and y positions
        closest_defenders["rel_x"] = closest_defenders["x"] - receiver_data[
            "x"
        ].values.repeat(2)
        closest_defenders["rel_y"] = closest_defenders["y"] - receiver_data[
            "y"
        ].values.repeat(2)

        return closest_defenders.groupby(np.repeat(receiver_data.index, 2)).apply(
            lambda x: x.reset_index(drop=True)
        )

    closest_defenders = get_closest_defenders(receiver_data, remaining_defense)
    # print(closest_defenders)
    # assert False

    if (
        qb_data.shape[0] != 1
        or receiver_data.shape[0] != 5
        or remaining_defense.shape[0] != 11
    ):
        return pd.DataFrame(), pd.DataFrame()
    # assert(qb_data.shape[0] == 1), "Expected 1 QB, got " + str(qb_data.shape[0]) + " for play " + str(combined_id)
    # assert receiver_data.shape[0] == 5, "Expected 5 receivers, got " + str(receiver_data.shape[0]) + " for play " + str(combined_id)

    # flattened = tracking_data.stack().swaplevel()
    # flattened.index = flattened.index.map("{0[0]}_{0[1]}".format)
    # flattened = flattened.to_frame().T

    receiver_ids = receiver_data[["nflId"]].stack().swaplevel()
    receiver_ids.index = receiver_ids.index.map("rec{0[1]}".format)
    receiver_ids = receiver_ids.to_frame().T

    qb_data.drop(columns=["team", "nflId"], inplace=True)
    receiver_data.drop(columns=["team", "nflId"], inplace=True)
    remaining_defense.drop(columns=["team", "nflId"], inplace=True)
    closest_defenders.drop(columns=["team", "nflId"], inplace=True)

    flattened_qb = qb_data.stack().swaplevel()
    flattened_qb.index = flattened_qb.index.map("{0[0]}_qb{0[1]}".format)
    flattened_qb = flattened_qb.to_frame().T

    flattened_receivers = receiver_data.stack().swaplevel()
    flattened_receivers.index = flattened_receivers.index.map("{0[0]}_rec{0[1]}".format)
    flattened_receivers = flattened_receivers.to_frame().T

    flattened_closest_defenders = closest_defenders.stack().swaplevel()
    flattened_closest_defenders.index = flattened_closest_defenders.index.map(
        "{0[1]}_rec{0[0]}_def{0[2]}".format
    )
    flattened_closest_defenders = flattened_closest_defenders.to_frame().T

    # print(flattened_receivers)
    # print(flattened_closest_defenders.columns.to_series().to_string())
    # assert False
    # interleaved_data = pd.concat(
    #     [flattened_receivers, flattened_closest_defenders], axis=1
    # )

    # flattened_remaining = remaining_data.stack().swaplevel()
    flattened_remaining = remaining_defense.stack().swaplevel()
    # flattened_remaining.index = flattened_remaining.index.map("{0[0]}_o{0[1]}".format)
    flattened_remaining.index = flattened_remaining.index.map("{0[0]}_def{0[1]}".format)
    flattened_remaining = flattened_remaining.to_frame().T

    # print(flattened_qb.head())
    # print(flattened_receivers.head())
    # print(flattened_remaining.head())

    metadata = metadata.to_frame().T.reset_index()
    # flattened = pd.concat([flattened, metadata], axis=1).drop("index", axis=1)
    flattened = pd.concat(
        [flattened_qb, flattened_receivers, flattened_closest_defenders, metadata],
        axis=1,
    ).drop("index", axis=1)

    return flattened, receiver_ids


def normalize_direction(tracking_data: pd.DataFrame):
    if tracking_data["playDirection"].iloc[0] == "right":
        return tracking_data

    tracking_data["x"] = 120 - tracking_data["x"]
    tracking_data["y"] = 53.3 - tracking_data["y"]
    tracking_data["sx"] = -tracking_data["sx"]
    tracking_data["sy"] = -tracking_data["sy"]
    tracking_data["ax"] = -tracking_data["ax"]
    tracking_data["ay"] = -tracking_data["ay"]
    # tracking_data["dir"] = (tracking_data["dir"] + np.pi) % (2 * np.pi)
    tracking_data["o"] = (tracking_data["o"] + np.pi) % (2 * np.pi)
    return tracking_data


if __name__ == "__main__":
    main()
