def scheduled_action(observation: dict, schedule: list[int]) -> int:
    """returns the planned trade quantity for the current time step"""

    time_step = observation["time_step"]
    remaining_lots = observation["remaining_lots"]

    if not 0 <= time_step < len(schedule):
        raise ValueError("Time step is outside the execution schedule.")

    lots_to_sell = schedule[time_step]

    if (
        type(lots_to_sell) is not int
        or lots_to_sell < 0
        or lots_to_sell > remaining_lots
    ):
        raise ValueError("Scheduled trade exceeds available inventory.")

    return lots_to_sell
