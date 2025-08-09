import json
import os

import figplot

from sglang.multi_model.scheduling.stats import ScheduleStats

with open("./benchmark/multi-model/stats.log", "r") as f:
    # Ensure the file is a JSON array
    json_list = json.load(f)
    stats = [ScheduleStats.from_dict(item) for item in json_list]
    figplot.plot_sim_stats(stats, f"mod_availability.pdf")
