# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import csv
import datetime
from collections import defaultdict

import torch
import wandb
from termcolor import colored

wandb.require("core")


COMMON_TRAIN_FORMAT = [
    ("frame", "F", "int"),
    ("step", "S", "int"),
    ("episode", "E", "int"),
    ("episode_length", "L", "int"),
    ("episode_reward", "R", "float"),
    ("buffer_size", "BS", "int"),
    ("fps", "FPS", "float"),
    ("total_time", "T", "time"),
]

COMMON_EVAL_FORMAT = [
    ("frame", "F", "int"),
    ("step", "S", "int"),
    ("episode", "E", "int"),
    ("episode_length", "L", "int"),
    ("episode_reward", "R", "float"),
    ("total_time", "T", "time"),
]

COMMON_TEST_FORMAT = [
    ("frame", "F", "int"),
    ("step", "S", "int"),
    ("episode", "E", "int"),
    ("episode_length", "L", "int"),
    ("episode_reward", "R", "float"),
    ("total_time", "T", "time"),
]


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, csv_file_name, formating):
        self._csv_file_name = csv_file_name
        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        self._csv_file = None
        self._csv_writer = None

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            data[key] = meter.value()
        return data

    def _remove_old_entries(self, data):
        rows = []
        with self._csv_file_name.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row["episode"]) >= data["episode"]:
                    break
                rows.append(row)
        with self._csv_file_name.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(data.keys()), restval=0.0)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _dump_to_csv(self, data):
        if self._csv_writer is None:
            should_write_header = True
            if self._csv_file_name.exists():
                self._remove_old_entries(data)
                should_write_header = False

            self._csv_file = self._csv_file_name.open("a")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=sorted(data.keys()), restval=0.0
            )
            if should_write_header:
                self._csv_writer.writeheader()

        self._csv_writer.writerow(data)
        self._csv_file.flush()

    def _format(self, key, value, ty):
        if ty == "int":
            value = int(value)
            return f"{key}: {value}"
        elif ty == "float":
            return f"{key}: {value:.04f}"
        elif ty == "time":
            value = str(datetime.timedelta(seconds=int(value)))
            return f"{key}: {value}"
        else:
            raise f"invalid format type: {ty}"

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, "yellow" if prefix == "train" else "green")
        pieces = [f"| {prefix: <14}"]
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print(" | ".join(pieces))

    def dump(self, step, prefix):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data["frame"] = step
        self._dump_to_csv(data)
        self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    def __init__(self, log_dir, wandb_kwargs={}):
        self._log_dir = log_dir

        self._train_mg = MetersGroup(
            log_dir / "train.csv", formating=COMMON_TRAIN_FORMAT
        )
        self._eval_mg = MetersGroup(log_dir / "eval.csv", formating=COMMON_EVAL_FORMAT)
        self._test_orig_mg = MetersGroup(
            log_dir / "test_orig.csv", formating=COMMON_EVAL_FORMAT
        )

        if wandb_kwargs:
            self.wandb_run = wandb.init(**wandb_kwargs)

            with open(self._log_dir / "wandb_id.txt", "w") as f:
                f.write(self.wandb_run.id)

        else:
            self.wandb_run = None

    def get_mg(self, ty):
        if ty is None or ty == "eval":
            return self._eval_mg
        if ty is None or ty == "train":
            return self._train_mg
        if ty is None or ty == "test_orig":
            return self._test_orig_mg

    def _try_wandb_log(self, key, value, step, ty):
        if self.wandb_run is not None:
            self.wandb_run.log({f"{ty}/{key}": value, "env_step": step})

    def log(self, key, value, step, ty):
        if type(value) == torch.Tensor:
            value = value.item()
        self._try_wandb_log(key, value, step, ty)
        mg = self.get_mg(ty)
        mg.log(key, value)

    def log_metrics(self, metrics, step, ty):
        for key, value in metrics.items():
            self.log(key, value, step, ty)

    def dump(self, step, ty=None):
        mg = self.get_mg(ty)
        mg.dump(step, ty)

    def log_and_dump_ctx(self, step, ty):
        return LogAndDumpCtx(self, step, ty)

    def dump_ctx(self, step, ty):
        return DumpCtx(self, step, ty)


class LogAndDumpCtx:
    def __init__(self, logger, step, ty):
        self._logger = logger
        self._step = step
        self._ty = ty

    def __enter__(self):
        return self

    def __call__(self, key, value):
        self._logger.log(key, value, self._step, self._ty)

    def __exit__(self, *args):
        self._logger.dump(self._step, self._ty)


class DumpCtx:
    def __init__(self, logger, step, ty):
        self._logger = logger
        self._step = step
        self._ty = ty

    def __enter__(self):
        return self

    def __call__(self, key, value):
        if type(value) == torch.Tensor:
            value = value.item()
        mg = self._logger.get_mg(self._ty)
        mg.log(key, value)

    def __exit__(self, *args):
        self._logger.dump(self._step, self._ty)
