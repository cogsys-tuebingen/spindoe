import numpy as np
import csv
import pickle


def save(log_dicts, filename):
    with open(filename, "wb") as f:
        pickle.dump(log_dicts, f)


def read(filename):
    with open(filename, "rb") as f:
        log_dicts = pickle.load(f)
    return log_dicts


def read_pattern(file_path):
    # Read the points in a csv file
    points = []
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            points.append(np.array([float(row[0]), float(row[1]), float(row[2])]))
    return np.array(points)


def write_pattern(file_path, points):
    with open(file_path, "w+") as f:
        writer = csv.writer(f)
        for point in points:
            writer.writerow(point)


def get_time(path):
    parts = str(path).split("/")
    time = parts[-1][:-4]
    t = int(time) * 1e-9
    return t
