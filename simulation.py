from __future__ import annotations

import binascii
import itertools
import pathlib
import random
import sys

import matplotlib.pyplot as plt
import numpy as np


DURATION_TIME_SLOT = 100  # in microseconds
NUM_TRIES_TRANSMISSION = 3
LENGTH_PAYLOAD_MAX = 84  # in bytes
LENGTH_ADDRESS = 6  # in bytes
LENGTH_CHECKSUM = 4  # in bytes


class Frame:
    id_iterator = itertools.count()

    def __init__(self, address_mac_source: bytes, address_mac_dest: bytes, payload: bytes):
        self.id = next(self.id_iterator)
        self.address_mac_source = address_mac_source
        self.address_mac_dest = address_mac_dest
        self.payload = payload
        self.checksum = self.calculate_checksum()

    def calculate_checksum(self):
        checksum = binascii.crc32(self.address_mac_source + self.address_mac_dest + self.payload)
        return checksum

    @property
    def valid(self):
        checksum = self.calculate_checksum()
        return self.checksum == checksum

    def __str__(self):
        match self.payload:
            case b"ACK":
                s = f"ACK Frame {self.id}"
            case b"SYNC":
                s = f"SYNC Frame {self.id}"
            case _:
                s = f"DATA Frame {self.id} ({len(self.payload)} bytes)"
        return s


def address_mac_bytes2str(address_mac: bytes):
    return "%02x:%02x:%02x:%02x:%02x:%02x" % tuple(address_mac)


class Node:
    id_iterator = itertools.count()

    def __init__(self, address_mac: bytes):
        self.id = next(self.id_iterator)
        self.address_mac = address_mac
        self.queue_frames = []
        self.time_backoff = 0
        self.transmissions_failed = 0

    @property
    def interval_backoff(self):
        return 2**self.transmissions_failed - 1

    @property
    def transmitting(self):
        return bool(self.queue_frames) and not self.time_backoff

    def sample_time_backoff(self):
        self.time_backoff = self.time_backoff or random.randint(0, self.interval_backoff)
        print(f"{self} - Backoff: {self.time_backoff}.")

    def decrement_time_backoff(self):
        if not self.time_backoff:
            return

        self.time_backoff -= 1
        print(f"{self} - Decrement backoff to {self.time_backoff}.")

    def reset_transmissions_failed(self):
        self.transmissions_failed = 0
        print(f"{self} - Reset failed transmission counter.")

    def increment_transmissions_failed(self):
        if self.transmissions_failed >= NUM_TRIES_TRANSMISSION:
            print(f"{self} - Max failed transmissions reached.")
            self.pop_queue()
            self.reset_transmissions_failed()
        else:
            self.transmissions_failed += 1
            print(f"{self} - Increment failed transmission counter to {self.transmissions_failed}.")

    def add_to_queue(self, frame: Frame):
        self.queue_frames.append(frame)
        print(f"{self} - Add to queue (pos. {len(self.queue_frames)}): {frame}")

    def pop_queue(self):
        if not self.queue_frames:
            return

        frame = self.queue_frames.pop(0)
        print(f"{self} - Remove from queue (pos. {len(self.queue_frames)}): {frame}.")
        return frame

    def generate_frame(self, address_mac_dest: bytes, payload: bytes):
        frame = Frame(self.address_mac, address_mac_dest, payload)
        return frame

    def send_frame(self, frame: Frame):
        print(f"{self} - Send: {frame}.")
        return frame

    def receive_frame(self, frame: Frame):
        valid = frame.valid
        print(f"{self} - Receive: {frame}. Valid: {valid}")
        return valid

    def generate_data(self, address_mac_dest: bytes):
        length_payload = random.randint(0, LENGTH_PAYLOAD_MAX)
        payload = random.randbytes(length_payload)
        frame = self.generate_frame(address_mac_dest, payload)
        self.add_to_queue(frame)

    def generate_ack(self, address_mac_dest: bytes):
        payload = b"ACK"
        ack = self.generate_frame(address_mac_dest, payload)
        return ack

    def generate_sync(self, address_mac_dest: bytes):
        payload = b"SYNC"
        sync = self.generate_frame(address_mac_dest, payload)
        return sync

    def send_data(self):
        frame = self.pop_queue()
        return self.send_frame(frame) if frame else None

    def send_ack(self, address_mac_dest: bytes):
        ack = self.generate_ack(address_mac_dest)
        return self.send_frame(ack)

    def send_sync(self, address_mac_dest: bytes):
        sync = self.generate_sync(address_mac_dest)
        return self.send_frame(sync)

    def __str__(self):
        return f"Node {self.id}"


class NetworkSimulation:
    def __init__(self, num_nodes: int, queue_probabilities: np.ndarray[float]):
        self.coordinator = Node(random.randbytes(LENGTH_ADDRESS))
        # num_nodes actually unneccessary but just following the assignment.
        self.nodes = np.array([Node(random.randbytes(LENGTH_ADDRESS)) for _ in range(num_nodes)])
        self.queue_probabilities = queue_probabilities
        self.bytes_transmitted = 0

    @property
    def node_is_transmitting(self):
        return np.asarray([node.transmitting for node in self.nodes], dtype=bool)

    @property
    def nodes_transmitting(self):
        return self.nodes[self.node_is_transmitting]

    @property
    def busy(self):
        # Thought it was needed, but its not since frames are always queued in discrete time steps.
        # Medium is only busy in between these.
        # Also there are not IFSs in the protocol defined.
        return np.any(self.node_is_transmitting)

    def run(self, duration: int):
        print("\n" * 5)

        for t in range(duration):
            print("#" * 10, f"Start time slot {t}", "#" * 10)

            self.enqueue()
            self.distribute()
            self.sync()
            self.data()
            self.backoff()

        throughput = self.throughput(self.bytes_transmitted, duration)
        print("Simulation terminated.")
        for node in self.nodes:
            print(f"{node} - Frames in queue: {len(node.queue_frames)}.")
        print(f"Throughput: {'%.2f' %throughput} Mbps.")
        return throughput

    def throughput(self, bytes, time):
        # In Mbps.
        return bytes / (time * len(self.nodes))

    def enqueue(self):
        nodes_enqueueing = self.nodes[np.random.random(self.queue_probabilities.shape[0]) < self.queue_probabilities]
        for node in nodes_enqueueing:
            # Assume sending to itself is possible.
            node_dest = np.random.choice(self.nodes)
            node.generate_data(node_dest.address_mac)

    def sync(self):
        syncs = []
        for receiver in self.nodes:
            sync = self.coordinator.send_sync(receiver.address_mac)
            syncs.append((receiver, sync))
        for receiver, sync in syncs:
            self.transmit(self.coordinator, receiver, sync)

    def transmit(self, sender: Node, receiver: Node, frame: Frame):
        success = receiver.receive_frame(frame)
        if success and frame.payload not in [b"ACK", b"SYNC"]:
            self.bytes_transmitted += len(frame.payload)

            ack = receiver.send_ack(frame.address_mac_source)
            self.transmit(receiver, sender, ack)

    def distribute(self):
        for node in self.nodes_transmitting:
            node.sample_time_backoff()

    def data(self):
        nodes_transmitting = self.nodes_transmitting
        if len(nodes_transmitting) > 1:
            print(f"Collission between nodes: {', '.join([str(node.id) for node in nodes_transmitting])}")
            for node in nodes_transmitting:
                node.increment_transmissions_failed()
        else:
            nodes_addresses = np.asarray([n.address_mac for n in self.nodes])
            for sender in nodes_transmitting:
                frame = sender.send_data()
                receiver = self.nodes[nodes_addresses == frame.address_mac_dest][0]
                self.transmit(sender, receiver, frame)

    def backoff(self):
        for node in self.nodes:
            node.decrement_time_backoff()


def save_figure():
    path_this = pathlib.Path(sys.argv[0])
    path_figure = path_this.parent / f"{path_this.stem}_plot.pdf"
    plt.savefig(path_figure)


def plot(subplot_func):
    """Plotter wrapper."""

    def plotter(*args, figsize=None, **kwargs):
        # Set figure size default to DIN A4.
        din_a4 = np.array([210, 297]) / 25.4

        fig = plt.figure(figsize=figsize or din_a4)

        subplot_func(*args, **kwargs)

        # Needs to be called after drawing. Therefore the wrapper.
        fig.tight_layout()

    return plotter


def subplots_thoughput(*values, **kwargs):
    "Organizer function for subplots."
    plt.gcf().add_subplot(2, 1, 1)
    subplot_thoughput(*values, **kwargs)


def subplot_thoughput(values, **kwargs):
    "Plotter."
    title = "Simulation results"
    label_x = r"queue probability $p$"
    label_y = r"average user throughput $d$ [Mbps]"

    ax = plt.gca()

    ax.set_title(title)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)

    for val in values:
        plt.plot(
            val[0],
            val[1],
            label=val[2],
        )

    ax.grid(True)
    ax.legend()


def run_series_of_experiments():
    num_simulations = 11
    results = []

    num_nodess = [2, 5, 10]
    for num_nodes in num_nodess:
        durations = [1000] * num_simulations
        probabilities = np.linspace(0, 1, num_simulations)
        queue_probabilitiess = np.outer(probabilities, np.ones(num_nodes))

        throughputs = []

        for duration, queue_probabilities in zip(durations, queue_probabilitiess):
            network = NetworkSimulation(num_nodes, queue_probabilities)
            throughput = network.run(duration)
            throughputs.append(throughput)

        throughputs = np.array(throughputs)
        results.append((probabilities, throughputs, rf"$N={{{num_nodes}}}$"))

    plot(subplots_thoughput)(results)
    save_figure()


def main():
    run_series_of_experiments()


if __name__ == "__main__":
    main()
