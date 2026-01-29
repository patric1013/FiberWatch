"""
Event processing utilities for clustering and filtering detected events.

This module provides functions for post-processing detected events,
including clustering nearby events and selecting the best representative
event from each cluster.
"""

from typing import List

import numpy as np

from ..core.detector import DetectedEvent


def cluster_events(
    events: List[DetectedEvent],
    distance_threshold_m: float = 5.0,
) -> List[DetectedEvent]:
    """
    Cluster nearby events that are within distance_threshold_m of each other.

    Args:
        events: List of detected events
        distance_threshold_m: Maximum distance between events in the same cluster (meters)

    Returns:
        List of clustered events with one representative per cluster
    """
    if not events:
        return []

    # Sort events by position
    sorted_events = sorted(events, key=lambda e: e.z_km)

    # Check for breaks and limit processing to before first break
    first_break_index = None
    for i, event in enumerate(sorted_events):
        if event.kind == "break":
            first_break_index = i
            break

    # If break detected, only process events up to and including the break
    if first_break_index is not None:
        sorted_events = sorted_events[: first_break_index + 1]
        print(
            f"Break detected at {sorted_events[first_break_index].z_km:.3f}km, ignoring subsequent events"
        )

    clusters = []
    current_cluster = [sorted_events[0]]

    for event in sorted_events[1:]:
        cluster_center = np.mean([e.z_km for e in current_cluster])
        effective_threshold = distance_threshold_m

        # Use larger threshold for connector events
        if (
            any("connector" in e.kind for e in current_cluster)
            and "connector" in event.kind
        ):
            effective_threshold = max(distance_threshold_m, 25.0)

        if abs(event.z_km - cluster_center) * 1000 <= effective_threshold:
            current_cluster.append(event)
        else:
            clusters.append(current_cluster)
            current_cluster = [event]

    clusters.append(current_cluster)

    # Select best event from each cluster
    clustered_events = []
    for cluster in clusters:
        if len(cluster) == 1:
            clustered_events.append(cluster[0])
        else:
            best_event = select_best_event_in_cluster(cluster)
            clustered_events.append(best_event)

    return _handle_end_of_fiber_events(clustered_events)


def select_best_event_in_cluster(cluster: List[DetectedEvent]) -> DetectedEvent:
    """
    Select the most significant event from a cluster of events.

    Priority: break > dirty_connector > clean_connector > reflection > splice > bend
    Within each type, select by magnitude.

    Args:
        cluster: List of events in the same cluster

    Returns:
        The most significant event from the cluster
    """
    # Separate by event type
    breaks = [e for e in cluster if e.kind == "break"]
    dirty_connectors = [e for e in cluster if e.kind == "dirty_connector"]
    clean_connectors = [e for e in cluster if e.kind == "clean_connector"]
    reflections = [e for e in cluster if e.kind == "reflection"]
    splices = [e for e in cluster if e.kind == "splice"]
    bends = [e for e in cluster if e.kind == "bend"]

    # Priority order: break > dirty_connector > clean_connector > reflection > splice > bend
    if breaks:
        return max(breaks, key=lambda e: e.reflect_db + abs(e.magnitude_db))
    if dirty_connectors:
        return max(dirty_connectors, key=lambda e: e.reflect_db + abs(e.magnitude_db))
    if clean_connectors:
        return max(clean_connectors, key=lambda e: e.reflect_db + abs(e.magnitude_db))
    if reflections:
        return max(reflections, key=lambda e: e.reflect_db + abs(e.magnitude_db))
    if splices:
        return max(splices, key=lambda e: abs(e.magnitude_db))
    return max(bends, key=lambda e: abs(e.magnitude_db))


def _handle_end_of_fiber_events(
    clustered_events: List[DetectedEvent],
) -> List[DetectedEvent]:
    """
    Handle special case of multiple events near fiber end.

    Args:
        clustered_events: List of clustered events

    Returns:
        Filtered list with end-of-fiber events handled
    """
    if not clustered_events:
        return clustered_events

    max_distance = max(e.z_km for e in clustered_events)
    end_threshold_km = 0.1  # Events within 100m of end

    end_events = [
        e for e in clustered_events if (max_distance - e.z_km) <= end_threshold_km
    ]
    other_events = [
        e for e in clustered_events if (max_distance - e.z_km) > end_threshold_km
    ]

    if len(end_events) > 1:
        breaks = [e for e in end_events if e.kind == "break"]
        if breaks:
            best_break = max(breaks, key=lambda e: e.reflect_db + abs(e.magnitude_db))
            filtered_end = [best_break] + [
                e for e in end_events if "connector" not in e.kind and e.kind != "break"
            ]
            if len(filtered_end) > 1:
                filtered_end = [best_break]
            return other_events + filtered_end

    return clustered_events


def get_event_statistics(events: List[DetectedEvent]) -> dict:
    """
    Calculate statistics for detected events.

    Args:
        events: List of detected events

    Returns:
        Dictionary with event statistics
    """
    if not events:
        return {
            "total_events": 0,
            "event_types": {},
            "total_loss_db": 0.0,
            "max_loss_db": 0.0,
            "fiber_length_km": 0.0,
        }

    event_counts = {}
    for event in events:
        event_counts[event.kind] = event_counts.get(event.kind, 0) + 1

    total_loss = sum(abs(event.magnitude_db) for event in events)
    max_loss = max(abs(event.magnitude_db) for event in events)
    fiber_length = max(event.z_km for event in events)

    return {
        "total_events": len(events),
        "event_types": event_counts,
        "total_loss_db": total_loss,
        "max_loss_db": max_loss,
        "fiber_length_km": fiber_length,
    }
