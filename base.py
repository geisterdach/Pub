#!/usr/bin/env python3
import requests
import json
import webbrowser
import os
from datetime import datetime
from typing import Callable

BASE_URL = "https://berghain.challenges.listenlabs.ai"
PLAYER_ID = "9a59582b-8857-4414-9b9b-e2925d50e296"

os.makedirs("logs", exist_ok=True)

LOG_FILE = None
FIRST_LOG_ENTRY = True


def init_logging(algo_name, scenario):
    global LOG_FILE, FIRST_LOG_ENTRY
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/{algo_name}_scenario_{scenario}_{timestamp}.json"
    LOG_FILE = open(log_filename, 'w', buffering=1)
    LOG_FILE.write('[\n')
    FIRST_LOG_ENTRY = True
    return log_filename


def log(message, data=None):
    global FIRST_LOG_ENTRY
    if LOG_FILE is None:
        return

    if not FIRST_LOG_ENTRY:
        LOG_FILE.write(',\n')
    else:
        FIRST_LOG_ENTRY = False

    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "message": message
    }

    if data is not None:
        log_entry["data"] = data

    LOG_FILE.write('  ' + json.dumps(log_entry, default=str))
    LOG_FILE.flush()


def close_logging():
    global LOG_FILE
    if LOG_FILE:
        LOG_FILE.write('\n]')
        LOG_FILE.close()
        LOG_FILE = None


def log_data(data, prefix=""):
    log(prefix.strip(), data)


def create_new_game(scenario=1):
    url = f"{BASE_URL}/new-game?scenario={scenario}&playerId={PLAYER_ID}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def decide_and_next(game_id, person_index, accept):
    # If accept is None, we don't include it in the URL
    if accept is None:
        url = f"{BASE_URL}/decide-and-next?gameId={game_id}&personIndex={person_index}"
    else:
        url = f"{BASE_URL}/decide-and-next?gameId={game_id}&personIndex={person_index}&accept={str(accept).lower()}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def run_algorithm(algo_function: Callable, scenario=1, max_iterations=20000, algo_name="algo"):
    log_filename = init_logging(algo_name=algo_name, scenario=scenario)
    print(f"Starting Berghain Bouncer Challenge")
    print(f"Logs will be written to: {log_filename}")
    log("Starting Berghain Bouncer Challenge")

    try:
        game_data = create_new_game(scenario=scenario)
        log_data(game_data, "NEW GAME")

        game_id = game_data["gameId"]
        print(f"Game ID: {game_id}")
        log(f"Game ID: {game_id}")

        game_url = f"{BASE_URL}/game/{game_id}"
        print(f"Opening game in browser: {game_url}")
        log(f"Opening game in browser: {game_url}")
        webbrowser.open(game_url)

        constraints = game_data["constraints"]
        attribute_statistics = game_data["attributeStatistics"]
        correlations = attribute_statistics.get("correlations", {})

        accepted_count = {constraint["attribute"]: 0 for constraint in constraints}

        status = "running"
        person_index = 0
        admitted_count = 0
        rejected_count = 0
        decision_data = None
        current_person = None

        while person_index <= max_iterations and status == "running":
            try:
                if current_person == None:
                    accept = None  # First person, no decision yet, we still need to accept them, otherwirse error
                else:
                    accept = algo_function(
                        constraints=constraints,
                        attribute_statistics=attribute_statistics,
                        correlations=correlations,
                        admitted_count=admitted_count,
                        rejected_count=rejected_count,
                        next_person=current_person,
                        accepted_count=accepted_count
                    )

                decision_data = decide_and_next(
                    game_id, person_index, accept=accept)
                next_person = decision_data.get("nextPerson")

                admitted_count = decision_data.get(
                    "admittedCount", admitted_count)
                rejected_count = decision_data.get(
                    "rejectedCount", rejected_count)

                if accept and current_person and current_person.get("attributes"):
                    for attr_id, has_attr in current_person.get("attributes", {}).items():
                        if has_attr and attr_id in accepted_count:
                            accepted_count[attr_id] += 1

                # Also log accepted_count for debugging
                decision_data["acceptedCountDetails"] = accepted_count.copy()
                log_data(
                    decision_data, f"P{person_index:05d} {'ACCEPT' if accept else 'REJECT'}")

                status = decision_data.get("status", "running")

                if not next_person:
                    break

                current_person = next_person
                person_index = next_person["personIndex"]

            except Exception as e:
                log(f"Error during decision for person {person_index}: {e}")
                print(f"Error during decision for person {person_index}: {e}")
                break

        if status in ["completed", "failed"] and decision_data:
            log_data(decision_data, f"FINAL ({status.upper()})")

        log(f"Game ended with status: {status}, total decisions: {person_index + 1}")
        print(
            f"Game ended with status: {status}, total decisions: {person_index + 1}")

    except Exception as e:
        log(f"Unexpected error: {e}")
        print(f"Unexpected error: {e}")
    finally:
        close_logging()
