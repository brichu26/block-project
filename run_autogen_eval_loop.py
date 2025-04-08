import os
import re
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import sys
import pickle
import random
import yaml
import time
import numpy as np
from datetime import datetime

# External packages
import wandb  # Make sure to run `wandb login` beforehand

# ALFWorld & Agents
import alfworld.agents.modules.generic as generic
import alfworld.agents.environment as environment
from gwt_agent import GWTAutogenAgent
from baseline_agent import BaselineAutogenAgent
from autogen import ConversableAgent, register_function, GroupChat, GroupChatManager

global_num_games_to_evaluate = 139
global_max_actions_per_game = 60
global_max_chat_rounds_per_game = 500
global_split_rounds_per_game = 1
base_path = os.path.join("runs", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
os.makedirs(base_path, exist_ok=True)

def parse_arguments():
    """
    Parse command-line arguments for evaluating Autogen Agents on the ALFWorld environment.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate different Autogen Agents on the ALFWorld environment."
    )
    parser.add_argument("config_file", help="Path to the YAML config file")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--baseline", action="store_true", help="Use BaselineAutogenAgent")
    group.add_argument("--gwt", action="store_true", help="Use GWTAutogenAgent")

    parser.add_argument("--long_term_guidance", action="store_true", help="Enable long-term guidance")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # Select the agent class
    if args.baseline:
        agent_class = BaselineAutogenAgent
        agent_name = "BaselineAutogenAgent"
    elif args.gwt:
        agent_class = GWTAutogenAgent
        agent_name = "GWTAutogenAgent"
    else:
        raise ValueError("No agent specified. Use --baseline or --gwt.")

    print(f"Selected Agent: {agent_name}")

    # Load config
    with open(args.config_file) as reader:
        config = yaml.safe_load(reader)

    # Initialize Weights & Biases
    wandb.init(
        project="cognitive_agents",
        entity="eduardocortes1100-university-of-california-berkeley")

    # Setup API key
    API_KEY = os.environ.get("BLOCK_KEY")
    llm_config = {"config_list": [{"model": "gpt-4o", "api_key": API_KEY}]}

    # Initialize Agent
    agent = agent_class(
        llm_config,
        log_path=base_path,
        max_chat_round=global_max_chat_rounds_per_game,
        max_actions=global_max_actions_per_game,
        rounds_per_game=global_split_rounds_per_game,
        args=args
    )

    # Extract evaluation parameters
    eval_paths = config["general"]["evaluate"]["eval_paths"]
    eval_envs = config["general"]["evaluate"]["envs"]
    controllers = config["general"]["evaluate"]["controllers"]
    repeats = config["general"]["evaluate"]["repeats"]

    chat_round_list = []

    for eval_env_type in eval_envs:
        for controller_type in (controllers if eval_env_type == "AlfredThorEnv" else ["tw"]):
            for eval_path in eval_paths:
                print(f"Evaluating: {eval_path}")

                # Configure the evaluation environment
                config["general"]["evaluate"]["env"]["type"] = eval_env_type
                config["dataset"]["eval_ood_data_path"] = eval_path
                config["controller"]["type"] = controller_type

                env_class = getattr(environment, eval_env_type)
                alfred_env = env_class(config, train_eval="eval_out_of_distribution")
                env = alfred_env.init_env(batch_size=1)
                total_num_games = alfred_env.num_games

                # Random selection of evaluation games
                if global_num_games_to_evaluate > total_num_games:
                    num_games_to_evaluate = total_num_games
                else:
                    num_games_to_evaluate = global_num_games_to_evaluate

                selected_games = sorted(random.sample(range(1, total_num_games + 1), num_games_to_evaluate))
                #selected_games = [124, 125] #[35, 94, 124, 125] #[35, 52, 57, 69, 77, 79, 86, 93, 94, 107, 109, 121, 123, 124, 125, 139]
                #num_games_to_evaluate = len(selected_games)
                print(f"Selected {num_games_to_evaluate} Games: {selected_games}")

                error_list = []
                success_list = []
                failure_list = []

                # Track metrics
                cumulative_successful_actions = 0
                avg_actions_taken_per_successful_game = 0
                cumulative_failing_actions = 0
                avg_actions_taken_per_failing_game = 0

                cumulative_successful_chat_rounds = 0
                avg_chat_rounds_per_successful_game = 0
                cumulative_failing_chat_rounds = 0
                avg_chat_rounds_per_failing_game = 0

                cumulative_successful_runtime = 0
                avg_runtime_per_successful_game = 0
                cumulative_failing_runtime = 0
                avg_runtime_per_failing_game = 0

                cumulative_runtime = 0

                num_games_evaluated = 0
                num_successes = 0
                success_rate = 0
                num_games_no_error = 0

                for i in range(1, total_num_games + 1):
                    obs, info = env.reset()

                    if i not in selected_games:
                        print(f"Skipped Game #{i}")
                        continue

                    num_games_evaluated += 1
                    agent.set_environment(env, obs, info, i)
                    log_paths = agent.log_paths
                    print(f"\n[Running Game #{i}]")
                    print(f"Evaluation {num_games_evaluated} of {num_games_to_evaluate}")

                    start_time = time.time()
                    try:
                        chat_result, error_message = agent.run_chat(agent.initial_message)
                    except Exception as e:
                        error_message = str(e)
                        chat_result = None
                        print(f"Chat Error: {error_message}")
                    end_time = time.time()

                    # Log errors
                    if error_message:
                        error_list.append(i)
                        with open(log_paths['error_message_path'], "a") as f:
                            f.write(f"Run Chat: {error_message}\n")

                    # Log chat history
                    if chat_result and getattr(chat_result, "chat_history", []):
                        with open(log_paths['chat_history_path'], "w") as f:
                            for message in chat_result.chat_history:
                                f.write('-' * 20 + '\n')
                                for key in ["name", "role", "content"]:
                                    if key in message:
                                        f.write(f"{key}:\n{message[key]}\n" if key == "content" else f"{key}: {message[key]}\n")
                                for k, v in message.items():
                                    if k not in ["name", "role", "content"]:
                                        f.write(f"{k}: {v}\n")
                        chat_round_list.append(len(chat_result.chat_history))
                    else:
                        chat_round_list.append(-1)
                        with open(log_paths['chat_history_path'], "w") as f:
                            f.write("Error Message: no chat history in chat result\n")

                    # Read the chat history to extract meta-data
                    with open(log_paths['chat_history_path'], "r") as f:
                        chat_text = f.read()

                    transitions = []
                    transition_pattern = r"name: (.*)"
                    matches = re.findall(transition_pattern, chat_text)
                    for idx in range(len(matches) - 1):
                        transitions.append({
                            "from": matches[idx],
                            "to": matches[idx + 1],
                            "step": idx
                    })

                    # Save transitions to file
                    transition_path = os.path.join(os.path.dirname(log_paths['chat_history_path']),
                                                   "transition_log.json")
                    with open(transition_path, "w") as f:
                        json.dump(transitions, f, indent=2)

                    belief_state_pattern = r"Belief State: (.*)"
                    matches = re.findall(belief_state_pattern, chat_text, re.IGNORECASE)
                    if matches:
                        agent.prev_episodic_memories.append({"episode_number": num_games_evaluated, "task_outcome": agent.task_status , "memory": matches})
                    else:
                        agent.prev_episodic_memories.append({"episode_number": num_games_evaluated, "task_outcome": agent.task_status , "memory": agent.curr_episodic_memory})

                    # Evaluate and log success
                    elapsed_minutes = (end_time - start_time) / 60
                    cumulative_runtime += elapsed_minutes

                    success = agent.success
                    if success:
                        num_successes += 1
                        success_list.append(i)
                        cumulative_successful_actions += agent.num_actions_taken
                        cumulative_successful_chat_rounds += chat_round_list[-1]
                        cumulative_successful_runtime += elapsed_minutes
                        avg_actions_taken_per_successful_game = cumulative_successful_actions / num_successes
                        avg_chat_rounds_per_successful_game = cumulative_successful_chat_rounds / num_successes
                        avg_runtime_per_successful_game = cumulative_successful_runtime / num_successes
                    else:
                        num_failures = num_games_evaluated - num_successes
                        failure_list.append(i)
                        cumulative_failing_actions += agent.num_actions_taken
                        cumulative_failing_chat_rounds += chat_round_list[-1]
                        cumulative_failing_runtime += elapsed_minutes
                        avg_actions_taken_per_failing_game = cumulative_failing_actions / num_failures
                        avg_chat_rounds_per_failing_game = cumulative_failing_chat_rounds / num_failures
                        avg_runtime_per_failing_game = cumulative_failing_runtime / num_failures

                    success_rate = num_successes / num_games_evaluated

                    num_games_no_error = num_games_evaluated - len(
                        [game for game in error_list if game not in success_list])
                    error_adjusted_success_rate = num_successes / num_games_no_error

                    wandb.log({
                        "game_no": i,
                        "success": int(success),
                        "actions_taken": agent.num_actions_taken,
                        "success_rate": success_rate,
                        "avg_actions_taken_per_successful_game": avg_actions_taken_per_successful_game,
                        "avg_chat_rounds_per_successful_game": avg_chat_rounds_per_successful_game,
                        "avg_runtime_per_successful_game": avg_runtime_per_successful_game,
                        "runtime": elapsed_minutes,
                        "cumulative_runtime": cumulative_runtime,
                        "chat_rounds": chat_round_list[-1],
                        "error_adjusted_success_rate": error_adjusted_success_rate
                    }, step=num_games_evaluated)

                    print(f"[Ran Game #{i}]")
                    print(f"Evaluation {num_games_evaluated} of {num_games_to_evaluate}")
                    print(f"Success: {success}")
                    print(f"Runtime: {elapsed_minutes:.2f} minutes")
                    print(f"Rounds Taken: {global_split_rounds_per_game - agent.rounds_left} out of {global_split_rounds_per_game}")
                    print(f"Actions Taken: {agent.num_actions_taken} out of {global_max_actions_per_game}")
                    print(f"Chat Rounds Taken: {chat_round_list[-1]}")
                    print(f"Success Rate: {num_successes}/{num_games_evaluated} = {100 * success_rate:.2f}%")
                    print(f"Average Actions per Successful Game: {avg_actions_taken_per_successful_game:.2f} out of {global_max_actions_per_game}")
                    print(f"Average Chat Rounds per Successful Game: {avg_chat_rounds_per_successful_game:.2f} out of {global_max_chat_rounds_per_game}")
                    print(f"Average Runtime per Successful Game: {avg_runtime_per_successful_game:.2f} minutes")
                    print(f"Average Actions per Failing Game: {avg_actions_taken_per_failing_game:.2f} out of {global_max_actions_per_game}")
                    print(f"Average Chat Rounds per Failing Game: {avg_chat_rounds_per_failing_game:.2f} out of {global_max_chat_rounds_per_game}")
                    print(f"Average Runtime per Failing Game: {avg_runtime_per_failing_game:.2f} minutes")
                    print(f"Successes: {success_list}")
                    print(f"Failures: {failure_list}")
                    print(f"Errors: {error_list}")
                    print(f"Error-Adjusted Success Rate: {num_successes}/{num_games_no_error} = {100 * error_adjusted_success_rate if num_games_no_error > 0 else 0:.2f}%")
                    print(f"Remaining Games: {selected_games[num_games_evaluated:]}")

                    total_seconds = int(cumulative_runtime * 60)
                    hours = total_seconds // 3600
                    minutes = (total_seconds % 3600) // 60
                    secs = total_seconds % 60
                    print(f"Cumulative Runtime: {hours:02}:{minutes:02}:{secs:02}\n")

                    if not selected_games[num_games_evaluated:]:
                        break

                # Final Success Summary
                print(f"Final Success Rate: {num_successes}/{num_games_evaluated} = {100 * success_rate:.2f}%")
                print(f"Final Error-Adjusted Success Rate: {num_successes}/{num_games_no_error} = {100 * num_successes / num_games_no_error if num_games_no_error > 0 else 0:.2f}%")

    wandb.finish()
