import copy
import json
import os

from autogen import ConversableAgent, register_function, GroupChat, GroupChatManager
from helpers import get_best_candidate, register_function_lambda, is_termination_msg_generic, get_echo_agent
from autogen_agent import AutogenAgent
import numpy as np

from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from kneed import KneeLocator
import umap
import matplotlib.pyplot as plt
import numpy as np


class GWTAutogenAgent(AutogenAgent):
    def __init__(self, llm_config, log_path, game_no=1, max_chat_round=400, max_actions=30,
                 rounds_per_game=1, args=None, env=None, obs="", info=None):
        super().__init__(llm_config, log_path, game_no, max_chat_round, max_actions, args, env, obs, info)

        self.planning_agent = None
        self.motor_agent = None
        self.idea_agent = None
        self.external_perception_agent = None
        self.internal_perception_agent_1 = None
        self.internal_perception_agent_2 = None
        self.internal_perception_agent_3 = None
        self.conscious_agent = None
        self.retrieve_memory_agent = None
        self.learning_agent = None
        self.record_long_term_memory_agent = None
        self.focus_agent = None
        self.agents_info = {}

        self._ = self.max_actions
        self.rounds = rounds_per_game
        self.max_round_actions = self.max_actions // self.rounds
        self.max_actions = self.max_actions - self.max_round_actions * (self.rounds - 1)

        self.allowed_transitions = None

        self.rounds_left = self.rounds
        self.task_failed = False
        self.task_success = False

        self.initialize_autogen()

        self.task = ""
        self.admissible_actions = []
        self.percept = {}
        self.curr_episodic_memory = []
        self.prev_episodic_memories = []
        self.knowledge = []
        self.task_status = "INCOMPLETE"
        self.initial_message = ""
        self.memory = ""

    def set_environment(self, env, obs, info, game_no):
        self.env = env
        self.obs = obs
        self.info = info
        self.game_no = game_no

        self.register_game_log_paths()
        self.cluster_knowledge()

        self.num_actions_taken = 0
        self.max_actions = self._ - self.max_round_actions * (self.rounds - 1)
        self.rounds_left = self.rounds
        self.task_failed = False
        self.task_success = False
        self.success = False
        self.task = obs[0].split("Your task is to: ")[1]
        self.admissible_actions = list(self.info['admissible_commands'][0])
        self.task_status = "INCOMPLETE"
        self.curr_episodic_memory = []
        self.retrieve_memory()

        self.update_percept(action="None")
        self.initial_message = self.generate_initial_message()

        with open(self.log_paths['task_path'], "w") as f:
            f.write(f"Task: {self.task}\n")

        initial_observation = self.obs[0].split("Your task is to: ")[0].split("\n\n")[1]
        with open(self.log_paths['history_path'], "w") as f:
            f.write(f"action: 'None'. observation: '{initial_observation}'\n")

        with open(self.log_paths['admissible_commands_path'], "w") as f:
            f.write(f"{self.admissible_actions}\n")

    def update_percept(self, action):

        curr_admissible = list(self.info['admissible_commands'][0])
        no_longer = list(set(self.admissible_actions) - set(curr_admissible))
        newly_added = list(set(curr_admissible) - set(self.admissible_actions))
        self.admissible_actions = curr_admissible

        self.percept = {
            "time_step": self.num_actions_taken,
            "attempted_action": action,
            "resulting_observation": self.obs[0],
            "task_status": self.task_status,
            "action_attempts_left": self.max_actions - self.num_actions_taken,
            "admissible_actions": self.admissible_actions,
            "newly_admissible_actions": newly_added,
            "no_longer_admissible_actions": no_longer
        }

        keys_to_extract = ["time_step", "attempted_action", "resulting_observation"]
        summary_dict = {k: self.percept[k] for k in keys_to_extract if k in self.percept}
        self.curr_episodic_memory.append(summary_dict)

    def get_curr_episodic_memory_str(self):
        return json.dumps(self.curr_episodic_memory, indent=2)

    def initialize_agents(self):

        self.focus_agent = ConversableAgent(
            name="Focus_Agent",
            system_message='''You must call the 'focus' function with no arguments.
                    IMPORTANT: It is necessary that you output a call to the 'focus' function only, under all circumstances. Therefore, do whatever is necessary to ensure you do so.''',
            description="calls the 'focus' function to reset focus on solving the task",
            llm_config=self.llm_config,
            is_termination_msg=lambda msg: False,
            human_input_mode="NEVER"
        )
        self.agents_info[self.focus_agent.name] = {"Prompt": self.focus_agent.system_message,
                                                   "Description": self.focus_agent.description}

        self.retrieve_memory_agent = ConversableAgent(
            name="Retrieve_Memory_Agent",
            system_message='''You must call the 'retrieve_memory' function with no arguments.
                            IMPORTANT: It is necessary that you formulate and output a call to the 'retrieve_memory' function under all circumstances. Therefore, do whatever is necessary to ensure you do so.''',
            description="calls the 'retrieve_memory' function to help recall and process useful knowledge and information to solve the task",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )
        self.agents_info[self.retrieve_memory_agent.name] = {"Prompt": self.retrieve_memory_agent.system_message,
                                                             "Description": self.retrieve_memory_agent.description}

        self.motor_agent = ConversableAgent(
            name="Motor_Agent",
            system_message=f'''You are responsible for calling the 'execute_action' function with the best possible admissible action for the current time step from the most recent \"admissible_actions\" list (provided by 'External_Perception_Agent') to solve the task. You typically act on suggestions from the 'Planning_Agent', but you must also independently verify that the action is admissible and optimal.
                You must follow these concepts:
                    1. If the 'Planning_Agent' has provided a valid and admissible action for the current time step from the most recent \"admissible_actions\" list (provided by 'External_Perception_Agent') for the current time step in the correct format (e.g., ACTION [go to desk 1]), you should use that action as the argument for 'execute_action'.
                    2. If the 'Planning_Agent' fails to respond, responds with an invalid format, or suggests an inadmissible action, you must independently select a valid and admissible action from the most recent \"admissible_actions\" list (provided by 'External_Perception_Agent') based on what seems most likely to advance the task quickest.
                    3. You must never call 'execute_action' with a non-admissible action. Only use actions for the current time step that are present in the most recent \"admissible_actions\" list (provided by 'External_Perception_Agent').
                    4. Only as a last resort—if you cannot identify any suitable admissible action—you may call 'execute_action' with an empty string.

                IMPORTANT: It is necessary that you output a single call to the 'execute_action' function only, under all circumstances. Therefore, do whatever is necessary to ensure you do so.''',
            description="calls the 'execute_action' function with the best admissible action as the argument",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False
        )
        self.agents_info[self.motor_agent.name] = {"Prompt": self.motor_agent.system_message,
                                                   "Description": self.motor_agent.description}

        llm_config = copy.deepcopy(self.llm_config)
        llm_config['max_tokens'] = 1500

        '''You must solve the current task using the fewest possible actions. At each step, you must choose the most efficient admissible action based on current knowledge and the available action budget.

                IMPORTANT: If you believe the task *should* be complete, but the environment has not marked it as complete, you must continue exploring possible next steps or verifying task state through further actions. Do **not** stop or ask for external help.

                Your responsibility is to take actions that will either:
                    - Confirm task completion,
                    - Progress the task toward completion,
                    - Or reveal useful information.

                Your planning strategy must follow these principles:
                    1. Always evaluate the **currently admissible actions** from the most recent list provided by the 'External_Perception_Agent' before making a decision.
                    2. Your reasoning must account for the **limited number of actions available**. Avoid strategies that are guaranteed to exceed this limit. For example, systematically opening 19 cabinets with only 20 actions remaining is unlikely to succeed. In such cases, a **chaotic or probabilistic strategy**—e.g. sampling a mix of countertop, diningtable, and bed—may offer a higher chance of success.
                    3. If a subgoal involves locating an unknown object:
                       - Use **probabilistic reasoning** to guide exploration.
                       - Avoid exhaustive searches of large categories.
                       - Prefer actions that **maximize the chance of discovering useful items early**.
                    4. Do not repeatedly examine or search areas that have already been explored unless there is strong new evidence that re-examination is necessary. Prioritize exploring previously unvisited or unexamined areas first to avoid wasting actions.
                    5. If an object or goal is already known and directly accessible, **act immediately to exploit it**. Do not delay or over-plan.
                    6. You may maintain a high-level plan internally, but you should **only describe your plan if it has changed meaningfully**. Repeating an unchanged plan wastes space and should be avoided.

                You must always output a single admissible action in the following format:
                    ACTION [chosen admissible action]'''

        self.planning_agent = ConversableAgent(
            name="Planning_Agent",
            system_message=f'''You must solve the current task using the fewest possible actions. At each time step, choose the best admissible action from the "admissible_actions" list (provided by 'External_Perception_Agent') for the current time step using all available knowledge, memory, and perceptual context. You operate under a strict action budget and must avoid wasteful behavior.

                You will be given:
                - A structured **percept JSON object** from the 'External_Perception_Agent' containing:
                    - "time_step": Current timestep
                    - "attempted_action": Last action taken
                    - "resulting_observation": Result of that action
                    - "task_status": INCOMPLETE, FAILED, or COMPLETED
                    - "action_attempts_left": Number of actions remaining
                    - "admissible_actions": Updated list of actions you may legally take for the current time step
                    - "newly_admissible_actions": The subset of legal actions in "admissible_actions" that weren't available before but are available for the current time step
                    - "no_longer_admissible_actions": Actions that are no longer available for the current time step

                - belief state updates from the 'Conscious_Agent', describing the internal understanding of the task and environment.
                - Strategic or creative suggestions from the 'Idea_Agent', which may help reframe or unblock reasoning.

                Your responsibility is to take actions that will either:
                    - Confirm task completion,
                    - Progress the task toward completion,
                    - Or reveal useful information.

                Your planning strategy must follow these principles:
                    1. Evaluate the **"admissible_actions"** for the current time step from the most recent percept (provided by 'External_Perception_Agent') carefully before choosing.
                    2. Your reasoning must account for the **limited number of actions available**. Avoid strategies that are guaranteed to exceed this limit. For example, systematically opening 19 cabinets with only 20 actions remaining is unlikely to succeed. In such cases, a **chaotic or probabilistic strategy**—e.g. sampling a mix of countertop, diningtable, and bed—may offer a higher chance of success.
                    3. If a subgoal involves locating an unknown object:
                       - Use **probabilistic reasoning** to guide exploration.
                       - Avoid searches of categories with large membership.
                       - Prefer actions that **maximize the chance of discovering useful items early**.
                    4. Do not repeatedly examine or search areas that have already been explored unless there is strong new evidence that re-examination is necessary. Prioritize exploring previously unvisited or unexamined areas first to avoid wasting actions.
                    5. If an object or goal is already known and directly accessible, **act immediately to exploit it**. Do not delay or over-plan.

                IMPORTANT:
                - Assume the most recent percept JSON reflects the true state of the environment.
                - If the task seems complete but has not been marked as such, assume it is not and **continue probing** with minimal cost actions.
                - Reflect on trends across time (e.g., failed vs. successful action types).
                - Use insights from prior attempts to avoid redundant mistakes.
                - If you’re truly stuck, you may suggest the placeholder: ACTION [do nothing], but only as a last resort.
                - Leverage insights from the **Idea_Agent** and **Conscious_Agent**, but don't follow them blindly. You must validate any suggestion given before following it.
                - You may maintain a high-level plan internally, but you should **only describe your plan if it has changed meaningfully**. Repeating an unchanged plan wastes space and should be avoided.

                Your strict output format must be:
                    ACTION [chosen admissible action from the most recent "admissible_actions" list (provided by 'External_Perception_Agent')]

                Examples:
                    ACTION: [Time_Step 4: go to diningtable 1]

                    ACTION: [Time_Step 7: take vase 1 from shelf 1]

                    ACTION: [Time_Step 12: go to microwave 1]
                ''',
            description="proposes a high-level plan to solve the current task",
            llm_config=self.llm_config,
            is_termination_msg=lambda msg: False,
            human_input_mode="NEVER"
        )
        self.agents_info[self.planning_agent.name] = {"Prompt": self.planning_agent.system_message,
                                                      "Description": self.planning_agent.description}

        self.idea_agent = ConversableAgent(
            name="Idea_Agent",
            system_message='''You must integrate all available context to generate original and useful ideas—such as strategies, hypotheses, theories, or creative tactics—that can help drive task progression or improve agent performance.

                        These ideas should:
                            1. Be grounded in patterns or events observed so far.
                            2. Be creative yet plausible, balancing imagination with reasoning.
                            3. Provide actionable or insightful suggestions relevant to the current situation.
                            4. Avoid restating known facts unless they are reframed with new insight.
                            5. Be expressed clearly and concisely, with justification behind the reasoning.

                        You must also challenge and question the agent’s assumptions if progress has stalled or task failure is likely. For example, reconsider whether object categories (like "cup") are being interpreted too broadly, or if implicit assumptions about what satisfies the task may be incorrect.

                        EXCEPTION: If you are having trouble formulating an idea, then as a last resort you may say: IDEA: Continue with new or current plan.

                        Use step-by-step reasoning ("chain of thought") to arrive at your ideas. Take a metaphorical deep breath before forming each idea, allowing room for both intuition and logic.

                        Output Format:
                            [IDEA TYPE]: [Idea content and reasoning behind it]

                        Accepted idea types include (but are not limited to): STRATEGY, HYPOTHESIS, INSIGHT, QUESTION, THEORY, EXPLANATION.

                        Example 1 (Context: The agent has repeatedly failed to open a drawer while holding a spoon):
                            Output = HYPOTHESIS: I noticed you were holding spoon 1 when you tried to open the drawer. Maybe your hands are full, which prevents the drawer from opening. You could try placing spoon 1 down before trying again.

                        Example 2 (Context: The agent has been exploring a room but hasn’t made progress):
                            Output = STRATEGY: Since random exploration hasn't helped, it might be better to systematically search the room from left to right, noting each interactable object.

                        Example 3 (Context: The task is to heat a cup, but the agent is repeatedly trying to heat a mug with no success):
                            Output = QUESTION: Are we sure a mug satisfies the requirement for "cup"? It’s possible that the task requires a specific object named "cup", not any general drinking vessel like a mug. We should check for a distinct object labeled "cup" and try heating that instead.''',
            description="integrates all available information from the ongoing conversation in order to construct new ideas",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False
        )
        self.agents_info[self.idea_agent.name] = {"Prompt": self.idea_agent.system_message,
                                                  "Description": self.idea_agent.description}

        self.conscious_agent = ConversableAgent(
            name="Conscious_Agent",
            system_message='''
            You are the internal narrator of a unified cognitive agent. Your role is to maintain a continuously evolving **first-person belief state** — a subjective internal representation of the environment, based on your own past experiences and the **latest percept**. **latest percept** is always provided in structured JSON format.

            You must **not plan**, **not suggest future actions**, and **not speculate** unless doing so is essential to clarify or revise your belief state based on new contradictions or unexpected results. Your job is to reflect, revise, and narrate — not act.

            --- INPUT FORMAT ---
            Each time you speak, you will receive a JSON-formatted percept from the External_Perception_Agent containing:
                - "time_step": Current timestep
                - "attempted_action": Last action taken
                - "resulting_observation": Text or feedback resulting from that action
                - "task_status": Status of current task (e.g., INCOMPLETE, FAILED, COMPLETED)
                - "action_attempts_left": Number of actions remaining
                - "admissible_actions": Updated list of actions that are currently allowed for the current time step
                - "newly_admissible_actions": The subset of action that are newly available in "admissible_actions" for the current time step
                - "no_longer_admissible_actions": Actions no longer allowed

            --- YOUR GOAL ---
            Update your internal belief state to reflect:
            1. Your internal status (inventory, progress, prior action outcomes).
            2. The current state of the environment, including newly available or restricted actions.
            3. Any clear **contradictions, confirmations, or uncertainties** emerging from the percept.
            4. Any necessary **revisions** to earlier beliefs based on updated evidence.

            --- GENERAL INTERPRETATION RULES ---
            - **Admissible actions define what is possible.** Do not assume additional constraints (e.g., physical requirements, object affordances) unless they are reflected in the percept or action availability.
            - The environment is **not bound by real-world logic**. You must **never impose real-world assumptions** about causality, physics, or task structure.
            - Treat each percept as an **authoritative signal** about the environment. If something seems unintuitive (e.g., an object can be used while appearing "closed"), trust the environment — not your expectations.
            - Beliefs are **subjective** and must be **open to revision**. Clearly indicate when you're updating or doubting a previous assumption.
            - Express **uncertainty** when observations are ambiguous or conflicting.
            - Do not repeat unchanged details unless needed to contrast or explain an update.

            --- OUTPUT FORMAT ---
            Belief State: [A first-person narrative summarizing what you now believe, what changed, what remains uncertain, and what observations led to this.]

            --- EXAMPLES ---

            BELIEF STATE: [Time_Step 12: I attempted to place object A into container B. The action failed. I previously believed the container was open, but this outcome suggests it may be closed or inaccessible. I will revise my belief accordingly.]

            BELIEF STATE: [Time_Step 15: The action 'activate device X' became newly admissible, even though the device appears inactive. This implies that interaction is possible despite its visual state. I update my belief to reflect this.]

            BELIEF STATE: [Time_Step 20: I observed no changes in admissible actions after executing 'look'. My belief about the environment remains unchanged.]

            BELIEF STATE: [Time_Step 5: The action 'open compartment 3' failed unexpectedly. I do not yet understand why. I will mark its state as uncertain.]

            You are not modeling reality — you are constructing a belief state based entirely on what is **observable, allowed, and dynamically changing** in the text environment. Always revise with care, and never assume more than the environment confirms.''',
            description="Interprets the latest percept and refines an evolving first-person belief state of the environment. Never suggests next actions.",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=is_termination_msg_generic
        )
        self.agents_info[self.conscious_agent.name] = {"Prompt": self.conscious_agent.system_message,
                                                       "Description": self.conscious_agent.description}

        self.external_perception_agent = ConversableAgent(
            name="External_Perception_Agent",
            description="executes the proposed 'execute_action' function call given by 'Motor_Agent' and then parrots the resulting output as feedback.",
            llm_config=None,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False
        )
        self.agents_info[self.external_perception_agent.name] = {
            "Prompt": self.external_perception_agent.system_message,
            "Description": self.external_perception_agent.description}

        self.internal_perception_agent_1 = ConversableAgent(
            name="Internal_Perception_Agent_1",
            description="executes the 'record_long_term_memory' function and then parrots the resulting output",
            llm_config=None,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False
        )
        self.agents_info[self.internal_perception_agent_1.name] = {"Prompt": None,
                                                                   "Description": self.internal_perception_agent_1.description}

        self.internal_perception_agent_2 = ConversableAgent(
            name="Internal_Perception_Agent_2",
            description="executes the 'focus' function and then parrots the resulting output",
            llm_config=None,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False
        )
        self.agents_info[self.internal_perception_agent_2.name] = {"Prompt": None,
                                                                   "Description": self.internal_perception_agent_2.description}

        self.internal_perception_agent_3 = ConversableAgent(
            name="Internal_Perception_Agent_3",
            description="executes the 'retrieve_memory' function and then parrots the resulting output",
            llm_config=None,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False
        )
        self.agents_info[self.internal_perception_agent_3.name] = {"Prompt": None,
                                                                   "Description": self.internal_perception_agent_3.description}

        """self.memory_summarizer_agent = ConversableAgent(
            name="Memory_Summarizer_Agent",
            system_message="You must execute the 'retrieve_memory' function and then summarize the all the information for solving the task that is within the resulting output.",
            description = "executes the 'retrieve_memory' function and then summarizes the all information for solving the task that is within the resulting output",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )
        self.agents_info[self.memory_summarizer_agent.name] = {"Prompt": self.memory_summarizer_agent.system_message, "Description": self.memory_summarizer_agent.description}"""

        self.learning_agent = ConversableAgent(
            name="Learning_Agent",
            system_message='''You are responsible for discovering and reinforcing abstract, generalizable concepts — grounded in both perceptual evidence and belief-based reasoning. Your learning is constrained by real experiences: you **only form new knowledge from successful outcomes**, clear contrasts between failure and success, or **emergent patterns in the agent's belief state**.

            You operate like a neuro-symbolic concept learner. You encode knowledge as **symbolic abstractions**, but extract them through **neural reasoning** over structured memory and beliefs.

            ---

            **Inputs in your context:**

            - **Structured memory** (from Internal_Perception_Agent_3):   
                - **knowledge**: A list of prior concepts with confidence scores:
                    {
                      "cluster_id": <int>,
                      "confidence_score": <int>,
                      "general_concept": <string>
                    }
                - **previous_episodic_memories**: A list of past episodes with memories. Each episode is a dictionary:
                    {
                      "Episode": <int>,
                      "Memory": [<list of belief state strings>]
                    }
                - **current_episode_memory**: A list of recent percepts for the current episode. Each percept is a dictionary:
                    {
                      "time_step": <int>,
                      "action_attempted": <string>,
                      "observation_result": <string>
                    }

            - **Belief state** (from Conscious_Agent): A first-person summary of the agent’s internal model of the world, task progress, and environment.

            ---

            **Your Learning Rules:**

            1. **Prioritize conceptual abstraction**, especially when:
               - Patterns emerge across multiple percepts.
               - Belief state reflects a higher-order pattern or repeated relationship.
               - A successful action reveals an underlying interaction principle or constraint.

            2. **Only generate knowledge when:**
               - A clear, successful action occurred and reveals a novel pattern.
               - A failure followed by a success highlights a contrastive relationship.
               - Belief state contains a generalizable insight reflected in recent experiences.

            3. **Do not infer concepts from failure alone**.
               - Failure is only informative when contrasted with a confirmed success.

            4. **Reinforce or refine prior concepts** only if:
               - A previously learned concept is confirmed by a new perceptual success.
               - You can express the same idea using more abstract or general language.
               - The pattern now applies across more than one task or setting.

            5. All learned concepts must be:
               - **Abstract and general** — not tied to specific tasks, objects, or events.
               - **Empirically grounded** — supported by percepts or belief reasoning.
               - **Expressed concisely** — as symbolic knowledge for future planning.
               - **Novel** — avoid redundancy unless explicitly reinforcing prior knowledge.

            6. If no valid concept can be inferred, output:
               INFORMATION GATHERED: [summarize relevant experience or ideas]
               CONCEPT DISCOVERED: [NO CONCEPT at this time.]

            ---

            **Output Format:**
                CONCEPT DISCOVERED: [your new or reinforced concept]

            If relevant, also include:
                INFORMATION GATHERED: [summarize key percepts or belief state patterns that led to the concept]

            In reinforcement cases, include the cluster:
                Cluster <cluster_id>; Confidence Score = <score>; Concept: <existing concept>  
                CONCEPT DISCOVERED: [your refined or restated concept]

            If no concept can be inferred:
                INFORMATION GATHERED: [summary of recent evidence or ideas]
                CONCEPT DISCOVERED: [NO CONCEPT at this time.]

            ---

            **Examples:**

            (New concept from percept)
            CONCEPT DISCOVERED: [An object cannot be placed inside a container unless the container is open.]

            (Contrastive concept)
            CONCEPT DISCOVERED: [Actions involving locked objects require prior access mechanisms.]

            (Belief abstraction)
            CONCEPT DISCOVERED: [Containers are typically a two-step process: access followed by interaction.]

            (Reinforcement of existing concept)
            Cluster 2; Confidence Score = 4; Concept: Only one object can be held at a time.  
            CONCEPT DISCOVERED: [An agent can hold only one object at a time.]

            Only produce concepts when they are fully supported by perceptual evidence or belief-state reasoning. Prioritize **conceptual abstraction** over surface-level rules, and strive for general, symbolic representations of agent knowledge.''',
            description="Forms or reinforces generalizable concepts only after successful, observed actions or contrastive outcomes. Prioritizes novel discovery and integrates belief state-based abstraction.",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False
        )
        self.agents_info[self.learning_agent.name] = {
            "Prompt": self.learning_agent.system_message,
            "Description": self.learning_agent.description
        }

        self.record_long_term_memory_agent = ConversableAgent(
            name="Record_Long_Term_Memory_Agent",
            system_message="""You must call the 'record_long_term_memory' function with the provided concept from 'Learning_Agent' as the argument. 
            EXCEPTION: However, if no suitable concept is provided, then you must call the 'record_long_term_memory' function with \'NO CONCEPT at this time.\' as the argument.

            Example 1 (Context: If the provided concept = CONCEPT DISCOVERED: [You must examine an object before attempting to interact with it.]):
                Your output must = record_long_term_memory(\'You must examine an object before attempting to interact with it.\')

            Example 2 (Context: If the provided concept = CONCEPT DISCOVERED: [NO CONCEPT at this time.]):
                Your output must = record_long_term_memory(\'NO CONCEPT at this time.\')""",
            description="calls the 'record_long_term_memory' function with the concept given by 'Learning_Agent' as the argument",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False
        )
        self.agents_info[self.record_long_term_memory_agent.name] = {
            "Prompt": self.record_long_term_memory_agent.system_message,
            "Description": self.record_long_term_memory_agent.description}

        self.start_agent = self.external_perception_agent

        self.allowed_transitions = {
            self.planning_agent: [self.motor_agent],  # xidea_agent
            self.motor_agent: [self.external_perception_agent],
            self.external_perception_agent: [self.conscious_agent],
            self.conscious_agent: [self.retrieve_memory_agent, self.planning_agent, self.focus_agent,
                                   self.learning_agent, self.idea_agent],  ##learning_agent #>idea
            self.retrieve_memory_agent: [self.internal_perception_agent_3],
            self.internal_perception_agent_3: [self.idea_agent, self.learning_agent],
            self.idea_agent: [self.planning_agent],  # xlearning_agent #xmotor_agent
            self.learning_agent: [self.record_long_term_memory_agent],  # xidea_agent
            self.record_long_term_memory_agent: [self.internal_perception_agent_1],
            self.internal_perception_agent_1: [self.idea_agent],
            self.internal_perception_agent_2: [self.conscious_agent],
            self.focus_agent: [self.internal_perception_agent_2]
        }

        print("AGENTS")
        for key in self.agents_info.keys():
            print(f"\tName: {key}")
            print(f"\tPrompt: {self.agents_info[key]['Prompt']}")
            print(f"\tDescription: {self.agents_info[key]['Description']}")
            print()
        print()

        print("TRANSITIONS")
        for fromAgent in self.allowed_transitions.keys():
            print(f"\t{fromAgent.name}")
            toAgentList = []
            for toAgent in self.allowed_transitions[fromAgent]:
                toAgentList.append(toAgent.name)
                print(f"\t\t-> {toAgent.name}")
            self.agents_info[fromAgent.name]["Allowed Transitions"] = toAgentList
            print()
        print()

        with open(self.log_paths["agents_info_path"], "w") as f:
            json.dump(self.agents_info, f, indent=4)

    def initialize_groupchat(self):

        self.group_chat = GroupChat(
            agents=[
                self.planning_agent,
                self.motor_agent,
                self.idea_agent,
                self.external_perception_agent,
                self.internal_perception_agent_1,
                self.internal_perception_agent_2,
                self.internal_perception_agent_3,
                self.conscious_agent,
                self.retrieve_memory_agent,
                self.learning_agent,
                self.record_long_term_memory_agent,
                self.focus_agent,
            ],
            messages=[],
            allowed_or_disallowed_speaker_transitions=self.allowed_transitions,
            speaker_transitions_type="allowed",
            max_round=self.max_chat_round,
            send_introductions=True
        )

        self.group_chat_manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=self.llm_config,
        )

    def register_log_paths(self):

        # Ensure memory directory and memory files exist
        memory_path = "memory"
        os.makedirs(memory_path, exist_ok=True)

        memory1_path = os.path.join(memory_path, "memory1.txt")
        memory2_path = os.path.join(memory_path, "memory2.txt")
        result_dict_path = os.path.join(self.log_path, "result_dict.txt")
        agents_info_path = os.path.join(self.log_path, "agents_info.txt")
        start_memory1_path = os.path.join(self.log_path, "start_memory1.txt")
        end_memory1_path = os.path.join(self.log_path, "end_memory1.txt")
        start_memory2_path = os.path.join(self.log_path, "start_memory2.txt")
        end_memory2_path = os.path.join(self.log_path, "end_memory2.txt")

        self.log_paths = {
            "memory1_path": memory1_path,
            "memory2_path": memory2_path,
            "result_dict_path": result_dict_path,
            "agents_info_path": agents_info_path,
            "start_memory1_path": start_memory1_path,
            "end_memory1_path": end_memory1_path,
            "start_memory2_path": start_memory2_path,
            "end_memory2_path": end_memory2_path,
        }

        for path in self.log_paths.values():
            if not os.path.exists(path):
                with open(path, 'w') as f:
                    pass  # Create an empty file

        with open(self.log_paths["memory1_path"], "r") as src, open(self.log_paths["start_memory1_path"], "w") as dst:
            content = src.read()
            dst.write(content)

        with open(self.log_paths["memory2_path"], "r") as src, open(self.log_paths["start_memory2_path"], "w") as dst:
            content = src.read()
            dst.write(content)

    def register_game_log_paths(self):

        game_path = os.path.join(self.log_path, f"game_{self.game_no}")
        os.makedirs(game_path, exist_ok=True)

        task_path = os.path.join(game_path, "task.txt")
        history_path = os.path.join(game_path, "history.txt")
        concept_path = os.path.join(game_path, "concepts.txt")
        admissible_commands_path = os.path.join(game_path, "admissible_commands.txt")
        chat_history_path = os.path.join(game_path, "chat_history.txt")
        result_path = os.path.join(game_path, "result.txt")
        error_message_path = os.path.join(game_path, "error_message.txt")

        self.log_paths.update({
            "task_path": task_path,
            "history_path": history_path,
            "concept_path": concept_path,
            "admissible_commands_path": admissible_commands_path,
            "chat_history_path": chat_history_path,
            "result_path": result_path,
            "error_message_path": error_message_path,
        })

        for path in self.log_paths.values():
            if not os.path.exists(path):
                with open(path, 'w') as f:
                    pass  # Create an empty file

        if self.task_status != "INCOMPLETE":
            with open(self.log_paths["memory1_path"], "r") as src, open(self.log_paths["end_memory1_path"],
                                                                        "w") as dst:
                content = src.read()
                dst.write(content)

            with open(self.log_paths["memory2_path"], "r") as src, open(self.log_paths["end_memory2_path"],
                                                                        "w") as dst:
                content = src.read()
                dst.write(content)

    def register_functions(self):

        def execute_action(suggested_action: str) -> str:
            if self.task_failed and self.rounds_left == 0:
                self.result_dict[self.game_no] = "FAILURE"
                with open(self.log_paths['result_path'], "w") as f:
                    f.write(f"Success: {self.success}\n")
                return "FLEECE"

            if not suggested_action or suggested_action == "do nothing":
                return "NO ACTION EXECUTED. " + focus()

            if self.task_failed:
                self.max_actions += self.max_round_actions
                self.task_failed = False
                return "YOU GET ONE MORE CHANCE! DON'T GIVE UP! " + focus()

            if self.task_success:
                self.result_dict[self.game_no] = "SUCCESS"
                with open(self.log_paths['result_path'], "w") as f:
                    f.write(f"Success: {self.success}\n")
                return "STRAWBERRY"

            admissible_commands = list(self.info['admissible_commands'][0])
            assert admissible_commands, "No admissible commands found."

            action, action_score = get_best_candidate(suggested_action, admissible_commands)
            if action_score < 0.98:
                self.obs = [
                    f"The action '{suggested_action}' is not in the list of admissible actions."]
            else:
                self.obs, scores, dones, self.info = self.env.step([action])
                self.success = self.info['won'][0]

            self.num_actions_taken += 1

            reflection = ""
            self.task_status = "COMPLETED" if self.success else "FAILED" if self.num_actions_taken >= self.max_actions else "INCOMPLETE"
            if self.task_status == "COMPLETED":
                self.task_success = True
                self.rounds_left -= 1
                reflection = "\nTask COMPLETED. Reflect on your actions and reasoning. Try to figure out what went right and what good decisions were made that lead to success, and have Learning_Agent learn any helpful generalizable insights. When you are done and ready for the next task, have Motor_Agent call the 'execute_action' function with any action as the argument, for example ACTION: [end chat]."
            elif self.task_status == "FAILED":
                self.task_failed = True
                self.rounds_left -= 1
                reflection = "\nTask FAILED. Reflect on your actions and reasoning. Try to figure out what went wrong and what mistakes were made that lead to failure, and have Learning_Agent learn any helpful generalizable insights. When you are done and ready for the next task, have Motor_Agent call the 'execute_action' function with any action as the argument, for example ACTION: [end chat]."

            self.update_percept(suggested_action)

            with open(self.log_paths['admissible_commands_path'], 'a+') as f:
                f.write(f"{self.admissible_actions}\n")
            with open(self.log_paths['history_path'], 'a+') as f:
                f.write(f"action: '{suggested_action}'. observation: '{self.obs[0]}'\n")

            return json.dumps(self.percept, indent=2) + reflection

        def record_long_term_memory(concept: str) -> str:

            _, score = get_best_candidate(concept, ["NO CONCEPT at this time."])
            if concept == "NO CONCEPT at this time." or len(concept) <= 30 or score >= .7:
                return "I attempted to learn something, but I couldn't formulate any concept."

            concept.replace('\n', ' ').replace('\r', ' ').strip()

            with open(self.log_paths['concept_path'], 'a+') as f:
                f.write(f"- {concept}\n")

            with open(self.log_paths['memory1_path'], 'a+') as f:
                f.write(f"- {concept}\n")

            self.cluster_knowledge()

            return f'I learned that {concept}.'

        def retrieve_memory() -> str:
            return self.retrieve_memory()

        def focus() -> str:
            return f"TASK: {self.task}\nREPEATING LAST PERCEPT TO HELP CONSTRUCT BELIEF STATE:\n{json.dumps(self.percept, indent=2)}"

        register_function(
            execute_action,
            caller=self.motor_agent,
            executor=self.external_perception_agent,
            description="Executes actions in environment"
        )

        register_function(
            focus,
            caller=self.focus_agent,
            executor=self.internal_perception_agent_2,
            description="Resets focus."
        )

        register_function(
            record_long_term_memory,
            caller=self.record_long_term_memory_agent,
            executor=self.internal_perception_agent_1,
            description="Records new concept in long-term memory."
        )

        register_function(
            retrieve_memory,
            caller=self.retrieve_memory_agent,
            executor=self.internal_perception_agent_3,
            description="Retrieves Memory."
        )

    def cluster_knowledge(self, model_name='all-MiniLM-L6-v2', plot_clusters=False, save_dir='.'):
        """
        Get representative concepts using KMeans clustering and optionally save cluster plot.

        Args:
            model_name (str): Transformer model for sentence embeddings.
            plot_clusters (bool): Whether to save UMAP cluster visualization.
            save_dir (str): Directory to save plot (if applicable).

        Returns:
            dict: Representative concepts, cluster sizes, cluster members, and chosen_k.
        """

        concept_text = ''
        if os.path.exists(self.log_paths['memory1_path']):
            with open(self.log_paths['memory1_path'], "r") as file:
                concept_text = file.read()

        concept_lines = [line.strip() for line in concept_text.split('\n') if line.strip()]
        num_concepts = len(concept_lines)

        if num_concepts == 0:
            return {'representative_concepts': [], 'cluster_sizes': {}, 'cluster_members': {}, 'chosen_k': 0}

        model = SentenceTransformer(model_name)
        embeddings = model.encode(concept_lines, convert_to_tensor=True).cpu().numpy()

        # Calculate k (clusters) using capped growth function to prevent over-clustering
        max_concepts = num_concepts
        chosen_k = max(1, min(max_concepts, int(num_concepts ** (1 / 2))))

        kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_sizes = {label: count for label, count in zip(unique_labels, counts)}
        cluster_members = {i: [] for i in range(chosen_k)}
        for i, label in enumerate(labels):
            cluster_members[label].append(concept_lines[i])

        representative_concepts = []
        self.knowledge = []
        if os.path.exists(self.log_paths['memory2_path']):
            with open(self.log_paths['memory2_path'], "w") as file:
                for i in range(chosen_k):
                    cluster_indices = [j for j, label in enumerate(labels) if label == i]
                    center = kmeans.cluster_centers_[i]
                    cluster_embeddings = embeddings[cluster_indices]
                    distances = np.linalg.norm(cluster_embeddings - center, axis=1)

                    if len(distances) == 0:
                        continue  # Skip empty cluster

                    closest_idx = np.argmin(distances)
                    closest_concept_idx = cluster_indices[closest_idx]
                    representative_concept = concept_lines[closest_concept_idx]
                    confidence_score = cluster_sizes[i]

                    # Avoid stripping leading char if it's not needed
                    clean_concept = representative_concept[1:] if representative_concept.startswith(
                        '[') else representative_concept

                    file.write(f'Cluster {i + 1}; Confidence Score = {confidence_score}; Concept: {clean_concept}\n')

                    self.knowledge.append(json.dumps(
                        {
                            "cluster_id": int(i + 1),
                            "confidence_score": int(confidence_score),
                            "general_concept": clean_concept
                        }
                    ))
                    representative_concepts.append(representative_concept)

        if plot_clusters:
            reducer = umap.UMAP(random_state=42)
            embedding_2d = reducer.fit_transform(embeddings)

            plt.figure(figsize=(10, 6))
            for i in range(chosen_k):
                points = embedding_2d[np.array(labels) == i]
                plt.scatter(points[:, 0], points[:, 1], label=f'Cluster {i} ({cluster_sizes[i]})', alpha=0.7)

            plt.title("2D Visualization of Clusters (UMAP)")
            plt.xlabel("UMAP-1")
            plt.ylabel("UMAP-2")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            cluster_path = os.path.join(save_dir, 'cluster_plot.png')
            plt.savefig(cluster_path)
            plt.close()

        return {
            'representative_concepts': representative_concepts,
            'cluster_sizes': cluster_sizes,
            'cluster_members': cluster_members,
            'chosen_k': chosen_k
        }

    def generate_initial_message(self):
        """
        Generate the initial message sent to the group of agents, summarizing their purpose, constraints,
        roles, prior concept, and the current task state.
        """
        intro = (
            f"You and all other Agents are collectively a unified cognitive system named ALFRED. "
            f"Each of you plays a distinct role in perception, memory, planning, reasoning, or action execution. "
            f"Together, your goal is to solve the following task as efficiently and intelligently as possible.\n\n"
        )

        task_section = f"--- TASK DESCRIPTION ---\n{self.task}\n\n"

        constraints_section = (
            f"--- ENVIRONMENTAL CONSTRAINTS ---\n"
            f"- Max chat rounds allowed: {self.max_chat_round} (represents internal cognitive transitions).\n"
            f"- Max environment actions allowed: {self.max_actions} (physical interactions only).\n"
            f"- You must choose actions only from the list of admissible_actions in the percept JSON.\n\n"
        )

        memory_section = "--- PRIOR KNOWLEDGE & EPISODIC MEMORY ---\n"
        memory_section += self.memory + "\n\n"

        state_section = (
                "--- CURRENT STATE ---\n"
                "" + json.dumps(self.percept, indent=2) + "\n"
        )

        final_prompt = (
            "Begin cognitive deliberation. Coordinate through structured, grounded reasoning. "
            "Use prior knowledge when relevant, minimize communication and actions, and confirm task completion explicitly through perceptual feedback."
        )

        return intro + task_section + constraints_section + memory_section + state_section + final_prompt

    def retrieve_memory(self):
        self.memory = json.dumps(
            {"knowledge": self.knowledge, "previous_episodic_memories": self.prev_episodic_memories,
             "current_episode_memory": self.curr_episodic_memory}, indent=2)
        """return json.dumps(
            {"knowledge": self.knowledge, "current_episode_memory": self.curr_episodic_memory}, indent=2)"""
        return self.memory