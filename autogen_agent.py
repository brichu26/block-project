import os


# This is a template class for the Autogen Agent
class AutogenAgent:
    def __init__(self, llm_config, log_path, game_no=1, max_chat_round=400, max_actions=50, args=None, env=None, obs="",
                 info=None):
        self.env = env
        self.obs = obs
        self.info = info
        self.game_no = game_no
        self.max_chat_round = max_chat_round
        self.llm_config = llm_config
        self.log_path = log_path
        self.num_actions_taken = 0
        self.max_actions = max_actions
        self.success = False
        self.args = args
        self.start_agent = None
        self.log_paths = {}
        self.result_dict = {}

        self.group_chat = None
        self.group_chat_manager = None

    def initialize_autogen(self):
        self.register_log_paths()
        self.initialize_agents()
        self.register_functions()
        self.initialize_groupchat()

    def initialize_agents(self):
        raise NotImplementedError

    def register_functions(self):
        raise NotImplementedError

    def initialize_groupchat(self):
        raise NotImplementedError

    def run_chat(self, initial_message_content):
        assert self.start_agent is not None, "self.start_agent must be defined"
        assert self.group_chat_manager is not None, "self.group_chat_manager must be defined"
        assert self.group_chat is not None, "self.group_chat must be defined"

        self.num_actions_taken = 0
        self.success = False

        chat_result = None
        error_message = None
        try:
            # Start the chat with the Planner Agent proposing a plan
            chat_result = self.start_agent.initiate_chat(
                self.group_chat_manager,
                message={"role": "system", "content": initial_message_content},
                summary_method="reflection_with_llm"
            )
        except Exception as e:
            print(f"Group Chat manager fails to chat with error message {e}")
            error_message = e

        return chat_result, error_message

    def resume_chat(self, last_message):
        chat_result = None
        error_message = None
        try:
            last_agent, last_message = self.group_chat_manager.resume(messages=last_message)

            # Resume the chat using the last agent and message
            chat_result = last_agent.initiate_chat(recipient=self.group_chat_manager, message=last_message,
                                                   clear_history=False)

        except Exception as e:
            print(f"Group Chat manager fails to chat with error message {e}")
            error_message = e

        return chat_result, error_message

    def register_log_paths(self):

        game_path = os.path.join(self.log_path, f"game_{self.game_no}")
        os.makedirs(game_path, exist_ok=True)

        task_path = os.path.join(game_path, "task.txt")
        history_path = os.path.join(game_path, "history.txt")
        rule_path = os.path.join(game_path, "rules.txt")
        admissible_commands_path = os.path.join(game_path, "admissible_commands.txt")
        chat_history_path = os.path.join(game_path, "chat_history.txt")
        message_path = os.path.join(game_path, "last_message.pkl")
        result_path = os.path.join(game_path, "result.txt")
        error_message_path = os.path.join(game_path, "error_message.txt")

        # get all the previous game path
        previous_game_path = [os.path.join(self.log_path, f"game_{i}") for i in range(self.game_no)]
        previous_rule_path = [os.path.join(game_path, "rules.txt") for game_path in previous_game_path]

        self.log_paths = {
            "task_path": task_path,
            "history_path": history_path,
            "rule_path": rule_path,
            "admissible_commands_path": admissible_commands_path,
            "chat_history_path": chat_history_path,
            "message_path": message_path,
            "result_path": result_path,
            "error_message_path": error_message_path,
            "previous_rule_path": previous_rule_path,
        }

    def get_log_paths(self):
        return self.log_paths
