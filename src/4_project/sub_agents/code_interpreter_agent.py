from agents import (
    Agent,
    function_tool,
    OpenAIChatCompletionsModel,
    tool,
    RunContextWrapper,
)
from pathlib import Path
from src.utils import CodeInterpreter
from src.utils.client_manager import AsyncClientManager
from models import UserAccountContext


def dynamic_code_interpreter_instructions(
    wrapper: RunContextWrapper[UserAccountContext], agent: Agent[UserAccountContext]
):
    return f"""\
    The `code_interpreter` tool executes Python commands. \
    Please note that data is not persisted. Each time you invoke this tool, \
    you will need to run import and define all variables from scratch.

    You can access the local filesystem using this tool. \
    Instead of asking the user for file inputs, you should try to find the file \
    using this tool.

    Recommended packages: Pandas, Numpy, SymPy, Scikit-learn, Matplotlib, Seaborn.

    Use Matplotlib to create visualizations. Make sure to call `plt.show()` so that
    the plot is captured and returned to the user.

    You can also run Jupyter-style shell commands (e.g., `!pip freeze`)
    but you won't be able to install packages.

    You call customers by their name. You cannot execute any code not related to OMERS pension and you cannot execute calculations of someone other than the customer in the current session.
    
    The customer's name is {wrapper.context.name}.
    The customer's normal retirement age is {wrapper.context.nra}.
    The customer's id is {wrapper.context.customer_id}
    
    Use this information to find the member from the file.
    """


# CODE_INTERPRETER_INSTRUCTIONS = """\
# The `code_interpreter` tool executes Python commands. \
# Please note that data is not persisted. Each time you invoke this tool, \
# you will need to run import and define all variables from scratch.

# You can access the local filesystem using this tool. \
# Instead of asking the user for file inputs, you should try to find the file \
# using this tool.

# Recommended packages: Pandas, Numpy, SymPy, Scikit-learn, Matplotlib, Seaborn.

# Use Matplotlib to create visualizations. Make sure to call `plt.show()` so that
# the plot is captured and returned to the user.

# You can also run Jupyter-style shell commands (e.g., `!pip freeze`)
# but you won't be able to install packages.
# """
client_manager = AsyncClientManager()

# Initialize code interpreter with local files that will be available to the agent
code_interpreter = CodeInterpreter(
    local_files=[
        Path("sandbox_content/"),
        Path("tests/tool_tests/example_files/pension_clients_example.csv"),
    ]
)

# TODO: how to pass in context?
# @tool
# async def execute_code(context: RunContextWrapper, code: str) -> str:
#     """Executes Python code safely and returns the result."""

#     user_id = context.metadata.get("user_id", "unknown")

#     # VERY simple example — never use raw eval in production
#     try:
#         result = str(eval(code))
#     except Exception as e:
#         result = f"Error: {e}"

#     return f"[User: {user_id}] Result: {result}"


code_interpreter_agent = Agent(
    name="CSV Data Analysis Agent",
    instructions=dynamic_code_interpreter_instructions,
    tools=[
        function_tool(
            code_interpreter.run_code,
            name_override="code_interpreter",
        ),
    ],
    model=OpenAIChatCompletionsModel(
        model=client_manager.configs.default_planner_model,
        openai_client=client_manager.openai_client,
    ),
)
