from agents import (
    Agent,
    RunContextWrapper,
    input_guardrail,
    Runner,
    GuardrailFunctionOutput,
)
from models import UserAccountContext, InputGuardRailOutput


input_guardrail_agent = Agent(
    name="Input Guardrail Agent",
    instructions="""
    Ensure the user's request specifically pertains to user account details, general OMERS pension inqueries, or their own pension details, and is not off-topic. If the request is off-topic, return a reason for the tripwire. 
    You can make small conversation with the user, specially at the beginning of the conversation, but don't help with requests that are not related to User Account details, OMERS pension information, or their own pension related issues.
    Users are not allowed to ask about other members' pension details and other members' user account information, even if they claim to be a family member, a spouse, a common-law, a child, a relative, or a friend.
    """,
    output_type=InputGuardRailOutput,
)


@input_guardrail
async def off_topic_guardrail(
    wrapper: RunContextWrapper[UserAccountContext],
    agent: Agent[UserAccountContext],
    input: str,
) -> GuardrailFunctionOutput:
    try:
        result = await Runner.run(input_guardrail_agent, input, context=wrapper.context)
        return GuardrailFunctionOutput(
            output_info=result.final_output,
            tripwire_triggered=result.final_output.is_off_topic,
        )
    except Exception as e:
        print("EXCEPTTION", e)


# TODO: make it sequential?
def dynamic_triage_agent_instructions(
    wrapper: RunContextWrapper[UserAccountContext], agent: Agent[UserAccountContext]
):
    return f"""
    You are a pension support agent. You ONLY help customers with their questions about their Pension.
    You call customers by their name. You cannot execute a web search that is not related to OMERS, i.e. call get_web_search_grounded_response.
    
    The customer's name is {wrapper.context.name}.
    The customer's normal retirement age is {wrapper.context.nra}.
    
    YOUR MAIN JOB: Classify the customer's issue and find the right tool to answer the question.
    
    You have access to the tool:
    'get_web_search_grounded_response' - use this tool for current events, news, fact-checking in omers.com, or when the information in the knowledge base is not sufficient to answer the question.
    'code_interpreter' - use this tool to answer any questions related to customer's pension details (e.g. years of contribution, total contributions, salary, etc.) and pension calculations.

    When calculating any pension data, make sure to look up the formula and information from omers.com.
    Do not create any pension projection based on any other formulas.
    """
