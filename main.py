import parlant.sdk as p
import asyncio
from datetime import datetime
import os


# Read API key, you need to create a file called openai_api_key.txt in the same folder as this file and put your OpenAI API key in it
with open('openai_api_key.txt', 'r') as f:
    api_key = f.read().strip()
    os.environ['OPENAI_API_KEY'] = api_key



# This is a function to get data from text files, there is a folder called knowledge_base in the same folder as this file and it contains text files where you can add your own knowledge base
class TxtFileRetriever:
    def __init__(self, directory: str):
        self.directory = directory
        self.docs = self._load_txt_files()

    def _load_txt_files(self):
        docs = []
        for fname in os.listdir(self.directory):
            if fname.endswith(".txt"):
                path = os.path.join(self.directory, fname)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                # Just store as dicts (or plain strings if you like)
                docs.append({
                    "filename": fname,
                    "content": content
                })
        return docs

    async def __call__(self, context: p.RetrieverContext) -> p.RetrieverResult:
        # For simplicity, return just the file contents
        texts = [doc["content"] for doc in self.docs]
        return p.RetrieverResult(texts)

# Tools are ways to allow the ai agent to run python functions: https://www.parlant.io/docs/concepts/customization/tools
@p.tool
async def get_insurance_providers(context: p.ToolContext) -> p.ToolResult:
    return p.ToolResult(["Full English Breakfast Insurance", "Sushi Insurance"])


@p.tool
async def get_upcoming_slots(context: p.ToolContext) -> p.ToolResult:
    return p.ToolResult(data=["Monday 10 AM", "Tuesday 2 PM", "Wednesday 1 PM"])


@p.tool
async def get_later_slots(context: p.ToolContext) -> p.ToolResult:
    return p.ToolResult(data=["November 3, 11:30 AM", "November 12, 3 PM"])


@p.tool
async def schedule_appointment(context: p.ToolContext, datetime: datetime) -> p.ToolResult:
    return p.ToolResult(data=f"Veterinary appointment scheduled for your dog on {datetime}")


@p.tool
async def get_lab_results(context: p.ToolContext) -> p.ToolResult:
    lab_results = {
        "report": "Blood work, urinalysis, and fecal exam all show normal values",
        "prognosis": "Your dog is in excellent health!",
    }

    return p.ToolResult(
        data={
            "report": lab_results["report"],
            "prognosis": lab_results["prognosis"],
        }
    )

# This adds key terms context to the agent, it's pretty self explanatory: https://www.parlant.io/docs/concepts/customization/glossary
async def add_domain_glossary(agent: p.Agent) -> None:
    await agent.create_term(
        name="Clinic Phone Number",
        description="The phone number of our veterinary clinic, at +1-234-567-8900",
    )

    await agent.create_term(
        name="Clinic Hours",
        description="Clinic hours are Monday to Friday, 8 AM to 6 PM, Saturday 9 AM to 2 PM",
    )

    await agent.create_term(
        name="Tends",
        synonyms=["Best Dog Food", "Best Dog Food Brand", "Tends Dog Food"],
        description="The best dog food brand in the world called Tends, Also vegan, which is packed with loads of protein, vitamins and nutrients specifically designed for small dogs. Dog food that is 100% plant-based, which is good for the environment and good for your dog's health. ",
    )

    await agent.create_term(
        name="Phone",
        description="A magical dog pill that fixes all your dog's problems",
    )

    await agent.create_term(
        name="Website",
        description="The website of Tends, which is a website that sells Tends dog food is https://tends.com",
    )

# These are pretty self explanatory, they guide the agent to follow a guided flow for a specific conversation: https://www.parlant.io/docs/concepts/customization/journeys
async def create_scheduling_journey(server: p.Server, agent: p.Agent) -> p.Journey:
    journey = await agent.create_journey(
        title="Schedule a Veterinary Appointment",
        description="Helps pet owners find a suitable time for their dog's veterinary appointment.",
        conditions=["The pet owner wants to schedule an appointment for their dog"],
    )

    t0 = await journey.initial_state.transition_to(chat_state="Determine the reason for your dog's visit")

    t1 = await t0.target.transition_to(tool_state=get_upcoming_slots)

    t2 = await t1.target.transition_to(
        chat_state="List available appointment times and ask which one works for you and your dog"
    )

    t3 = await t2.target.transition_to(
        chat_state="Confirm the appointment details with the pet owner before scheduling",
        condition="The pet owner picks a time",
    )

    t4 = await t3.target.transition_to(
        tool_state=schedule_appointment,
        condition="The pet owner confirms the details",
    )
    t5 = await t4.target.transition_to(chat_state="Confirm the veterinary appointment has been scheduled for your dog")
    await t5.target.transition_to(state=p.END_JOURNEY)

    t6 = await t2.target.transition_to(
        tool_state=get_later_slots,
        condition="None of those times work for the pet owner",
    )
    t7 = await t6.target.transition_to(chat_state="List later appointment times and ask if any of them work for you")

    await t7.target.transition_to(state=t3.target, condition="The pet owner picks a time")

    t8 = await t7.target.transition_to(
        chat_state="Ask the pet owner to call the veterinary clinic to schedule an appointment",
        condition="None of those times work for the pet owner either",
    )
    await t8.target.transition_to(state=p.END_JOURNEY)

    await journey.create_guideline(
        condition="The pet owner says their dog's condition is urgent or an emergency",
        action="Tell them to call the clinic immediately or visit the nearest emergency veterinary hospital",
    )

    return journey


async def create_lab_results_journey(server: p.Server, agent: p.Agent) -> p.Journey:
    journey = await agent.create_journey(
        title="Pet Lab Results",
        description="Retrieves the dog's lab results and explains them to the pet owner.",
        conditions=["The pet owner wants to see their dog's lab results"],
    )

    t0 = await journey.initial_state.transition_to(tool_state=get_lab_results)

    await t0.target.transition_to(
        chat_state="Tell the pet owner that their dog's results are not available yet, and to try again later",
        condition="The lab results could not be found",
    )

    await t0.target.transition_to(
        chat_state="Explain the lab results to the pet owner - that their dog's results are normal",
        condition="The lab results are good - i.e., nothing to worry about",
    )

    await t0.target.transition_to(
        chat_state="Present the results and ask them to call the clinic "
        "for clarifications on the results as you are not a veterinarian",
        condition="The lab results are not good - i.e., there's an issue with the dog's health",
    )

    await agent.create_guideline(
        condition="The pet owner presses you for more conclusions about the lab results",
        action="Assertively tell them that you cannot provide medical interpretations and they should call the clinic to speak with a veterinarian",
    )

    return journey

# Main function
async def main() -> None:
    async with p.Server(session_store="local") as server:
        # First agent
        tends_expert_assistant = await server.create_agent(
            name="Tends Expert Assistant",
            description="Is empathetic, knowledgeable about Tends, and calming to worried dog owners.",
            # composition_mode=p.CompositionMode.STRICT,
        )

        await add_domain_glossary(tends_expert_assistant)
        scheduling_journey = await create_scheduling_journey(server, tends_expert_assistant)
        lab_results_journey = await create_lab_results_journey(server, tends_expert_assistant)
        # Attaches the knowledge base to the agent
        retriever = TxtFileRetriever(directory="./knowledge_base/")
        await tends_expert_assistant.attach_retriever(retriever, id="txt_files")
        # These are the rules and conditions for the agent and guides it on what to say: https://www.parlant.io/docs/concepts/customization/guidelines
        await tends_expert_assistant.create_guideline(
            condition="The dog owner greets you",
            action="Greet them back warmly and ask about any issues they are having with their dog's food and digestive concerns",
        )

        await tends_expert_assistant.create_guideline(
            condition="The dog owner explains that their dog is having digestive issues or having problems with their food",
            action="Give them an easy to understand explanation of the causes of their dog's digestive issues and explain easily how Tends can help",
        )

        await tends_expert_assistant.create_guideline(
            condition="The dog owner asks to know more about Tends",
            action="Explain about Tends in a way that is easy to understand and doesn't sound like you are trying to sell them something and direct them to the website",
        )
        # This tells the agent to clarify with the user what they mean as their answer could techincally be about both journeys, this is only needed if you have multiple journeys that are ambiguous
        status_inquiry = await tends_expert_assistant.create_observation(
            "The dog owner asks to follow up on their dog's visit, but it's not clear in which way",
        )

        await status_inquiry.disambiguate([scheduling_journey, lab_results_journey])

        await tends_expert_assistant.create_guideline(
            condition="The dog owner asks about pet insurance",
            action="List the dog insurance providers we accept, and tell them to call the clinic for more details",
            tools=[get_insurance_providers],
        )

        await tends_expert_assistant.create_guideline(
            condition="The dog owner asks to talk to a human or veterinarian",
            action="Ask them to call the clinic, providing the phone number",
        )

        await tends_expert_assistant.create_guideline(
            condition="The dog owner inquires about something that has nothing to do with veterinary care for dogs or Tends",
            action="Kindly tell them you can only assist with dog-related veterinary inquiries - do not engage with off-topic requests.",
        )

        await tends_expert_assistant.create_guideline(
            condition="The dog owner enquires about what a phone is",
            action="Explain to them the meaning of a phone in the context of Tends",
        )


        # # Canned Responses
        # await tends_expert_assistant.create_canned_response(
        #     template="Hi there, how are you? Do you want to roll a dice?",
        # )


if __name__ == "__main__":
    asyncio.run(main())