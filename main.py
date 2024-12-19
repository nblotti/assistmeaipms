import os
from typing import List

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import ValidationError
from starlette.config import undefined
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware

from model import AiAnalysis, ConversationMessage, ConversationState, EnhancedOrderInitiation, \
    order_schema

if os.getenv("ENVIRONNEMENT") == "PROD":
    load_dotenv("config/.env")
else:
    load_dotenv()

from fastapi import FastAPI

from tool import chat_gpt_4o, chat_gpt_4


# Middleware to disable caching
class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0, post-check=0, pre-check=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response


app = FastAPI(
    title="SORB AI",
    description="First version API SORB with AI",
    version="v0",
    openapi_url="/openapi.json",
)

# CORS origins allowed
origins = ["*"]
# Middleware for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware
app.add_middleware(NoCacheMiddleware)


def categorize_order(state: ConversationState):
    conversation_messages: List[ConversationMessage] = state["conversation_messages"]

    messages = []
    for message in conversation_messages:
        if message.data_type == "AI":
            if message.payload is not undefined:
                messages.append(AIMessage(content=message.payload))
        messages.append(AIMessage(content=message.data))
    return {"messages": messages}


def create_order(state: ConversationState):
    system_prompt = (

        """ Populate an OrderInitiation object using AI-human chat interaction, adhering strictly to handling unclear or 
        incomplete order details without altering the original inputs or using one input for multiple fields. Always 
        verify and clarify missing information based on the template default and on your educated guesses, and ensure 
        that each field input complies with the JSON schema requirements including format and length. But NEVER use an 
        input to complete two different fields (example intervenant and security) in the  order and NEVER truncate or 
        change an input.
        
        Interpret the structure provided to accurately identify and manage missing fields :
        
        {order_initiation_class_structure}


        Example Process:

        Received User Input: '12334567 2000 50'
        Analysis: Fields are missing, 1234567 match the intervenant required length. Based on the schema orderClass and
        OrderType are missing. Use the template default ('N' an 'BUY' and ask the user to provide the security id"
        Expected response : Thank you for initiating your order. It looks like we're missing some information to 
        proceed: Could you please provide the intervenant ID and confirm that the other fields are correct? Your 
        cooperation is greatly appreciated.
        """
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="messages"),

        ]
    ).partial(order_initiation_class_structure=order_schema)

    content_parser_pydantic = PydanticToolsParser(tools=[AiAnalysis])
    create_content_chain = (
            prompt | chat_gpt_4.bind_tools([AiAnalysis],
                                           tool_choice="AiAnalysis") | content_parser_pydantic)

    # create_content_chain = prompt | chat_gpt_4o
    res = create_content_chain.invoke({"messages": state["messages"]})

    print(res[0])
    return {"order_initiation": res[0]}


def validate_order(state: ConversationState, validate_content_chain=None):
    system_prompt = (

        """
         Given a user's initial input related to initiating an order and a list of validation errors 
        pertaining to missing or incorrect required fields, create a response that politely prompts the user 
        to address these errors in a functional, non-technical manner. NEVER use variable type (like string, boolean,
        etc, in your response use the names. Don't ask two times informations for the same field
        
        Parameters:
        - initial_message (str): 
        {initial_message}
        
        - validation_errors :
        {validation_errors}
        
         ) 
        """
    )
    order: AiAnalysis = state["order_initiation"]
    try:
        # Assuming EnhancedOrderInitiation is a pre-defined class similar to OrderInitiation with additional validation
        enhanced_order = EnhancedOrderInitiation(**order.orderInitiation.model_dump())
        state["order_initiation"].orderInitiation = enhanced_order
    except ValidationError as e:
        print(e)
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_prompt),

            ]
        )

        validate_content_chain = prompt | chat_gpt_4o.bind_tools([AIMessage],
                                                                 tool_choice="AIMessage") | PydanticToolsParser(
            tools=[AIMessage])

        res = validate_content_chain.invoke(input={"validation_errors": str(e), "initial_message": order.comment})

        order.comment = res[0].content

        return {"order_initiation": order}
    pass


def create_response(state: ConversationState):
    conversation = ConversationMessage(
        data_type="AI",
        data=state["order_initiation"].comment,
        payload_type="OrderInitiation",
        payload=state["order_initiation"].orderInitiation.model_dump_json()
    )
    return {"conversation_messages": state["conversation_messages"] + [conversation]}


workflow = StateGraph(ConversationState)

workflow.add_node("categorize_order", categorize_order)
workflow.add_node("create_order", create_order)
workflow.add_node("validate_order", validate_order)
workflow.add_node("create_response", create_response)

workflow.add_edge(START, "categorize_order")
workflow.add_edge("categorize_order", "create_order")
workflow.add_edge("create_order", "validate_order")
workflow.add_edge("validate_order", "create_response")
workflow.add_edge("create_response", END)
graph = workflow.compile()


@app.options("/{rest_of_path:path}")
async def preflight_handler():
    return


@app.post("/orderinitiation/")
async def load_default_values(message_list: List[ConversationMessage]):
    inputs = {"conversation_messages": message_list}

    result = graph.invoke(inputs)
    return result["conversation_messages"]
