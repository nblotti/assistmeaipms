import re
from enum import Enum
from typing import Literal, Optional, TypedDict, List

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, field_validator


class OrderType(str, Enum):
    buy = "BUY"
    sell = "SELL"


class OrderClass(str, Enum):
    normal = "N"
    # Add other specific classifications as needed


order_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "OrderInitiation",
    "type": "object",
    "properties": {
        "order_confirmed": {
            "type": "boolean",
            "description": "Indicates if the order is confirmed",
            "default": "false"
        },
        "intervenant": {
            "type": ["string", "null"],
            "description": "The identifier (not the name) for the intervening party in the order process. It is a 7-character alphanumeric string.",
            "example": "0001091"
        },
        "security_id": {
            "type": ["string", "null"],
            "description": "The identifier of the financial security. The format is 6 alphanumeric characters, a dot, followed by 3 alphanumeric characters (e.g., '507170.000').",
            "example": "507170.000"
        },
        "orderType": {
            "type": ["string", "null"],
            "description": "The type of order being placed, can be either BUY or SELL",
            "enum": ["BUY", "SELL"],
            "example": "BUY",
            "default": "BUY"
        },
        "quantity": {
            "type": ["number", "null"],
            "description": "The quantity of the financial security being traded",
            "example": 1000.5
        },
        "orderClass": {
            "type": ["string", "null"],
            "description": "The class of the order, such as N for normal order",
            "example": "N",
            "default": "N"
        }
    },
    "required": ["order_confirmed"],
    "additionalProperties": False
}


class OrderInitiation(BaseModel):
    order_confirmed: bool
    intervenant: Optional[str] = Field(
        default=None,
        description="The identifier for the intervening party in the order process",
    )
    security_id: Optional[str] = Field(
        default=None,
        description="The id of the financial security",
    )
    orderType: Optional[OrderType] = Field(
        default=None,
        description="The type of order being placed, can be either BUY or SELL",
    )
    quantity: Optional[float] = Field(
        default=None,
        description="The quantity of the financial security being traded",
    )
    orderClass: Optional[OrderClass] = Field(
        default="N",
        description="The classification of the order, such as N for normal order or other specific classifications",
    )


class EnhancedOrderInitiation(OrderInitiation):
    order_confirmed: bool
    intervenant: str = Field(
        description="The identifier for the intervening party in the order process (format:7 digits)",
    )
    security_id: str = Field(
        description="The id of the financial security (format: 6 digits, dot, 3 digits)",
    )
    orderType: OrderType = Field(
        description="The type of order being placed, can be either BUY or SELL",
    )
    quantity: float = Field(
        description="The quantity of the financial security being traded",
    )
    orderClass: OrderClass = Field(
        description="The classification of the order, such as N for normal order or other specific classifications",
    )

    @field_validator('intervenant')
    def validate_intervenant(cls, intervenant):
        if not intervenant or not re.match(r'^[\da-zA-Z]{7}$', intervenant):
            raise ValueError("It seems that the intervenant is missing or incomplete, please correct")
        return intervenant

    @field_validator('security_id')
    def validate_security(cls, security_id):
        if not security_id or not re.match(r"^[\da-zA-Z]{6}\.[\da-zA-Z]{3}$", security_id):
            raise ValueError("It seems that the security is missing or incomplete, please correct")
        return security_id

    @field_validator('quantity')
    def validate_quantity(cls, quantity):
        if quantity is None or quantity <= 0:
            raise ValueError("Quantity must be a positive number.")
        return quantity


class AiAnalysis(BaseModel):
    orderInitiation: OrderInitiation = Field(
        description="The initial details of the order provided. May be incomplete and require further information.")
    comment: str = Field(
        description="Question or comment from the AI prompting for additional details to complete the order "
                    "initiation if needed.")


class ConversationMessage(BaseModel):
    data_type: Literal["Human", "System", "AI"]
    data: str
    payload_type: Optional[Literal["FinancialPosition", "OrderInitiation"]] = None
    payload: Optional[str] = None


class ConversationState(TypedDict):
    messages: List[BaseMessage]
    conversation_messages: List[ConversationMessage]
    conversation_type: Literal["FinancialPosition", "OrderInitiation"]
    order_initiation: AiAnalysis
    question: str
