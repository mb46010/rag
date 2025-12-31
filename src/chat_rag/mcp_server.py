"""
FastMCP server providing employee information tools.

This server exposes two MCP tools:
- get_employee_profile: Returns employee information
- get_time_off_balance: Returns remaining PTO days
"""

import logging
from typing import Optional
from fastmcp import FastMCP
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("HR Employee Services")


# Pydantic models for validation
class EmployeeProfile(BaseModel):
    """Employee profile information."""
    email: str = Field(..., description="Employee email address")
    name: str = Field(..., description="Employee first name")
    surname: str = Field(..., description="Employee last name")
    role: str = Field(..., description="Employee job role")
    tenure: str = Field(..., description="Years of service")
    band: str = Field(..., description="Employee band/level")
    country: str = Field(..., description="Employee country")
    division: str = Field(..., description="Employee division")


class TimeOffBalance(BaseModel):
    """Time off balance information."""
    email: str = Field(..., description="Employee email address")
    number_of_days_left: int = Field(..., description="Remaining PTO days")


# Mock employee database
EMPLOYEE_DATABASE = {
    "john.doe@company.com": {
        "email": "john.doe@company.com",
        "name": "John",
        "surname": "Doe",
        "role": "Software Engineer",
        "tenure": "3 years",
        "band": "IC3",
        "country": "CH",
        "division": "Engineering"
    },
    "jane.smith@company.com": {
        "email": "jane.smith@company.com",
        "name": "Jane",
        "surname": "Smith",
        "role": "Product Manager",
        "tenure": "5 years",
        "band": "IC4",
        "country": "CH",
        "division": "Product"
    },
    "mario.rossi@company.com": {
        "email": "mario.rossi@company.com",
        "name": "Mario",
        "surname": "Rossi",
        "role": "Senior Engineer",
        "tenure": "7 years",
        "band": "IC5",
        "country": "IT",
        "division": "Engineering"
    },
    "sarah.jones@company.com": {
        "email": "sarah.jones@company.com",
        "name": "Sarah",
        "surname": "Jones",
        "role": "Engineering Manager",
        "tenure": "2 years",
        "band": "M3",
        "country": "CH",
        "division": "Engineering"
    },
    "test@company.com": {
        "email": "test@company.com",
        "name": "Test",
        "surname": "User",
        "role": "Software Engineer",
        "tenure": "1 year",
        "band": "IC2",
        "country": "CH",
        "division": "Engineering"
    }
}

# Mock time off balances
TIME_OFF_DATABASE = {
    "john.doe@company.com": 15,
    "jane.smith@company.com": 22,
    "mario.rossi@company.com": 28,
    "sarah.jones@company.com": 12,
    "test@company.com": 20
}


def get_employee_profile(email: str) -> EmployeeProfile:
    """
    Retrieve employee profile information by email.

    Args:
        email: Employee email address

    Returns:
        EmployeeProfile with all employee information

    Raises:
        ValueError: If email is not found in the system
    """
    logger.info(f"get_employee_profile called with email: {email}")

    if email not in EMPLOYEE_DATABASE:
        logger.warning(f"Email not found: {email}")
        raise ValueError(f"Employee with email '{email}' not found in the system")

    profile_data = EMPLOYEE_DATABASE[email]
    profile = EmployeeProfile(**profile_data)

    logger.info(f"Returning profile for: {profile.name} {profile.surname}")
    return profile


def get_time_off_balance(email: str) -> TimeOffBalance:
    """
    Retrieve remaining PTO (Paid Time Off) days for an employee.

    Args:
        email: Employee email address

    Returns:
        TimeOffBalance with remaining days

    Raises:
        ValueError: If email is not found in the system
    """
    logger.info(f"get_time_off_balance called with email: {email}")

    if email not in TIME_OFF_DATABASE:
        logger.warning(f"Email not found in time off database: {email}")
        raise ValueError(f"Employee with email '{email}' not found in the system")

    days_left = TIME_OFF_DATABASE[email]
    balance = TimeOffBalance(email=email, number_of_days_left=days_left)

    logger.info(f"Returning time off balance for {email}: {days_left} days")
    return balance


# Register tools with FastMCP
@mcp.tool()
def get_employee_profile_tool(email: str) -> dict:
    """Get employee profile information by email."""
    profile = get_employee_profile(email)
    return profile.model_dump()


@mcp.tool()
def get_time_off_balance_tool(email: str) -> dict:
    """Get the number of PTO days remaining for an employee."""
    balance = get_time_off_balance(email)
    return balance.model_dump()


if __name__ == "__main__":
    logger.info("Starting FastMCP HR Employee Services server on localhost:9000")
    logger.info(f"Mock employees: {list(EMPLOYEE_DATABASE.keys())}")

    # Run the FastMCP server
    mcp.run(transport="sse", host="localhost", port=9000)
