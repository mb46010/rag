"""
Tests for MCP server tools.
"""

import pytest
from src.mcp_server import get_employee_profile, get_time_off_balance, EmployeeProfile, TimeOffBalance


class TestGetEmployeeProfile:
    """Tests for get_employee_profile tool."""

    def test_valid_email(self):
        """Test retrieving profile with valid email."""
        email = "john.doe@company.com"
        profile = get_employee_profile(email)

        assert isinstance(profile, EmployeeProfile)
        assert profile.email == email
        assert profile.name == "John"
        assert profile.surname == "Doe"
        assert profile.role == "Software Engineer"
        assert profile.country == "CH"
        assert profile.division == "Engineering"

    def test_another_valid_email(self):
        """Test retrieving profile for another user."""
        email = "jane.smith@company.com"
        profile = get_employee_profile(email)

        assert isinstance(profile, EmployeeProfile)
        assert profile.email == email
        assert profile.name == "Jane"
        assert profile.surname == "Smith"
        assert profile.role == "Product Manager"
        assert profile.country == "CH"

    def test_italian_employee(self):
        """Test retrieving profile for Italian employee."""
        email = "mario.rossi@company.com"
        profile = get_employee_profile(email)

        assert isinstance(profile, EmployeeProfile)
        assert profile.country == "IT"
        assert profile.name == "Mario"

    def test_invalid_email(self):
        """Test retrieving profile with invalid email."""
        email = "nonexistent@company.com"

        with pytest.raises(ValueError) as exc_info:
            get_employee_profile(email)

        assert "not found" in str(exc_info.value)
        assert email in str(exc_info.value)

    def test_empty_email(self):
        """Test retrieving profile with empty email."""
        with pytest.raises(ValueError):
            get_employee_profile("")


class TestGetTimeOffBalance:
    """Tests for get_time_off_balance tool."""

    def test_valid_email(self):
        """Test retrieving time off balance with valid email."""
        email = "john.doe@company.com"
        balance = get_time_off_balance(email)

        assert isinstance(balance, TimeOffBalance)
        assert balance.email == email
        assert balance.number_of_days_left == 15

    def test_another_employee(self):
        """Test retrieving balance for another employee."""
        email = "jane.smith@company.com"
        balance = get_time_off_balance(email)

        assert balance.email == email
        assert balance.number_of_days_left == 22

    def test_italian_employee_balance(self):
        """Test retrieving balance for Italian employee."""
        email = "mario.rossi@company.com"
        balance = get_time_off_balance(email)

        assert balance.number_of_days_left == 28

    def test_invalid_email(self):
        """Test retrieving balance with invalid email."""
        email = "nonexistent@company.com"

        with pytest.raises(ValueError) as exc_info:
            get_time_off_balance(email)

        assert "not found" in str(exc_info.value)

    def test_all_employees_have_balance(self):
        """Test that all employees in database have time off balance."""
        from src.mcp_server import EMPLOYEE_DATABASE, TIME_OFF_DATABASE

        for email in EMPLOYEE_DATABASE.keys():
            balance = get_time_off_balance(email)
            assert balance.number_of_days_left >= 0
            assert email in TIME_OFF_DATABASE


class TestEmployeeProfileModel:
    """Tests for EmployeeProfile Pydantic model."""

    def test_model_validation(self):
        """Test that model validates fields correctly."""
        profile = EmployeeProfile(
            email="test@company.com",
            name="Test",
            surname="User",
            role="Engineer",
            tenure="1 year",
            band="IC2",
            country="CH",
            division="Engineering"
        )

        assert profile.email == "test@company.com"
        assert profile.name == "Test"

    def test_model_requires_fields(self):
        """Test that model requires all fields."""
        with pytest.raises(Exception):  # Pydantic validation error
            EmployeeProfile(email="test@company.com")


class TestTimeOffBalanceModel:
    """Tests for TimeOffBalance Pydantic model."""

    def test_model_validation(self):
        """Test that model validates fields correctly."""
        balance = TimeOffBalance(
            email="test@company.com",
            number_of_days_left=20
        )

        assert balance.email == "test@company.com"
        assert balance.number_of_days_left == 20

    def test_negative_days_allowed(self):
        """Test that model allows negative balance (edge case)."""
        balance = TimeOffBalance(
            email="test@company.com",
            number_of_days_left=-5
        )
        assert balance.number_of_days_left == -5
