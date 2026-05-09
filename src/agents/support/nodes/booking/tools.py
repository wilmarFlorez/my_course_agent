from langchain.tools import tool


@tool(
    "booking_appoinment",
    description="book a medical appoinment for a given date, time, doctor an patient",
)
def booking_appoinment(date: str, time: str, doctor: str, patient: str) -> str:
    # TODO: implement booking logic
    return f"Appoinment booked for {date} at {time} with {doctor} for {patient}!"


@tool(
    "get_appoinment_availability",
    description=(
        "get the availability of a medical appoinment for a given date, time and doctor"
    ),
)
def get_appoinment_availability(date: str, time: str, doctor: str) -> str:
    # TODO: Implement the vailability Logic
    return """\
The availability slots for the {doctor} are:
- Monday: 10:00 - 15:00
- Wednesday: 10:00-15:00
- Thursday: 10:00-15:00
- Friday: 10:00-12:00
"""


tools = [booking_appoinment, get_appoinment_availability]
