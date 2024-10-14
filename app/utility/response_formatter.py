def create_response(status: str, message: str = None, data: dict = None):
    response = {
        "status": status,
        "message": message or "",
        "data": data or {}
    }
    return response
