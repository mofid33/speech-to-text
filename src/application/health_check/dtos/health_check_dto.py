from pydantic import BaseModel


class HealthCheckResponseDto(BaseModel):
    status: str
    service: str
    version: str
    environment: str


class ReadinessResponseDto(BaseModel):
    status: str
    database: str