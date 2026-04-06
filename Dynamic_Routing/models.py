from typing import Dict, List, Optional, Union
from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field, Json


class Package(BaseModel):
    id: str
    destination: str
    deadline: int
    status: str


class Truck(BaseModel):
    id: str
    current_location: str
    route_order: List[str]
    assigned_packages: List[str]
    fuel: float = 100.0
    fuel_capacity: float = 100.0


class EventAlert(BaseModel):
    description: str
    blocked_edges: List[List[str]]
    traffic_delays: Dict[str, int]


class RouteUpdate(BaseModel):
    truck_id: str
    new_route_order: List[str]


class LoadTransfer(BaseModel):
    from_truck_id: str
    to_truck_id: str
    package_id: str


class DynamicRouteAction(Action):
    route_updates: Optional[Union[List[RouteUpdate], Json[List[RouteUpdate]]]] = Field(
        default=None,
        description="One RouteUpdate per truck with truck_id and new_route_order."
    )
    load_transfers: Optional[Union[List[LoadTransfer], Json[List[LoadTransfer]]]] = Field(
        default=None,
        description="Transfer packages between trucks at the same location."
    )


class DynamicRouteObservation(Observation):
    time_step: int = 0
    trucks: List[Truck] = Field(default_factory=list)
    packages: List[Package] = Field(default_factory=list)
    event: Optional[EventAlert] = None
    distances: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    fuel_stations: List[str] = Field(default_factory=list)