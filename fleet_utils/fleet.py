from typing import Optional
import random

class Agent:
    """
    Rapresent a single agent, like an AMR in the system.
    Mantains infos on initial position, goal and current state
    """

    def __init__(self, id: int, start: int, goal: Optional[int], start_t: int):
        """
        Args:
            id:    id of agent
            start: starting node in the graph
            goal:  goal node in the graph, None if the agent doesn't have a goal assigned yet
        """
        self.id = id
        self.start = start
        self.goal = goal
        self.state = "busy" if goal is not None else "idle"
        self.start_t = start_t


class Fleet:
    """
    Rapresent the whole set of agents through a dictionary and provides method for get agent and update agents.
    """

    def __init__(self, ids: list, starts: list, goals: list, start_ts: list):
        """
        Built a fleet creating an Agent object with the infos: (id, start, goal).

        Args:
            ids:    list ids agents,        
            starts: list starting node of agents,        
            goals:  list goal node of agents (or None), 
            starts_t: list of time of starting  
        """
        self.agents = {}
        for aid, s, g, t in zip(ids, starts, goals, start_ts):
            self.agents[aid] = Agent(id=aid, start=s, goal=g, start_t=t)

    def num_agents(self) -> int:
        """Return number of agents in the fleet"""
        return len(self.agents)

    def get_agent(self, agent_id: int) -> Agent:
        """
        Return an object agent that corrispondend to the id required.

        Args:
            agent_id: id agents to retrieve
        Returns:
            object agent correspondent
        """
        return self.agents[agent_id]

    def update_goal(self, agent_id: int, goal: Optional[int]) -> None:
        """
        Update goal of an specified agent from agent_id

        Args:
            agent_id: agent id to update
            goal:     new node or None
        """
        self.agents[agent_id].assign_goal(goal)

    def idle_agents(self) -> list:
        """Return the list of idle agents."""
        return [a for a in self.agents.values() if a.state == "idle"]

    def busy_agents(self) -> list:
        """"Return the list of busy agents (goal assigned)"""
        return [a for a in self.agents.values() if a.state == "busy"]

    def __repr__(self) -> str:
        lines = [f"Fleet ({self.num_agents()} agents):"]
        for agent in self.agents.values():
            lines.append(f"  {agent}")
        return "\n".join(lines)
    
