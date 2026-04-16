from __future__ import annotations

from cardiomas.shared.messages import DepartmentReport, TaskMessage


class WorkerAgent:
    """Small, inspectable worker abstraction for organization-style workflows."""

    def __init__(self, name: str, department_name: str) -> None:
        self.name = name
        self.department_name = department_name

    def execute(self, message: TaskMessage) -> DepartmentReport:
        raise NotImplementedError


class DepartmentHead:
    """Routes organization-level tasks to workers inside one department."""

    def __init__(
        self,
        name: str,
        department_name: str,
        workers: list[WorkerAgent],
        input_contract: str,
        output_contract: str,
    ) -> None:
        if not workers:
            raise ValueError("DepartmentHead requires at least one worker")
        self.name = name
        self.department_name = department_name
        self.workers = workers
        self.input_contract = input_contract
        self.output_contract = output_contract

    def handle_task(self, message: TaskMessage) -> DepartmentReport:
        worker = self.workers[0]
        return worker.execute(message)
