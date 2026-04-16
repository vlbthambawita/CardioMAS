from cardiomas.organization.base import DepartmentHead, WorkerAgent

__all__ = ["DepartmentHead", "WorkerAgent", "OrganizationHead", "OrganizationRunResult", "build_default_organization"]


def __getattr__(name: str):
    if name in {"OrganizationHead", "OrganizationRunResult", "build_default_organization"}:
        from cardiomas.organization.head import OrganizationHead, OrganizationRunResult, build_default_organization

        return {
            "OrganizationHead": OrganizationHead,
            "OrganizationRunResult": OrganizationRunResult,
            "build_default_organization": build_default_organization,
        }[name]
    raise AttributeError(name)
