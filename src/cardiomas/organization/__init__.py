from cardiomas.organization.base import DepartmentHead, WorkerAgent

__all__ = [
    "DepartmentHead",
    "WorkerAgent",
    "OrganizationConfig",
    "OrganizationHead",
    "OrganizationRunResult",
    "build_default_organization",
    "resolve_organization_config",
]


def __getattr__(name: str):
    if name in {
        "OrganizationConfig",
        "OrganizationHead",
        "OrganizationRunResult",
        "build_default_organization",
        "resolve_organization_config",
    }:
        from cardiomas.organization.config import OrganizationConfig, resolve_organization_config
        from cardiomas.organization.head import OrganizationHead, OrganizationRunResult, build_default_organization

        return {
            "OrganizationConfig": OrganizationConfig,
            "OrganizationHead": OrganizationHead,
            "OrganizationRunResult": OrganizationRunResult,
            "build_default_organization": build_default_organization,
            "resolve_organization_config": resolve_organization_config,
        }[name]
    raise AttributeError(name)
