"""Read-only external incident sources."""

from .github import GitHubIssueProvider, GitHubSourceError

__all__ = ["GitHubIssueProvider", "GitHubSourceError"]
