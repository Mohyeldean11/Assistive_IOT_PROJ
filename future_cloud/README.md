# Future cloud integration — disabled

No cloud SDK is imported and no cloud connection is made in the current project.

The legacy Azure credential was removed. Rotate/revoke it before any future experiment. Future integration should use environment variables or a managed secret store and should be developed as an optional adapter behind a disabled feature flag.
