# CLOUD INTEGRATION IS INTENTIONALLY DISABLED.
#
# This file is not imported anywhere in the local application.
# Do not add credentials here. When a future cloud trial is approved, implement a
# separate adapter that reads secrets from environment variables or a secret store.
#
# Example future interface only:
#
# class CloudAdapter:
#     def send(self, payload: dict) -> bool:
#         raise NotImplementedError
#
# No Azure, AWS, or other cloud SDK is required by the current local prototype.
