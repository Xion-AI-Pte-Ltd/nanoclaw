# Direct Python runtime

Build `container/Dockerfile.script`, not the existing agent Dockerfile. This image
has a distinct entry point and image tag; no agent SDK or repair loop is started.
The existing NanoClaw agent image is unchanged.

The Marathon host stages and verifies one reviewed retained Python script and
provides a context file. The entry point verifies the checksum again and calls
`run(context)`. It writes an atomic, bounded JSON receipt. Scripts may explicitly
call supported AI extraction and business services through their existing clients.

Scripts can use `from appdefs import sync_requests`. This instantiates the existing
`CountoCookieSession`, version 0.1.1, with its normal deployment-provided environment.
This module adds no authentication behavior. Verify the configured internal service
access and owner permissions from the real container before enabling writes.
The image defaults to the shared client's backend `DEV_MODE=REMOTE` configuration;
this depends on the existing secure-route server-to-server integration and does
not grant permissions or bypass a rejected service call.

Return `status`, `summary`, `effects` (including partial owner IDs after failures)
and, for completion, `verification: {"passed": true}`. The runner attaches the
execution identity. Exceptions produce an uncertain outcome without retrying or
repairing code. Raw script/SDK output is discarded; return sanitized error details
in the explicit receipt. Never put credentials in context or receipt fields.

`tests/test_approved_script_runtime.py` runs actual Python subprocesses, including a
shared-client extraction/save/readback sequence against a local contract fixture.
That test does not establish deployed service authorization or real Vault success.

## Dev release

Merge approved-script changes into `dev` after PR review. The existing Node CI
also checks PRs targeting `dev`. Do not merge dev-only work into `main`: the
existing production trigger on that branch rebuilds the agent image.

Build the script image manually from the reviewed dev commit using
`cloudbuild_script_dev.yaml`, with project `vault-dev-270023` and an explicit
`_IMAGE_TAG=dev-<commit-sha>`. This build runs all Python runtime tests through
the existing authenticated package index before building and publishing only
`nanoclaw-script`. It does not build, pull or replace `nanoclaw-agent`.

Configure Marathon with the resulting immutable image digest, initially leaving
approved-script execution disabled. The existing Marathon manual dev deployment
remains the deployment path; a separate NanoClaw deployment trigger is not needed
for this pilot. Production promotion is a separate reviewed release after the
existing PDF flow and new saved-output flow pass dev acceptance.
