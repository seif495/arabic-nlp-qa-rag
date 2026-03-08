# Ticket Board (Markdown)

This folder is the canonical local ticket board for the project.

Structure:

- `docs/tickets/ms1/`
- `docs/tickets/ms2/`
- `docs/tickets/ms3/`

Each milestone folder contains exactly four files:

- `roadmap.md`: milestone-level implementation roadmap and dependency order.
- `alice.md`: tickets assigned to Alice.
- `bob.md`: tickets assigned to Bob.
- `charly.md`: tickets assigned to Charly.

Ticket IDs should remain stable, milestone-scoped, and easy to grep.
Use this format:

`MS<milestone>-<stream>-<index>`

Example: `MS1-INGEST-01`
