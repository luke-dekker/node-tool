"""Autoresearch subsystem — textual graph format, mutation ops, ledger.

All modules here are free of heavy imports; the MutatorNode depends only on
this subpackage (not on pytorch), so textual-graph introspection works even
when torch isn't installed.
"""
